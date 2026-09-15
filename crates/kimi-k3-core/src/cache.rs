//! Bounded LRU cache for streamed routed experts.
//!
//! The cache stores packed MXFP4 bytes, never widened float weights. The compact form is
//! what the eventual kernel consumes, and its fixed slots make the memory budget an
//! enforceable upper bound rather than an estimate.

use std::fmt;

use crate::{
    expert::{ExpertError, ExpertRef, QuantizedMatrix},
    layer::{ExpertFetchError, ExpertSource, Mxfp4Matrix, PackedExpert},
    safetensors::SafeTensorIndex,
};

const DIRECT_ALIGNMENT: usize = 4096;
const DIRECT_READ_SLACK: usize = DIRECT_ALIGNMENT * 2;

/// One key in an expert-cache access trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertKey {
    /// Zero-based transformer layer.
    pub layer: usize,
    /// Zero-based routed expert.
    pub expert: usize,
}

/// A currently resident expert returned by [`ExpertCache::get`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentExpert {
    slot: usize,
    key: ExpertKey,
    layout: ExpertRef,
}

impl ResidentExpert {
    /// Returns the resolved matrix layout for the cached expert.
    #[must_use]
    pub const fn layout(&self) -> &ExpertRef {
        &self.layout
    }
}

/// Request and I/O counters for the current measurement window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    /// Direct `get` requests served without a new direct load.
    pub hits: u64,
    /// Direct `get` requests that caused a load.
    pub misses: u64,
    /// Resident slots displaced to make room for another expert.
    pub evictions: u64,
    /// Expert payload bytes fetched from storage.
    pub bytes_read: u64,
    /// Experts fetched by [`ExpertCache::prefetch_many`].
    pub prefetch_reads: u64,
}

/// Fixed-capacity, LRU-replaced packed-expert storage.
#[derive(Debug)]
pub struct ExpertCache {
    layers: usize,
    experts: usize,
    slot_stride: usize,
    slots: Vec<Slot>,
    slot_of: Vec<Option<usize>>,
    clock: u64,
    stats: CacheStats,
    histogram: Vec<u32>,
    trace: Vec<ExpertKey>,
}

impl ExpertCache {
    /// Creates a cache sized from a known routed-expert probe.
    ///
    /// Slots reserve the C implementation's `O_DIRECT` alignment slack and are rounded to
    /// 4096 bytes. The current Rust I/O path is buffered, but retaining this stride
    /// keeps capacity calculations compatible with the direct-I/O design it will use.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError`] if dimensions are invalid, overflow addressable memory, or
    /// `budget_bytes` cannot hold at least `top_k + 1` whole expert slots.
    pub fn new(
        layers: usize,
        experts: usize,
        top_k: usize,
        budget_bytes: usize,
        probe: &ExpertRef,
    ) -> Result<Self, CacheError> {
        if layers == 0 || experts == 0 || top_k == 0 || top_k > experts {
            return Err(CacheError::InvalidDimensions {
                layers,
                experts,
                top_k,
            });
        }
        let slot_stride = probe
            .nbytes
            .checked_add(DIRECT_READ_SLACK)
            .and_then(round_up_to_direct_alignment)
            .ok_or(CacheError::CapacityOverflow)?;
        let slot_count = budget_bytes / slot_stride;
        if slot_count < top_k + 1 {
            return Err(CacheError::InsufficientBudget {
                budget_bytes,
                slot_stride,
                slot_count,
                required_slots: top_k + 1,
            });
        }
        let keys = layers
            .checked_mul(experts)
            .ok_or(CacheError::CapacityOverflow)?;
        let allocation = slot_count
            .checked_mul(slot_stride)
            .ok_or(CacheError::CapacityOverflow)?;
        if allocation > budget_bytes {
            return Err(CacheError::CapacityOverflow);
        }
        let slots = (0..slot_count)
            .map(|_| Slot::new(slot_stride))
            .collect::<Vec<_>>();
        Ok(Self {
            layers,
            experts,
            slot_stride,
            slots,
            slot_of: vec![None; keys],
            clock: 0,
            stats: CacheStats::default(),
            histogram: vec![0; keys],
            trace: Vec::new(),
        })
    }

    /// Returns the number of fixed cache slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Returns the reserved byte stride per slot, including direct-I/O alignment slack.
    #[must_use]
    pub const fn slot_stride(&self) -> usize {
        self.slot_stride
    }

    /// Returns counters from the current measurement window.
    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Returns one request counter per `(layer, expert)` key in row-major order.
    #[must_use]
    pub fn histogram(&self) -> &[u32] {
        &self.histogram
    }

    /// Returns every direct `get` request in model order.
    #[must_use]
    pub fn trace(&self) -> &[ExpertKey] {
        &self.trace
    }

    /// Clears cache-window counters without evicting data or discarding the trace.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// Returns whether one expert is already resident, without changing its LRU age.
    #[must_use]
    pub fn is_resident(&self, layer: usize, expert: usize) -> bool {
        self.key_index(layer, expert)
            .ok()
            .and_then(|key| self.slot_of[key])
            .is_some()
    }

    /// Pins or unpins a resident expert. A missing expert is left unchanged.
    ///
    /// Returns `true` when the requested expert was resident and its pin was updated.
    pub fn pin(&mut self, layer: usize, expert: usize, pinned: bool) -> bool {
        let Some(slot) = self
            .key_index(layer, expert)
            .ok()
            .and_then(|key| self.slot_of[key])
        else {
            return false;
        };
        self.slots[slot].pinned = pinned;
        true
    }

    /// Gets one expert, recording the request, histogram, and access trace.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError`] for an invalid key, a bad checkpoint expert, an I/O
    /// failure, or when every slot is pinned.
    pub fn get(
        &mut self,
        index: &SafeTensorIndex,
        layer: usize,
        expert: usize,
    ) -> Result<ResidentExpert, CacheError> {
        let key_index = self.key_index(layer, expert)?;
        self.histogram[key_index] = self.histogram[key_index].saturating_add(1);
        self.trace.push(ExpertKey { layer, expert });
        self.admit(index, layer, expert)
    }

    /// Loads one expert without recording a model request.
    ///
    /// This follows the C cache's single-prefetch accounting: it still counts a cache
    /// hit or miss, unlike the batched prefetch path.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError`] under the same conditions as [`Self::get`].
    pub fn prefetch(
        &mut self,
        index: &SafeTensorIndex,
        layer: usize,
        expert: usize,
    ) -> Result<ResidentExpert, CacheError> {
        self.key_index(layer, expert)?;
        self.admit(index, layer, expert)
    }

    /// Warms a unique batch of experts and returns the count newly loaded.
    ///
    /// Mirrors the C cache's three phases. Resolution and slot reservation are serial,
    /// because choosing a slot reads and updates the LRU, and every reserved slot stays
    /// pinned until publication so no two experts in one batch can share storage. All
    /// reads then go to the proactor as one batch, each straight into its slot. Finally
    /// the experts are published. Already-resident and duplicate IDs are skipped
    /// without changing demand hit/miss counters, matching the C batch path.
    ///
    /// On any error nothing from the batch is published, and reserved slots are left
    /// empty and unpinned.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError`] for invalid keys, failed expert resolution/I/O, or too few
    /// unpinned slots for the batch.
    pub fn prefetch_many(
        &mut self,
        index: &SafeTensorIndex,
        layer: usize,
        experts: &[usize],
    ) -> Result<usize, CacheError> {
        let mut unique = Vec::with_capacity(experts.len());
        for &expert in experts {
            self.key_index(layer, expert)?;
            if !unique.contains(&expert) && !self.is_resident(layer, expert) {
                unique.push(expert);
            }
        }

        let mut reserved: Vec<(usize, ExpertKey, ExpertRef)> = Vec::with_capacity(unique.len());
        for expert in unique {
            let key = ExpertKey { layer, expert };
            match self.reserve(index, key) {
                Ok((slot, layout)) => reserved.push((slot, key, layout)),
                Err(error) => {
                    self.release(&reserved);
                    return Err(error);
                }
            }
        }

        let batch = reserved
            .iter()
            .map(|(slot, _, layout)| (layout, std::mem::take(&mut self.slots[*slot].bytes)))
            .collect();
        let loaded = match ExpertRef::load_batch(index, batch) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.release(&reserved);
                return Err(error.into());
            }
        };

        let count = reserved.len();
        for ((slot, key, layout), bytes) in reserved.into_iter().zip(loaded) {
            self.stats.bytes_read += layout.nbytes as u64;
            self.stats.prefetch_reads += 1;
            self.publish(slot, key, layout, bytes);
        }
        Ok(count)
    }

    /// Borrows the canonical expert bytes for a still-resident handle.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::StaleHandle`] when the slot was evicted after the handle was
    /// returned. Model code must consume a handle before requesting enough other experts
    /// to evict it, just as the C `MoE` path does.
    pub fn bytes(&self, resident: &ResidentExpert) -> Result<&[u8], CacheError> {
        let slot = self
            .slots
            .get(resident.slot)
            .ok_or(CacheError::StaleHandle { key: resident.key })?;
        if slot.key != Some(resident.key) {
            return Err(CacheError::StaleHandle { key: resident.key });
        }
        Ok(&slot.bytes[..resident.layout.nbytes])
    }

    fn admit(
        &mut self,
        index: &SafeTensorIndex,
        layer: usize,
        expert: usize,
    ) -> Result<ResidentExpert, CacheError> {
        let key_index = self.key_index(layer, expert)?;
        let key = ExpertKey { layer, expert };
        if let Some(slot) = self.slot_of[key_index] {
            self.stats.hits += 1;
            self.touch(slot);
            let layout = self.slots[slot]
                .layout
                .clone()
                .ok_or(CacheError::StaleHandle { key })?;
            return Ok(ResidentExpert { slot, key, layout });
        }
        self.stats.misses += 1;

        let (slot, layout) = self.reserve(index, key)?;
        let buffer = std::mem::take(&mut self.slots[slot].bytes);
        let bytes = match ExpertRef::load_batch(index, vec![(&layout, buffer)]) {
            Ok(mut loaded) => loaded.pop().unwrap_or_default(),
            Err(error) => {
                self.release(&[(slot, key, layout)]);
                return Err(error.into());
            }
        };
        self.stats.bytes_read += layout.nbytes as u64;
        self.publish(slot, key, layout.clone(), bytes);
        Ok(ResidentExpert { slot, key, layout })
    }

    /// Resolves `key` and claims a slot for it: evicted, emptied, and pinned so a later
    /// reservation in the same batch cannot claim it again.
    fn reserve(
        &mut self,
        index: &SafeTensorIndex,
        key: ExpertKey,
    ) -> Result<(usize, ExpertRef), CacheError> {
        let layout = ExpertRef::resolve(index, key.layer, key.expert)?;
        if layout.nbytes > self.slot_stride {
            return Err(CacheError::ExpertTooLarge {
                key,
                expert_bytes: layout.nbytes,
                slot_stride: self.slot_stride,
            });
        }
        let slot = self.pick_victim().ok_or(CacheError::AllPinned)?;
        self.evict(slot);
        self.slots[slot].pinned = true;
        Ok((slot, layout))
    }

    /// Returns reserved slots to the pool after a failed load. They stay empty.
    fn release(&mut self, reserved: &[(usize, ExpertKey, ExpertRef)]) {
        for &(slot, _, _) in reserved {
            self.slots[slot].pinned = false;
        }
    }

    /// Makes a loaded expert resident in its reserved slot.
    fn publish(&mut self, slot: usize, key: ExpertKey, layout: ExpertRef, bytes: Vec<u8>) {
        let key_index = key.layer * self.experts + key.expert;
        let entry = &mut self.slots[slot];
        entry.bytes = bytes;
        entry.key = Some(key);
        entry.layout = Some(layout);
        entry.pinned = false;
        self.slot_of[key_index] = Some(slot);
        self.touch(slot);
    }

    fn key_index(&self, layer: usize, expert: usize) -> Result<usize, CacheError> {
        if layer >= self.layers || expert >= self.experts {
            return Err(CacheError::InvalidKey {
                layer,
                expert,
                layers: self.layers,
                experts: self.experts,
            });
        }
        Ok(layer * self.experts + expert)
    }

    fn pick_victim(&self) -> Option<usize> {
        // An empty slot can already be reserved (pinned) by the batch in progress.
        if let Some((slot, _)) = self
            .slots
            .iter()
            .enumerate()
            .find(|(_, slot)| slot.key.is_none() && !slot.pinned)
        {
            return Some(slot);
        }
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.pinned)
            .min_by_key(|(_, slot)| slot.used_at)
            .map(|(slot, _)| slot)
    }

    fn evict(&mut self, slot: usize) {
        if let Some(old_key) = self.slots[slot].key.take() {
            let old_index = old_key.layer * self.experts + old_key.expert;
            self.slot_of[old_index] = None;
            self.stats.evictions += 1;
        }
        self.slots[slot].layout = None;
        self.slots[slot].pinned = false;
    }

    fn touch(&mut self, slot: usize) {
        self.clock = self.clock.wrapping_add(1);
        self.slots[slot].used_at = self.clock;
    }
}

/// Cache construction, lookup, or streaming failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CacheError {
    /// Cache dimensions cannot describe a valid routed-expert domain.
    InvalidDimensions {
        layers: usize,
        experts: usize,
        top_k: usize,
    },
    /// A request did not name a configured layer and expert.
    InvalidKey {
        layer: usize,
        expert: usize,
        layers: usize,
        experts: usize,
    },
    /// Cache dimension or allocation arithmetic overflowed `usize`.
    CapacityOverflow,
    /// The budget cannot hold the required one-token working set.
    InsufficientBudget {
        budget_bytes: usize,
        slot_stride: usize,
        slot_count: usize,
        required_slots: usize,
    },
    /// No unpinned slot can admit another expert.
    AllPinned,
    /// A checkpoint expert did not fit the probe-sized cache slot.
    ExpertTooLarge {
        key: ExpertKey,
        expert_bytes: usize,
        slot_stride: usize,
    },
    /// A previously returned handle was evicted before its bytes were consumed.
    StaleHandle { key: ExpertKey },
    /// Routed-expert validation or I/O failure.
    Expert(ExpertError),
}

impl fmt::Display for CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                layers,
                experts,
                top_k,
            } => write!(
                formatter,
                "invalid cache dimensions: {layers} layers, {experts} experts, top-{top_k}"
            ),
            Self::InvalidKey {
                layer,
                expert,
                layers,
                experts,
            } => write!(
                formatter,
                "layer {layer} expert {expert} is outside {layers} layers x {experts} experts"
            ),
            Self::CapacityOverflow => write!(formatter, "expert-cache capacity overflowed"),
            Self::InsufficientBudget {
                budget_bytes,
                slot_stride,
                slot_count,
                required_slots,
            } => write!(
                formatter,
                "{budget_bytes} byte cache budget yields {slot_count} slots of {slot_stride} bytes; requires {required_slots}"
            ),
            Self::AllPinned => write!(formatter, "every expert-cache slot is pinned"),
            Self::ExpertTooLarge {
                key,
                expert_bytes,
                slot_stride,
            } => write!(
                formatter,
                "layer {} expert {} needs {expert_bytes} bytes but cache slots hold {slot_stride}",
                key.layer, key.expert
            ),
            Self::StaleHandle { key } => write!(
                formatter,
                "layer {} expert {} is no longer resident in its cache slot",
                key.layer, key.expert
            ),
            Self::Expert(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<ExpertError> for CacheError {
    fn from(error: ExpertError) -> Self {
        Self::Expert(error)
    }
}

#[derive(Debug)]
struct Slot {
    key: Option<ExpertKey>,
    bytes: Vec<u8>,
    layout: Option<ExpertRef>,
    used_at: u64,
    pinned: bool,
}

impl Slot {
    fn new(slot_stride: usize) -> Self {
        Self {
            key: None,
            // Reserved, not written: like the C arena, a slot's pages are committed by
            // the first expert read into it rather than by zeroing the whole budget.
            bytes: Vec::with_capacity(slot_stride),
            layout: None,
            used_at: 0,
            pinned: false,
        }
    }
}

fn round_up_to_direct_alignment(bytes: usize) -> Option<usize> {
    bytes
        .checked_add(DIRECT_ALIGNMENT - 1)
        .map(|value| value & !(DIRECT_ALIGNMENT - 1))
}

/// The engine's [`ExpertSource`]: routed experts streamed from the checkpoint shards
/// through this cache, and multiplied straight out of the cached packed bytes.
pub struct CachedExperts<'a> {
    cache: &'a mut ExpertCache,
    index: &'a SafeTensorIndex,
}

impl<'a> CachedExperts<'a> {
    #[must_use]
    pub const fn new(cache: &'a mut ExpertCache, index: &'a SafeTensorIndex) -> Self {
        Self { cache, index }
    }
}

impl ExpertSource for CachedExperts<'_> {
    fn prefetch(&mut self, layer: usize, experts: &[usize]) -> Result<(), ExpertFetchError> {
        // A failed batch is not fatal, exactly as the C `getmany`: every expert is still
        // requested through `expert`, which reads it on demand or reports the failure.
        let _ = self.cache.prefetch_many(self.index, layer, experts);
        Ok(())
    }

    fn expert(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<PackedExpert<'_>, ExpertFetchError> {
        let fetch_error = |error: CacheError| ExpertFetchError {
            layer,
            expert,
            detail: error.to_string(),
        };
        let resident = self
            .cache
            .get(self.index, layer, expert)
            .map_err(fetch_error)?;
        let bytes = self.cache.bytes(&resident).map_err(fetch_error)?;
        Ok(packed_expert(bytes, resident.layout()))
    }
}

/// Views one expert's canonical bytes as its three MXFP4 matrices.
fn packed_expert<'b>(bytes: &'b [u8], layout: &ExpertRef) -> PackedExpert<'b> {
    let view = |matrix: &QuantizedMatrix| Mxfp4Matrix {
        packed: &bytes[matrix.packed_offset..matrix.packed_offset + matrix.packed_bytes],
        scales: &bytes[matrix.scale_offset..matrix.scale_offset + matrix.scale_bytes],
        rows: matrix.rows,
        columns: matrix.packed_columns * 2,
    };
    let [w1, w2, w3] = &layout.matrices;
    PackedExpert {
        w1: view(w1),
        w2: view(w2),
        w3: view(w3),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CacheError, CachedExperts, ExpertCache, packed_expert};
    use crate::{expert::ExpertRef, layer::ExpertSource, safetensors::SafeTensorIndex};

    fn fixture_index() -> SafeTensorIndex {
        let directory =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cache");
        SafeTensorIndex::open(directory).expect("cache fixture index opens")
    }

    fn cache(index: &SafeTensorIndex, slots: usize) -> ExpertCache {
        let probe = ExpertRef::resolve(index, 0, 0).expect("probe expert resolves");
        let stride = (probe.nbytes + 8192).div_ceil(4096) * 4096;
        ExpertCache::new(1, 24, 4, slots * stride, &probe).expect("cache initializes")
    }

    fn direct_bytes(index: &SafeTensorIndex, expert: usize) -> Vec<u8> {
        let layout = ExpertRef::resolve(index, 0, expert).expect("expert resolves");
        let mut bytes = vec![0; layout.nbytes];
        layout
            .load_into(index, &mut bytes)
            .expect("direct expert read");
        bytes
    }

    #[test]
    fn cached_expert_source_serves_the_same_bytes_and_products_as_a_direct_load() {
        let index = fixture_index();
        let mut cache = cache(&index, 5);
        let x: Vec<f32> = (0..128_u16).map(|i| (f32::from(i) * 0.37).sin()).collect();

        // Two passes over 24 experts with 5 slots, some through a batch prefetch, so the
        // views are taken under real eviction pressure.
        for pass in 0..2 {
            for expert in 0..24 {
                let direct = direct_bytes(&index, expert);
                let layout = ExpertRef::resolve(&index, 0, expert).expect("expert resolves");
                let want = packed_expert(&direct, &layout);

                let mut source = CachedExperts::new(&mut cache, &index);
                if (expert + pass) % 3 == 0 {
                    source
                        .prefetch(0, &[expert, (expert + 1) % 24])
                        .expect("prefetch never fails the token");
                }
                let got = source.expert(0, expert).expect("expert fetches");
                for (got, want) in [(got.w1, want.w1), (got.w2, want.w2), (got.w3, want.w3)] {
                    assert_eq!((got.rows, got.columns), (want.rows, want.columns));
                    assert_eq!(got.packed, want.packed, "expert {expert} packed bytes");
                    assert_eq!(got.scales, want.scales, "expert {expert} scales");
                    let mut a = vec![0.0; got.rows];
                    let mut b = vec![0.0; want.rows];
                    got.mul(&mut a, &x[..got.columns]);
                    want.mul(&mut b, &x[..want.columns]);
                    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
                }
            }
        }
    }

    #[test]
    fn cached_expert_source_reports_a_bad_key_instead_of_serving_nothing() {
        let index = fixture_index();
        let mut cache = cache(&index, 5);
        let mut source = CachedExperts::new(&mut cache, &index);
        let error = source.expert(3, 0).expect_err("layer 3 does not exist");
        assert_eq!((error.layer, error.expert), (3, 0));
    }

    #[test]
    fn lru_reads_are_byte_exact_under_eviction_pressure() {
        let index = fixture_index();
        let mut cache = cache(&index, 5);
        assert_eq!(cache.slot_count(), 5);

        for _ in 0..3 {
            for expert in 0..24 {
                let resident = cache.get(&index, 0, expert).expect("expert admits");
                assert_eq!(
                    cache.bytes(&resident).expect("handle stays resident"),
                    direct_bytes(&index, expert)
                );
            }
        }
        assert_eq!(cache.stats().misses, 72);
        assert!(cache.stats().evictions > 0);
        assert_eq!(cache.trace().len(), 72);
        assert!(cache.histogram().iter().all(|&count| count == 3));
    }

    #[test]
    fn batch_prefetch_warms_distinct_experts_without_falsifying_request_stats() {
        let index = fixture_index();
        let mut cache = cache(&index, 5);

        for batch in 0..6 {
            let experts = [batch * 4, batch * 4 + 1, batch * 4 + 2, batch * 4 + 3];
            assert_eq!(
                cache
                    .prefetch_many(&index, 0, &experts)
                    .expect("batch prefetches"),
                4
            );
            for expert in experts {
                let resident = cache
                    .get(&index, 0, expert)
                    .expect("prefetched expert is resident");
                assert_eq!(
                    cache.bytes(&resident).expect("handle stays resident"),
                    direct_bytes(&index, expert)
                );
            }
        }
        assert_eq!(cache.stats().prefetch_reads, 24);
        assert_eq!(cache.stats().hits, 24);
        assert!(cache.stats().prefetch_reads <= cache.stats().hits);
    }

    #[test]
    fn a_batch_into_a_full_cache_gives_every_expert_its_own_slot() {
        // Every slot is full, so each reservation must evict, and no two experts in the
        // batch may be handed the same slot. Aliasing would show up as one expert's
        // bytes under another's key.
        let index = fixture_index();
        let mut cache = cache(&index, 5);
        for expert in 0..5 {
            cache.get(&index, 0, expert).expect("expert admits");
        }

        let batch = [10, 11, 12, 13, 14];
        assert_eq!(
            cache
                .prefetch_many(&index, 0, &batch)
                .expect("batch prefetches"),
            5
        );
        for expert in 0..5 {
            assert!(!cache.is_resident(0, expert), "expert {expert} not evicted");
        }
        for expert in batch {
            assert!(cache.is_resident(0, expert));
            let resident = cache.get(&index, 0, expert).expect("resident expert");
            assert_eq!(
                cache.bytes(&resident).expect("handle stays resident"),
                direct_bytes(&index, expert)
            );
        }
        assert_eq!(cache.stats().evictions, 5);
    }

    #[test]
    fn pinned_working_set_refuses_an_admission_until_a_slot_is_unpinned() {
        let index = fixture_index();
        let mut cache = cache(&index, 5);

        for expert in 0..5 {
            cache.get(&index, 0, expert).expect("expert admits");
            assert!(cache.pin(0, expert, true));
        }
        assert_eq!(cache.get(&index, 0, 5), Err(CacheError::AllPinned));
        assert!(cache.pin(0, 2, false));
        assert!(cache.get(&index, 0, 5).is_ok());
    }
}
