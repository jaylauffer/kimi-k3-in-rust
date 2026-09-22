//! Streams the checkpoint's per-layer trunk weights through a small ring plus a
//! pinned prefix, so RAM is a dial instead of a floor -- the Rust port of
//! `src/io/k3_trunk.c`. Read that file's own header comment for the full design
//! rationale: why streaming costs zero error (the bytes are the checkpoint's own
//! bytes), why a cyclic LRU would be the worst possible policy for this access
//! pattern (a pinned prefix plus a ring is not LRU and does not have that
//! pathology), and why the fixed, always-forward layer walk (every token visits
//! layer 0, 1, ..., `num_hidden_layers - 1`) makes one-layer-ahead prefetch safe
//! without predicting anything.
//!
//! Prefetch is the loadngo proactor's own async submit/wait split
//! (`bind::BoundStorage::submit_layer`/`absorb_layer`), never a reader thread:
//! [`TrunkRing::prefetch`] submits a whole layer's reads and returns immediately,
//! and [`TrunkRing::bind`] only waits for them when that layer is actually
//! needed. Call `prefetch(L + 1)` before `bind(L)`, not after: `bind`'s returned
//! [`LayerWeights`] borrows from `self`, so nothing can call `prefetch` again
//! (which needs `&mut self`) while that borrow is still alive for the compute
//! pass. Prefetching the next layer first, then binding and computing on the
//! current one, gets the overlap without fighting that -- the read is already
//! submitted to the operating system by the time `bind`'s wait (if any) and the
//! caller's compute run, so both proceed concurrently with it regardless.
//!
//! Not ported from the C engine: budget-based automatic sizing of the pinned
//! prefix (`k3_trunk_open`'s `budget_bytes`) and preallocated, uniformly-sized
//! ring slots for `O_DIRECT`. A caller here states `pin_layers`/`ring_slots`
//! directly, and ring slots are ordinary per-layer allocations, freed and
//! replaced as layers stream through -- consistent with the rest of this port
//! not yet implementing direct I/O or aligned slot memory (see `docs/RUST_PORT.md`).

use crate::{
    bind::{BindError, BoundStorage, PendingLayer},
    config::K3Config,
    layer::LayerWeights,
    safetensors::SafeTensorIndex,
};

enum SlotState {
    /// A read for this layer was submitted and has not been waited on yet.
    InFlight(PendingLayer),
    /// This layer's weights are resident and ready to bind.
    Loaded(BoundStorage),
}

struct RingSlot {
    layer: usize,
    state: SlotState,
}

/// Streams `config.num_hidden_layers` decoder layers' trunk weights: a pinned
/// prefix held resident for the ring's whole lifetime, plus the rest streamed
/// through a small ring of slots walked in the fixed layer order every token
/// visits. See the module docs for the full design.
pub struct TrunkRing {
    /// Layers `0..pin.len()`, each pinned resident for this ring's whole lifetime.
    pin: Vec<BoundStorage>,
    /// At most `nslot` layers streamed at once, oldest evicted first since the
    /// fixed forward walk never revisits a layer once passed.
    ring: Vec<RingSlot>,
    nslot: usize,
    hits: u64,
    misses: u64,
}

impl TrunkRing {
    /// Opens a ring over `index`: layers `0..pin_layers` (clamped to
    /// `config.num_hidden_layers`) are read now and pinned resident for the
    /// ring's whole lifetime; the rest stream through `ring_slots` slots
    /// (`ring_slots.max(1)` -- a ring needs at least one slot to hold anything).
    /// Two slots is what actually enables overlap (one holds the layer just
    /// bound, the other a prefetch in flight for the next one); one slot is
    /// still correct, just serial, matching `k3_trunk_open`'s own
    /// budget-exhausted fallback to one slot.
    ///
    /// # Errors
    ///
    /// Returns [`BindError`] if a pinned layer's tensors cannot be read.
    pub fn open(
        index: &SafeTensorIndex,
        config: &K3Config,
        pin_layers: usize,
        ring_slots: usize,
    ) -> Result<Self, BindError> {
        let pin_layers = pin_layers.min(config.num_hidden_layers);
        let mut pin = Vec::with_capacity(pin_layers);
        for layer in 0..pin_layers {
            let mut storage = BoundStorage::new();
            storage.load_layer(index, config, layer)?;
            pin.push(storage);
        }
        Ok(Self {
            pin,
            ring: Vec::new(),
            nslot: ring_slots.max(1),
            hits: 0,
            misses: 0,
        })
    }

    /// True when `layer` is pinned resident for this ring's whole lifetime.
    #[must_use]
    pub fn is_pinned(&self, layer: usize) -> bool {
        layer < self.pin.len()
    }

    fn ring_position(&self, layer: usize) -> Option<usize> {
        self.ring.iter().position(|slot| slot.layer == layer)
    }

    /// Starts an asynchronous read of `layer`'s trunk weights into a ring slot,
    /// if it is not already pinned, resident, or already in flight. A no-op for
    /// a pinned or already-loaded layer, matching `k3_trunk_prefetch`.
    ///
    /// Call this for layer `L + 1` right before [`Self::bind`]ing layer `L` --
    /// see the module docs for why that ordering, not the reverse, is what
    /// makes the read overlap with compute.
    ///
    /// # Errors
    ///
    /// Returns [`BindError`] if `layer` is out of range or its tensors cannot be
    /// planned (a name is missing, or one is too large for one unchunked read).
    pub fn prefetch(
        &mut self,
        index: &SafeTensorIndex,
        config: &K3Config,
        layer: usize,
    ) -> Result<(), BindError> {
        if self.is_pinned(layer) || self.ring_position(layer).is_some() {
            return Ok(());
        }
        if self.ring.len() >= self.nslot {
            // Every slot is busy (loaded or itself in flight). One spare slot is
            // what this ring promises, matching the C engine's "one asynchronous
            // reader owns one spare ring slot" -- a second prefetch request
            // beyond that is simply not started early; `bind` still loads the
            // layer correctly when it is actually needed, just without overlap
            // that time.
            return Ok(());
        }
        let pending = BoundStorage::submit_layer(index, config, layer)?;
        self.ring.push(RingSlot {
            layer,
            state: SlotState::InFlight(pending),
        });
        Ok(())
    }

    /// Makes `layer` resident -- waiting for an in-flight prefetch, loading it
    /// synchronously if none was started, or returning it directly if pinned --
    /// and returns its weights. Evicts any ring-resident layer behind `layer`:
    /// the fixed forward walk never revisits it.
    ///
    /// # Errors
    ///
    /// Returns [`BindError`] if `layer`'s tensors cannot be read.
    ///
    /// # Panics
    ///
    /// Never in practice: after this method makes `layer` resident, it always
    /// remains in the ring, since eviction only ever removes layers strictly
    /// behind it.
    pub fn bind(
        &mut self,
        index: &SafeTensorIndex,
        config: &K3Config,
        layer: usize,
    ) -> Result<LayerWeights<'_>, BindError> {
        if self.is_pinned(layer) {
            return self.pin[layer].layer_weights(config, layer);
        }

        if let Some(position) = self.ring_position(layer) {
            // Found via an earlier prefetch (still in flight, or already fully
            // absorbed by a previous bind of this same layer) -- either way, no
            // cold synchronous load was needed, so this is a hit.
            if matches!(self.ring[position].state, SlotState::InFlight(_)) {
                let SlotState::InFlight(pending) = std::mem::replace(
                    &mut self.ring[position].state,
                    SlotState::Loaded(BoundStorage::new()),
                ) else {
                    unreachable!("just matched InFlight");
                };
                let mut storage = BoundStorage::new();
                storage.absorb_layer(index, pending)?;
                self.ring[position].state = SlotState::Loaded(storage);
            }
            self.hits += 1;
        } else {
            // Never prefetched: a genuine cold miss, loaded synchronously now.
            let mut storage = BoundStorage::new();
            storage.load_layer(index, config, layer)?;
            if self.ring.len() >= self.nslot {
                self.ring.remove(0);
            }
            self.ring.push(RingSlot {
                layer,
                state: SlotState::Loaded(storage),
            });
            self.misses += 1;
        }

        self.ring.retain(|slot| slot.layer >= layer);
        let position = self
            .ring_position(layer)
            .expect("just loaded or already resident");
        let SlotState::Loaded(storage) = &self.ring[position].state else {
            unreachable!("bind always resolves InFlight to Loaded before this point")
        };
        storage.layer_weights(config, layer)
    }

    /// How many non-pinned [`Self::bind`] calls found their layer already
    /// prefetched (a hit -- whether or not the read had actually finished yet,
    /// since either way no cold synchronous load was needed) versus never
    /// prefetched at all (a miss, loaded synchronously on the spot). Binding a
    /// pinned layer touches neither counter.
    #[must_use]
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}
