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

use std::fmt;

use crate::{
    bind::{BindError, BoundStorage, PendingLayer},
    config::K3Config,
    layer::{
        ExpertFetchError, ExpertSource, KdaState, LayerState, LayerWeights, Matrix, MlaCache,
        decoder_layer,
    },
    ops::{attn_res, rmsnorm},
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

    /// Logits for every position of `ids`, recomputed from an empty state,
    /// sourcing each decoder layer from this ring instead of a fully resident
    /// [`crate::model::Model`]. The `embed`/`final_norm`/`lm_head`/`out_res`
    /// tensors are not part of the ring -- they are loaded once, up front, via
    /// [`crate::bind::BoundStorage::load_top_level`], since unlike the trunk
    /// they are needed for every position in one pass, not once per layer.
    ///
    /// Mirrors [`crate::model::Model::forward`]'s full-recompute path exactly
    /// (`Positions::All`, no shared KDA slot); the incremental [`crate::model::Session`]
    /// path is not ported to stream through a ring.
    ///
    /// # Errors
    ///
    /// Returns [`TrunkForwardError::Bind`] if a layer's tensors cannot be read,
    /// or [`TrunkForwardError::Expert`] if a streamed expert cannot be fetched.
    pub fn forward(
        &mut self,
        index: &SafeTensorIndex,
        config: &K3Config,
        top: &TopLevelWeights<'_>,
        ids: &[u32],
        experts: &mut dyn ExpertSource,
    ) -> Result<Vec<f32>, TrunkForwardError> {
        let e = config.hidden_size;
        let vocab = config.vocab_size;
        let t = ids.len();

        let mut h = vec![0.0_f32; t * e];
        for (row, &id) in h.chunks_exact_mut(e).zip(ids) {
            top.embed.row_into(row, id as usize, e);
        }

        let mut states: Vec<LayerState> = (0..config.num_hidden_layers)
            .map(|layer| {
                if config.is_mla(layer) {
                    LayerState::Mla(MlaCache::new(config, t))
                } else {
                    LayerState::Kda(KdaState::new(config))
                }
            })
            .collect();

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let last_layer = config.num_hidden_layers - 1;
        for (layer, state) in states.iter_mut().enumerate() {
            // Prefetch the next layer BEFORE binding this one: `bind`'s returned
            // weights borrow `&mut self` for exactly this iteration, so nothing
            // can call `prefetch` again (also `&mut self`) until that borrow's
            // last use, the `decoder_layer` call below, ends. See the module
            // docs for why this ordering still gets real overlap.
            if layer < last_layer {
                self.prefetch(index, config, layer + 1)
                    .map_err(TrunkForwardError::Bind)?;
            }
            let weights = self
                .bind(index, config, layer)
                .map_err(TrunkForwardError::Bind)?;
            decoder_layer(
                &mut h,
                &mut blocks,
                &weights,
                config,
                layer,
                t,
                state,
                0,
                experts,
            )
            .map_err(TrunkForwardError::Expert)?;
        }

        let mut logits = vec![0.0_f32; t * vocab];
        let mut src = vec![0.0_f32; (blocks.len() + 1) * e];
        let mut normed = vec![0.0_f32; e];
        for (row, step) in logits.chunks_exact_mut(vocab).zip(0..t) {
            let ht = &mut h[step * e..(step + 1) * e];
            if let Some((norm, proj)) = top.out_res {
                let fold: Vec<f32> = norm.iter().zip(proj).map(|(&n, &p)| n * p).collect();
                for (b, block) in blocks.iter().enumerate() {
                    src[b * e..(b + 1) * e].copy_from_slice(&block[step * e..(step + 1) * e]);
                }
                src[blocks.len() * e..].copy_from_slice(ht);
                attn_res(ht, &src, &fold, blocks.len() + 1, e, config.rms_norm_eps);
            }
            rmsnorm(&mut normed, ht, top.final_norm, config.rms_norm_eps);
            top.lm_head.mul(row, &normed, e, vocab);
        }
        Ok(logits)
    }
}

/// The tensors [`TrunkRing::forward`] needs outside the layer stack -- loaded
/// once via [`crate::bind::BoundStorage::load_top_level`], not part of the ring
/// since, unlike the trunk, every one of them is needed for every position in
/// one pass rather than once per layer.
pub struct TopLevelWeights<'a> {
    /// `[vocab][hidden]`, gathered one row per token.
    pub embed: Matrix<'a>,
    pub lm_head: Matrix<'a>,
    pub final_norm: &'a [f32],
    /// `output_attn_res_{norm,proj}`, the one aggregator outside the layers.
    pub out_res: Option<(&'a [f32], &'a [f32])>,
}

/// Either half of what [`TrunkRing::forward`] can fail on: a layer's tensors
/// could not be read, or a streamed expert could not be fetched.
#[derive(Debug)]
pub enum TrunkForwardError {
    Bind(BindError),
    Expert(ExpertFetchError),
}

impl fmt::Display for TrunkForwardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind(error) => write!(formatter, "{error}"),
            Self::Expert(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TrunkForwardError {}
