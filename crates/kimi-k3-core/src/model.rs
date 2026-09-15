//! The whole-model forward pass: embedding, the decoder stack, the model-level
//! Attention Residual aggregator, the final norm and the LM head.
//!
//! Ported from the forward in `tests/unit/k3_model.c`, which is the C engine's oracle
//! gate. Three entry points cover the paths that gate exercises:
//!
//! - [`Model::forward`] recomputes a whole sequence from nothing (prefill semantics);
//! - [`Model::forward_shared_kda_slot`] is the same computation with one KDA state slot
//!   reused across layers, the C CLI's low-memory layout, and must be bit-identical;
//! - [`Session`] prefills once and then feeds tokens one at a time, carrying KDA state
//!   and an MLA KV cache.
//!
//! Every entry point takes the [`ExpertSource`] streamed `MoE` layers fetch from; a model
//! with only resident experts passes [`crate::layer::NoStreamedExperts`].

use crate::{
    config::K3Config,
    layer::{
        ExpertFetchError, ExpertSource, KdaState, LayerState, LayerWeights, Matrix, MlaCache,
        decoder_layer,
    },
    ops::{attn_res, rmsnorm},
};

/// Every weight the model reads, borrowed from wherever the checkpoint lives.
#[derive(Clone, Debug)]
pub struct Model<'a> {
    pub config: K3Config,
    /// `[vocab][hidden]`, gathered one row per token.
    pub embed: Matrix<'a>,
    pub lm_head: Matrix<'a>,
    pub final_norm: &'a [f32],
    /// `output_attn_res_{norm,proj}`, the one aggregator outside the layers.
    pub out_res: Option<(&'a [f32], &'a [f32])>,
    pub layers: Vec<LayerWeights<'a>>,
}

/// Incremental decoding state: per-layer attention memory and the next position.
#[derive(Clone, Debug)]
pub struct Session {
    states: Vec<LayerState>,
    position: usize,
    broken: bool,
}

impl Session {
    /// Tokens consumed so far.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }
}

impl Model<'_> {
    /// Logits for every position of `ids`, recomputed from an empty state.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertFetchError`] when a streamed expert cannot be fetched.
    pub fn forward(
        &self,
        ids: &[u32],
        experts: &mut dyn ExpertSource,
    ) -> Result<Vec<f32>, ExpertFetchError> {
        let mut states = self.fresh_states(ids.len());
        self.run(ids, &mut states, 0, Positions::All, None, experts)
    }

    /// [`Model::forward`] with ONE KDA state slot, cleared before each KDA layer. Full
    /// recompute never returns to a layer after its sequence is done, so this must give
    /// bit-identical logits.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertFetchError`] when a streamed expert cannot be fetched.
    pub fn forward_shared_kda_slot(
        &self,
        ids: &[u32],
        experts: &mut dyn ExpertSource,
    ) -> Result<Vec<f32>, ExpertFetchError> {
        let mut states = self.fresh_states(ids.len());
        let mut slot = KdaState::new(&self.config);
        self.run(
            ids,
            &mut states,
            0,
            Positions::All,
            Some(&mut slot),
            experts,
        )
    }

    /// A session able to hold `capacity` positions.
    #[must_use]
    pub fn session(&self, capacity: usize) -> Session {
        Session {
            states: self.fresh_states(capacity),
            position: 0,
            broken: false,
        }
    }

    /// Feeds `ids` after everything the session has already seen and returns the logits
    /// for the last of them.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertFetchError`] when a streamed expert cannot be fetched. The session
    /// is then left partially updated and refuses further tokens.
    ///
    /// # Panics
    ///
    /// Panics when `ids` is empty, the session's capacity would be exceeded, or the
    /// session was broken by an earlier error.
    pub fn feed(
        &self,
        session: &mut Session,
        ids: &[u32],
        experts: &mut dyn ExpertSource,
    ) -> Result<Vec<f32>, ExpertFetchError> {
        assert!(!ids.is_empty(), "feed needs at least one token");
        assert!(
            !session.broken,
            "this session was left partially updated by an earlier error"
        );
        match self.run(
            ids,
            &mut session.states,
            session.position,
            Positions::Last,
            None,
            experts,
        ) {
            Ok(logits) => {
                session.position += ids.len();
                Ok(logits)
            }
            Err(error) => {
                session.broken = true;
                Err(error)
            }
        }
    }

    fn fresh_states(&self, capacity: usize) -> Vec<LayerState> {
        (0..self.config.num_hidden_layers)
            .map(|layer| {
                if self.config.is_mla(layer) {
                    LayerState::Mla(MlaCache::new(&self.config, capacity))
                } else {
                    LayerState::Kda(KdaState::new(&self.config))
                }
            })
            .collect()
    }

    fn run(
        &self,
        ids: &[u32],
        states: &mut [LayerState],
        cached: usize,
        positions: Positions,
        mut shared_slot: Option<&mut KdaState>,
        experts: &mut dyn ExpertSource,
    ) -> Result<Vec<f32>, ExpertFetchError> {
        let c = &self.config;
        let e = c.hidden_size;
        let vocab = c.vocab_size;
        let t = ids.len();

        let mut h = vec![0.0_f32; t * e];
        for (row, &id) in h.chunks_exact_mut(e).zip(ids) {
            self.embed.row_into(row, id as usize, e);
        }

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        for (layer, (weights, state)) in self.layers.iter().zip(states.iter_mut()).enumerate() {
            // The shared slot stands in for this layer's own KDA state for the duration of
            // the layer, then goes back to being the one slot every KDA layer reuses.
            let slot = match (shared_slot.as_deref_mut(), &mut *state) {
                (Some(slot), LayerState::Kda(own)) => {
                    slot.clear();
                    std::mem::swap(own, slot);
                    Some((slot, own))
                }
                _ => None,
            };
            if let Some((slot, own)) = slot {
                let mut lent = LayerState::Kda(std::mem::replace(own, KdaState::empty()));
                let result = decoder_layer(
                    &mut h,
                    &mut blocks,
                    weights,
                    c,
                    layer,
                    t,
                    &mut lent,
                    cached,
                    experts,
                );
                if let LayerState::Kda(used) = lent {
                    *own = used;
                }
                std::mem::swap(own, slot);
                result?;
            } else {
                decoder_layer(
                    &mut h,
                    &mut blocks,
                    weights,
                    c,
                    layer,
                    t,
                    state,
                    cached,
                    experts,
                )?;
            }
        }

        let first = match positions {
            Positions::All => 0,
            Positions::Last => t - 1,
        };
        let mut logits = vec![0.0_f32; (t - first) * vocab];
        let mut src = vec![0.0_f32; (blocks.len() + 1) * e];
        let mut normed = vec![0.0_f32; e];
        for (row, step) in logits.chunks_exact_mut(vocab).zip(first..t) {
            let ht = &mut h[step * e..(step + 1) * e];
            if let Some((norm, proj)) = self.out_res {
                let fold: Vec<f32> = norm.iter().zip(proj).map(|(&n, &p)| n * p).collect();
                for (b, block) in blocks.iter().enumerate() {
                    src[b * e..(b + 1) * e].copy_from_slice(&block[step * e..(step + 1) * e]);
                }
                src[blocks.len() * e..].copy_from_slice(ht);
                attn_res(ht, &src, &fold, blocks.len() + 1, e, c.rms_norm_eps);
            }
            rmsnorm(&mut normed, ht, self.final_norm, c.rms_norm_eps);
            self.lm_head.mul(row, &normed, e, vocab);
        }
        Ok(logits)
    }
}

#[derive(Clone, Copy)]
enum Positions {
    All,
    Last,
}

/// Index of the largest logit, first index on ties, as the C engine's greedy pick.
#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            best = i;
        }
    }
    best
}
