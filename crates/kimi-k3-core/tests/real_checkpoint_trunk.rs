//! Validates `kimi-k3-core::trunk::TrunkRing` against the real, released
//! checkpoint: that prefetch-then-bind sequencing actually overlaps I/O with
//! nothing (no thread, no blocking call in between) and, most importantly,
//! that the ring produces exactly the same weights `bind::BoundStorage`'s
//! direct, already-validated path would for the same layers.
//!
//! Ignored by default, same reason and same checkpoint as
//! `real_checkpoint_binding.rs`: run explicitly with
//! `cargo test -p kimi-k3-core --test real_checkpoint_trunk -- --ignored --nocapture`.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use kimi_k3_core::{
    bind::BoundStorage,
    cache::{CachedExperts, ExpertCache},
    config::K3Config,
    expert::ExpertRef,
    layer::{Attention, Matrix, Mlp, NoStreamedExperts},
    safetensors::SafeTensorIndex,
    trunk::{TopLevelWeights, TrunkRing},
};

fn checkpoint_dir() -> PathBuf {
    env::var("KIMI_K3_CHECKPOINT")
        .map_or_else(|_| PathBuf::from("/Volumes/Jarraya/kimi-k3"), PathBuf::from)
}

fn matrix_len(matrix: &Matrix<'_>) -> usize {
    match *matrix {
        Matrix::F32(values) => values.len(),
        Matrix::Bf16(values) => values.len(),
    }
}

/// A handful of numbers that identify a layer's weights well enough to catch
/// the ring returning the wrong layer, a stale slot, or truncated data --
/// without needing `LayerWeights` to implement equality.
fn fingerprint(layer: &kimi_k3_core::layer::LayerWeights<'_>) -> (usize, usize, u64) {
    let attn_len = match layer.attention {
        Attention::Kda(w) => matrix_len(&w.o),
        Attention::Mla(w) => matrix_len(&w.o),
    };
    let (mlp_len, mut hash): (usize, u64) = match &layer.mlp {
        Mlp::Dense { down, .. } => (matrix_len(down), 0),
        Mlp::Moe(weights) => (matrix_len(&weights.down), 0),
    };
    for &value in layer.in_norm {
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(u64::from(value.to_bits()));
    }
    (attn_len, mlp_len, hash)
}

#[test]
#[ignore = "needs the real ~1.4 TB checkpoint locally; run explicitly with --ignored"]
fn ring_matches_direct_binding_across_a_pin_boundary_and_prefetch() {
    let dir = checkpoint_dir();
    let config = K3Config::from_path(dir.join("config.json")).expect("real config parses");
    let index = SafeTensorIndex::open(&dir).expect("real checkpoint indexes");

    // Ground truth: bind layers 0..=3 directly, the already-validated path.
    let mut direct_storage = BoundStorage::new();
    for layer in 0..=3 {
        direct_storage
            .load_layer(&index, &config, layer)
            .unwrap_or_else(|error| panic!("direct load of layer {layer} failed: {error}"));
    }
    let direct_fingerprints: Vec<_> = (0..=3)
        .map(|layer| fingerprint(&direct_storage.layer_weights(&config, layer).unwrap()))
        .collect();

    // Ring: pin only layer 0, stream 1..=3 through a two-slot ring, prefetching
    // one layer ahead each time -- exactly the intended call pattern.
    let mut ring = TrunkRing::open(&index, &config, 1, 2).expect("ring opens");
    assert!(ring.is_pinned(0));
    assert!(!ring.is_pinned(1));

    let start = Instant::now();
    let mut ring_fingerprints = Vec::new();
    for layer in 0..=3 {
        if layer < 3 {
            ring.prefetch(&index, &config, layer + 1).expect("prefetch");
        }
        let bound = ring
            .bind(&index, &config, layer)
            .unwrap_or_else(|error| panic!("ring bind of layer {layer} failed: {error}"));
        ring_fingerprints.push(fingerprint(&bound));
    }
    eprintln!("ring walked layers 0..=3 in {:?}", start.elapsed());

    assert_eq!(
        ring_fingerprints, direct_fingerprints,
        "the ring must return exactly what direct binding does, layer for layer"
    );

    let (hits, misses) = ring.stats();
    eprintln!("ring stats: {hits} hits, {misses} misses");
    assert!(
        hits >= 2,
        "layers 2 and 3 were each prefetched one iteration ahead, so binding them \
         should find completed prefetches (hits), not block on a fresh load"
    );
}

/// `TrunkRing::forward`'s driver loop (embedding, state init, the final norm
/// and LM head) is a near-verbatim copy of `Model::run`'s, adapted to source
/// each layer from the ring instead of a resident `Vec`. This proves that copy
/// is exact -- bit-identical logits -- for real data, without needing the
/// streamed-expert cache wired up: layer 0 is the checkpoint's one dense,
/// non-MoE layer, so `NoStreamedExperts` is honestly correct here, not a stub
/// standing in for something unimplemented. Multi-layer block aggregation
/// across a dense/MoE boundary and the real streamed-expert path are exercised
/// by other tests (`bind`'s own gates, `cache.rs`), not this one.
#[test]
#[ignore = "needs the real ~1.4 TB checkpoint locally; run explicitly with --ignored"]
fn ring_forward_matches_model_forward_bit_exactly_on_the_real_dense_layer() {
    let dir = checkpoint_dir();
    let mut config = K3Config::from_path(dir.join("config.json")).expect("real config parses");
    assert!(config.is_dense(0), "layer 0 must be dense for this test");
    config.num_hidden_layers = 1;
    let index = SafeTensorIndex::open(&dir).expect("real checkpoint indexes");
    let ids = [42_u32, 100, 7];

    let mut storage = BoundStorage::new();
    storage
        .load_top_level(&index)
        .expect("top-level tensors bind");
    storage
        .load_layer(&index, &config, 0)
        .expect("layer 0 binds");
    let model = storage.model(&config).expect("one-layer model builds");
    let direct_logits = model
        .forward(&ids, &mut NoStreamedExperts)
        .expect("direct forward succeeds");

    let mut ring = TrunkRing::open(&index, &config, 1, 1).expect("ring opens with layer 0 pinned");
    let top = TopLevelWeights {
        embed: model.embed,
        lm_head: model.lm_head,
        final_norm: model.final_norm,
        out_res: model.out_res,
    };
    let ring_logits = ring
        .forward(&index, &config, &top, &ids, &mut NoStreamedExperts)
        .expect("ring forward succeeds");

    assert_eq!(
        ring_logits.len(),
        direct_logits.len(),
        "same sequence, same vocab, same logits shape"
    );
    assert_eq!(
        ring_logits, direct_logits,
        "TrunkRing::forward must be bit-identical to Model::forward on the same data"
    );
}

/// The dense-layer test above deliberately used `NoStreamedExperts`, honest for
/// a dense layer but not a real test of the streamed-expert path: 92 of the
/// checkpoint's 93 layers are `MoE`, and `TrunkRing::forward` accepting `experts:
/// &mut dyn ExpertSource` generically means `cache::CachedExperts` (the
/// engine's real, already independently-tested `ExpertSource`) plugs in with
/// no change to `trunk.rs` at all -- so this test's job is to prove that is
/// actually true against real routing and real expert bytes, not to add new
/// production code. Layers 0..=3 (one dense, three `MoE`) run through both
/// `TrunkRing::forward` and direct `Model::forward`, each with its own fresh
/// `ExpertCache` over the same index, and must still land on bit-identical
/// logits: real per-token top-k routing, decided by the real gate weights, is
/// deterministic given the same weights and the same input.
#[test]
#[ignore = "needs the real ~1.4 TB checkpoint locally; run explicitly with --ignored"]
fn ring_forward_matches_model_forward_with_the_real_streamed_expert_cache() {
    let dir = checkpoint_dir();
    let mut config = K3Config::from_path(dir.join("config.json")).expect("real config parses");
    assert!(config.is_dense(0), "layer 0 must be dense for this test");
    assert!(!config.is_dense(1), "layer 1 must be MoE for this test");
    config.num_hidden_layers = 4;
    let index = SafeTensorIndex::open(&dir).expect("real checkpoint indexes");
    let ids = [42_u32, 100, 7];

    let probe = ExpertRef::resolve(&index, 1, 0).expect("layer 1 expert 0 resolves");
    let new_cache = || {
        ExpertCache::new(
            config.num_hidden_layers,
            config.num_experts,
            config.num_experts_per_token,
            1 << 30, // 1 GiB: comfortably more than top_k + 1 experts' worth
            &probe,
        )
        .expect("expert cache sizes")
    };

    let mut storage = BoundStorage::new();
    storage
        .load_top_level(&index)
        .expect("top-level tensors bind");
    for layer in 0..config.num_hidden_layers {
        storage
            .load_layer(&index, &config, layer)
            .unwrap_or_else(|error| panic!("layer {layer} binds: {error}"));
    }
    let model = storage.model(&config).expect("four-layer model builds");
    let mut direct_cache = new_cache();
    let start = Instant::now();
    let direct_logits = model
        .forward(&ids, &mut CachedExperts::new(&mut direct_cache, &index))
        .expect("direct forward with real streamed experts succeeds");
    eprintln!(
        "direct forward over layers 0..=3 took {:?}",
        start.elapsed()
    );

    let mut ring = TrunkRing::open(&index, &config, 1, 2).expect("ring opens");
    let top = TopLevelWeights {
        embed: model.embed,
        lm_head: model.lm_head,
        final_norm: model.final_norm,
        out_res: model.out_res,
    };
    let mut ring_cache = new_cache();
    let start = Instant::now();
    let ring_logits = ring
        .forward(
            &index,
            &config,
            &top,
            &ids,
            &mut CachedExperts::new(&mut ring_cache, &index),
        )
        .expect("ring forward with real streamed experts succeeds");
    eprintln!("ring forward over layers 0..=3 took {:?}", start.elapsed());

    assert_eq!(
        ring_logits, direct_logits,
        "TrunkRing::forward must be bit-identical to Model::forward with the real \
         streamed-expert cache too, not just the dense-layer NoStreamedExperts case"
    );
    assert!(
        direct_logits.iter().all(|value| value.is_finite()),
        "real routed-expert output must be finite, not NaN/inf from a wiring mistake"
    );
}
