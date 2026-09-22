//! Validates `kimi-k3-core::bind` against the real, released checkpoint on disk,
//! rather than a hand-built fixture shaped like its names.
//!
//! Ignored by default: it needs the real ~1.4 TB checkpoint locally
//! (`KIMI_K3_CHECKPOINT`, default `/Volumes/Jarraya/kimi-k3`) and reads a real
//! several-GB slice of it (the embedding table, LM head, and two decoder
//! layers), so it is deliberately not part of the fast `cargo test` gate.
//!
//! Run explicitly: `cargo test -p kimi-k3-core --test real_checkpoint_binding -- --ignored --nocapture`

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use kimi_k3_core::{
    bind::BoundStorage, config::K3Config, layer::Matrix, safetensors::SafeTensorIndex,
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

#[test]
#[ignore = "needs the real ~1.4 TB checkpoint locally; run explicitly with --ignored"]
fn binds_the_embedding_lm_head_and_first_two_real_layers() {
    let dir = checkpoint_dir();
    let config = K3Config::from_path(dir.join("config.json")).expect("real config parses");
    assert_eq!(config.hidden_size, 7168);
    assert_eq!(config.num_hidden_layers, 93);
    assert_eq!(config.vocab_size, 163_840);
    assert!(
        config.is_dense(0),
        "layer 0 is the checkpoint's one dense layer"
    );
    assert!(
        !config.is_dense(1),
        "layer 1 is the checkpoint's first MoE layer"
    );

    let start = Instant::now();
    let index = SafeTensorIndex::open(&dir).expect("real checkpoint indexes");
    eprintln!(
        "indexed {} tensors across {} shards in {:?}",
        index.tensors().len(),
        index.shard_paths().len(),
        start.elapsed()
    );
    assert_eq!(index.shard_paths().len(), 96);

    let mut storage = BoundStorage::new();

    let start = Instant::now();
    storage
        .load_top_level(&index)
        .expect("top-level tensors bind");
    eprintln!(
        "loaded embedding + lm_head + final norm in {:?}",
        start.elapsed()
    );

    let start = Instant::now();
    storage
        .load_layer(&index, &config, 0)
        .expect("dense layer 0 binds");
    storage
        .load_layer(&index, &config, 1)
        .expect("first MoE layer binds");
    eprintln!("loaded layers 0 and 1 in {:?}", start.elapsed());

    let dense = storage
        .layer_weights(&config, 0)
        .expect("dense layer weights build");
    let moe = storage
        .layer_weights(&config, 1)
        .expect("moe layer weights build");

    assert!(
        matches!(dense.attention, kimi_k3_core::layer::Attention::Kda(_)),
        "layer 0 must be KDA, not MLA"
    );
    match &dense.mlp {
        kimi_k3_core::layer::Mlp::Dense { gate, up, down } => {
            let expected_gate = config.intermediate_size * config.hidden_size;
            assert_eq!(matrix_len(gate), expected_gate);
            assert_eq!(matrix_len(up), expected_gate);
            assert_eq!(matrix_len(down), expected_gate);
        }
        kimi_k3_core::layer::Mlp::Moe(_) => panic!("layer 0 must be dense, not MoE"),
    }

    match &moe.mlp {
        kimi_k3_core::layer::Mlp::Moe(weights) => {
            assert_eq!(weights.gate.len(), config.num_experts * config.hidden_size);
            let expected_bottleneck = config.routed_expert_hidden_size * config.hidden_size;
            assert_eq!(matrix_len(&weights.down), expected_bottleneck);
            assert_eq!(matrix_len(&weights.up), expected_bottleneck);
            assert_eq!(weights.latent_norm.len(), config.routed_expert_hidden_size);
            assert!(
                matches!(
                    weights.experts,
                    kimi_k3_core::layer::RoutedExperts::Streamed
                ),
                "real routed experts must never bind resident"
            );
        }
        kimi_k3_core::layer::Mlp::Dense { .. } => panic!("layer 1 must be MoE, not dense"),
    }

    eprintln!("real-checkpoint binding validated for the embedding, LM head, and layers 0/1");
}
