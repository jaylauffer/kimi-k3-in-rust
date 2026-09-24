mod common;

use kimi_k3_core::{
    layer::{Matrix, NoStreamedExperts},
    safetensors::SafeTensorIndex,
    trunk::{TopLevelWeights, TrunkForwardError, TrunkRing},
};

#[test]
fn cancelled_forward_does_not_read_layers_or_index_the_embedding() {
    let config = common::tiny_config();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cache");
    let index = SafeTensorIndex::open(dir).unwrap();
    let mut ring = TrunkRing::open(&index, &config, 0, 1).unwrap();
    let top = TopLevelWeights {
        embed: Matrix::F32(&[]),
        lm_head: Matrix::F32(&[]),
        final_norm: &[],
        out_res: None,
    };
    let result = ring.forward_controlled(
        &index,
        &config,
        &top,
        &[u32::MAX],
        &mut NoStreamedExperts,
        || false,
    );
    assert!(matches!(result, Err(TrunkForwardError::Cancelled)));
    assert_eq!(ring.stats(), (0, 0));
}
