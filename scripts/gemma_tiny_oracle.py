#!/usr/bin/env python3
"""Writes tests/fixtures/gemma4/tiny/: a random-weight Gemma 4 text model with the 31B's
structure at toy size, and transformers' own logits for it, the oracle for
kimi-k3-core's gemma module (tests/gemma_tiny_oracle.rs).

    python3 scripts/gemma_tiny_oracle.py

Needs torch, safetensors and transformers with Gemma 4 (5.5 or later). The model mixes
sliding and full layers like the 31B (values from keys on full layers, proportional
rotary on a quarter of their pairs, grouped heads, logit soft-capping); every weight,
norm and layer scalar is random so no term can hide behind a default of 1 or 0. The
sliding window is 4 and the text 600 tokens, so the Rust ring of window + 512 slots
wraps.
"""
import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

out = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/gemma4/tiny"
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261003)

text = dict(
    vocab_size=1024,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_global_key_value_heads=1,
    head_dim=32,
    global_head_dim=64,
    layer_types=["sliding_attention", "sliding_attention", "full_attention",
                 "sliding_attention", "sliding_attention", "full_attention"],
    sliding_window=4,
    attention_k_eq_v=True,
    final_logit_softcapping=30.0,
    hidden_activation="gelu_pytorch_tanh",
    rms_norm_eps=1e-6,
    rope_parameters={
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                           "rope_theta": 1000000.0},
    },
    max_position_embeddings=4096,
    tie_word_embeddings=True,
    bos_token_id=2,
    eos_token_id=1,
    pad_token_id=0,
    enable_moe_block=False,
    hidden_size_per_layer_input=0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    attention_bias=False,
)
config = Gemma4TextConfig(**text)
config._attn_implementation = "eager"
model = Gemma4ForCausalLM(config).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        else:
            p.copy_(0.08 * torch.randn_like(p))
    for layer in model.model.layers:
        layer.layer_scalar.copy_(0.5 + torch.rand(1))
    # The embedding is stored and computed in bfloat16 by the real checkpoint; keep every
    # weight exactly representable in bfloat16 so both sides read the same numbers.
    for p in list(model.parameters()) + [l.layer_scalar for l in model.model.layers]:
        p.copy_(p.to(torch.bfloat16).to(torch.float32))

tensors = {}
for name, p in model.state_dict().items():
    if name == "lm_head.weight":
        continue  # tied
    tensors[name.replace("model.", "model.language_model.", 1)] = (
        p.detach().to(torch.bfloat16).contiguous()
    )
save_file(tensors, out / "model.safetensors")
(out / "config.json").write_text(json.dumps(
    {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
     "tie_word_embeddings": True, "eos_token_id": [1, 106], "text_config": text},
    indent=1))

ids = torch.randint(3, 1024, (1, 600), generator=torch.Generator().manual_seed(7))
ids[0, 0] = 2
with torch.no_grad():
    # The checkpoint's embedding scale is sqrt(hidden) rounded to bfloat16.
    logits = model(ids).logits[0]
positions = sorted(set(list(range(0, 12)) + list(range(500, 530)) + [599]))
(out / "oracle.json").write_text(json.dumps({
    "ids": ids[0].tolist(),
    "positions": positions,
    "logits": [logits[p].tolist() for p in positions],
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
}))
print("embed scale", model.model.embed_tokens.embed_scale.item(),
      "tensors", len(tensors), "positions", len(positions))
