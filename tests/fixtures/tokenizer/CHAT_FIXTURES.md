# Chat fixtures

`byte_chat/` is a synthetic 256-byte vocabulary plus the five XTML/EOS tokens
used by terminal tests. It enables deterministic CI without downloaded models;
it is not Kimi's vocabulary and must never be used for real inference.

`chat_segments.json` was generated from the locally downloaded official
`encoding_k3.py`, checkpoint revision `f831ab66814297da540d832a5235f8e904f29d06`,
on 2026-09-24. It records `build_chat_segments` results for a first user message
(`thinking_effort=low`) and an appended user message containing literal control
marker spellings and Unicode (no repeated system prefix). Both use default
`thinking=True`, `add_generation_prompt=True`, no tools or media. Each object's
`segments` retains `text` and `allow_special` verbatim, in order. The encoder is
the reference, not our Rust renderer.

The CLI checks exact token equality to those independently rendered segments
using the byte vocabulary in normal tests, and the actual checkpoint tokenizer
in its explicit ignored local test. Existing tokenizer parity fixtures separately
gate byte-level BPE against an independent tokenizer. This does not claim
end-to-end output-quality parity or tool/media support.
