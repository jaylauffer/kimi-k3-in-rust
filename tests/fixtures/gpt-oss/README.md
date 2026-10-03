# gpt-oss-20b fixtures

`chat_template.jinja` is the `tokenizer.chat_template` from ggml-org's
`gpt-oss-20b-mxfp4.gguf` as downloaded 2026-04-27 (16,738 bytes). That copy was
deleted on 2026-10-04 when the GGUFs were deduplicated (loadngo
`docs/ZHOENUS_HEAD_MODEL_RUNNER.md`). Its template is 804 characters longer than the
one in the archived GGUF (CAS object `56fcc05c…`), which is now the only copy of the
weights. It is kept here as the reference for the harmony chat format, for a port of
gpt-oss to this engine (`docs/ORCHESTRATION.md`). gpt-oss is Apache 2.0.
