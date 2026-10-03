#!/usr/bin/env python3
"""Writes tests/fixtures/gemma4/opening.json: the Gemma 4 chat template's rendering of a
conversation that declares every chat tool, as token ids.

    K3_WRITE_GEMMA_DECLARATION=1 cargo test -p kimi-k3-cli gemma_opening
    python3 scripts/gemma_template_fixture.py /Volumes/Jarraya/gemma-4-31b-it

Needs jinja2 and tokenizers. The environment is the one transformers'
apply_chat_template uses (sandboxed, trim_blocks, lstrip_blocks, loopcontrols).
"""
import json
import pathlib
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

checkpoint = pathlib.Path(sys.argv[1])
fixtures = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/gemma4"
declaration = (fixtures / "declaration.json").read_text()

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

env = ImmutableSandboxedEnvironment(
    trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
)
env.globals["raise_exception"] = raise_exception
template = env.from_string((checkpoint / "chat_template.jinja").read_text())
system = "You are Gemma, running locally.\nUse the tools.\n"
user = "Read docs/README.md and summarise it."
text = template.render(
    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    tools=json.loads(declaration),
    add_generation_prompt=True,
    bos_token="<bos>",
)
ids = Tokenizer.from_file(str(checkpoint / "tokenizer.json")).encode(
    text, add_special_tokens=False
).ids
(fixtures / "opening.json").write_text(
    json.dumps({"declaration": declaration, "system": system, "user": user, "text": text, "ids": ids},
               ensure_ascii=False, indent=0)
)
print(f"{len(ids)} tokens")
