# Tokenizer NoPE hybrid results

October27,2026. Preparation complete for a finite width64 hybrid comparison;
no quality result or adoption yet. See [protocol](TOKENIZER_HYBRID_PROTOCOL.md)
and [implementation/pilot evidence](TOKENIZER_HYBRID_PREPARATION.md).

The20-update hybrid feasibility pilot passed. Controller and benchmark tests
passed9 checks. Both homogeneous-control MPS checks and prospective storage review passed.
The initial check failed before any optimizer update because its target slice
was noncontiguous; its log and plan were preserved, and a separately frozen
input-harness correction passed without changing model math. All prior LR/GQA/horizon/MLA rounds remain closed.
Test loss remains closed and every research checkpoint is preserved.
