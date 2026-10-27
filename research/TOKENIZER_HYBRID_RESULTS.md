# Tokenizer NoPE hybrid results

The finite width64 comparison is running. 6 completed quality trials have passed saved-artifact audit. No adoption yet. See [protocol](TOKENIZER_HYBRID_PROTOCOL.md).

| Stage | Family | Seed | LR | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|
| screen | hybrid | 3001 | 0.00125 | 4.954479 | 168.87 |
| screen | hybrid | 3001 | 0.0025 | 4.961288 | 171.62 |
| screen | gated_mla | 3001 | 0.0025 | 5.027217 | 112.39 |
| screen | gated_mla | 3001 | 0.00125 | 5.050826 | 113.10 |
| screen | kda | 3001 | 0.00125 | 4.922120 | 202.67 |
| screen | kda | 3001 | 0.0025 | 4.954309 | 201.60 |

The hybrid is a four-layer dense NoPE KDA/KDA/KDA/gated-MLA adaptation. All three layouts receive equal two-LR screening; only hybrid and the selected homogeneous control receive fresh confirmation. Screening alone establishes no improvement. Quality gates require all fresh pairs and conditional intervals; final serial cache/cost measurements and artifact review are required before adoption.

All research checkpoints and preflight failure evidence are preserved. Test loss remains closed; this is adaptive validation evidence at width64 only. The earlier width256 GQA and RoPE MLA options remain separate.

Screening selected all-KDA as the homogeneous control, with LR0.00125 for both it and hybrid (all-gated-MLA selected LR0.0025). Screen H200 CE is4.922120 for all-KDA versus4.954479 for hybrid, a single-seed delta+0.032359; this is not confirmation evidence. Fresh paired seeds3011/3012/3013 are now running. The cache-memory route is inapplicable against all-KDA, which has smaller allocated state; only the frozen strict-quality route remains eligible for adoption.
