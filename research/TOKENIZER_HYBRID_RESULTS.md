# Tokenizer NoPE hybrid results

The finite width64 comparison is running. 10 completed quality trials have passed saved-artifact audit. No adoption yet. See [protocol](TOKENIZER_HYBRID_PROTOCOL.md).

| Stage | Family | Seed | LR | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|
| screen | hybrid | 3001 | 0.00125 | 4.954479 | 168.87 |
| screen | hybrid | 3001 | 0.0025 | 4.961288 | 171.62 |
| screen | gated_mla | 3001 | 0.0025 | 5.027217 | 112.39 |
| screen | gated_mla | 3001 | 0.00125 | 5.050826 | 113.10 |
| screen | kda | 3001 | 0.00125 | 4.922120 | 202.67 |
| screen | kda | 3001 | 0.0025 | 4.954309 | 201.60 |
| confirm | hybrid | 3011 | 0.00125 | 4.965797 | 171.51 |
| confirm | kda | 3011 | 0.00125 | 4.957274 | 201.59 |
| confirm | kda | 3012 | 0.00125 | 4.930151 | 199.64 |
| confirm | hybrid | 3012 | 0.00125 | 4.944646 | 172.14 |

The hybrid is a four-layer dense NoPE KDA/KDA/KDA/gated-MLA adaptation. All three layouts receive equal two-LR screening; only hybrid and the selected homogeneous control receive fresh confirmation. Screening alone establishes no improvement. Quality gates require all fresh pairs and conditional intervals; final serial cache/cost measurements and artifact review are required before adoption.

All research checkpoints and preflight failure evidence are preserved. Test loss remains closed; this is adaptive validation evidence at width64 only. The earlier width256 GQA and RoPE MLA options remain separate.
