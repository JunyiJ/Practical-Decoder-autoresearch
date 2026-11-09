# Output composition: exact batching is feasible, quality remains untested

Three new bounded work items completed: analytical vocabulary cost, a chunked
gradient pilot, and a justified batching optimization. No model was trained,
loss scored, scientific role drawn, selected weight modified, or holdout opened.
The original model and all closed quality outcomes remain unchanged.

## What full-vocabulary composition costs

The exact pinned codebook has50257 rows, mean code length8.383489 including
boundaries, minimum3 and maximum130. The existing composer has725235 parameters.
It cannot be costed as if only the1024 input positions required composition.

| Analytical layout | Forward MACs | Relative to existing head matmul | One copy of all convolution values | All unfolded windows |
|---|---:|---:|---:|---:|
| Every row padded to130 | 250.815 billion | 19.038× | 13.368 GB | 8.021 GB |
| Exact length grouping | 42.575 billion | 3.232× | .554 GB | .332 GB |

These are arithmetic/tensor inventories, not measured RSS or elapsed time.
Highway/projection dense layers account for34.459 billion of the grouped
forward MACs. Composed output rows alone occupy51,463,168bytes in FP32.
One independent composer adds725235 parameters (before any optional gate);
it does not reduce parameters when the independent output table is retained.

The nominal raw/H200 float32 tensor payload for three incumbent runs and six
augmented runs is1,212,145,128bytes, including34,811,280 additional bytes from
the composers. This is a lower-level payload calculation, not a complete panel
forecast: gates, metadata, temporary serialization overlap, possible resumable
states and reserve/headroom must be included before admission. No storage
blocker or quality panel is declared from this partial forecast.

## Exact recomputation and measured execution

The prototype builds the table without retaining a graph, then recomputes each
chunk to apply its upstream gradient and accumulate all parameter gradients.
This implements the first-order chain rule without stale output rows. Nominal
dense work becomes four times forward MACs because of the extra forward.
It has not yet been integrated with full-model gradients, clipping, optimization,
averaging or checkpoint export.

| Pilot | Grouping | Chunks | Table forward | Recompute + VJP | Combined | Entire guarded pilot | Peak process RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| 001 | At most1024 rows per length | 73 | 2.9174s | 6.7201s | 9.6375s | 18.0258s | 1.253 GB |
| 002 | At most133120 byte positions | 30 | .5414s | 2.1247s | 2.6661s | 7.3842s | 1.451 GB |

The original conservative row cap caused avoidable execution overhead. The
existing embedding primitive actually bounds total positions; larger short-token
batches respect that same bound. Pilot002 used a separate source and output
directory and preserved001. This is an execution improvement, not a model-quality
gain. The3.61× measured difference comes from one pass per implementation, with
different warming histories; it is not a sustained throughput benchmark.

Both pilots passed the fixed36-row CPU float64 reference comparison, grouped
CPU/MPS outputs, every one of18 parameter gradients, and MPS retained-graph
versus recomputed gradients. Pilot002 additionally passed the first1025
length6 IDs, comparing the old and enlarged batches. Maximum difference in
that added case was1.6392e-7 for outputs and1.1176e-8 for gradients, within
unchanged atol4e-6/rtol4e-5. Full50257-row passes had finite outputs/gradients
and exact coverage. Full-vocabulary gradients were not compared to a dense
float64 oracle; the numerical proof is the specified fixture plus the exact
chain-rule construction. No broader proof is claimed.

MPS driver allocation at the end of pilot002 was1,155,219,456bytes; this is not
a peak or interchangeable with process RSS. All guards finished successfully;
final free disk was11,093,159,936bytes. The analytical check took.204603seconds.

## Research decision and next action

Do not launch a long panel from these component results. Extrapolating the
single measured2.6661seconds over six1200-step augmented runs gives5.33hours
of added component work, before the Transformer, validation and engineering.
That is a planning estimate, not a measured total or a guaranteed bound. The
first implementation would have implied19.27hours; it is superseded for future
integration, while its evidence remains immutable.

The evidence in `BYTE_OUTPUT_LEXICAL_REVIEW.md` supports considering an
independent lexical augmentation and a matched nonlexical control, with an
unchanged byte-input incumbent. Full replacement and input/output tying have
contrary evidence and are not inferred to be better from this pilot.

Next executable action: specify the minimal full-model augmentation and its
initialization/optimizer ownership, then prepare a bounded integration/resource
pilot using the position-bounded exact VJP. It must verify the complete model
gradient chain (including table and context), global clipping order, one-step
optimizer/averaging ownership and a saved inference roundtrip. Before quality
training, freeze the exact candidate and controls, complete wall/memory/disk
forecast, fresh roles and confirmation requirements, and prospective futility
rules that can stop costly unpromising work without permitting partial-panel
adoption. No quality coefficient, seed, endpoint or new training is selected here.

Evidence roots:
`tokenizer-evidence/byte-output-composition-cost-001`,
`tokenizer-evidence/byte-output-chunk-pilot-001`,
`tokenizer-evidence/byte-output-chunk-pilot-002`.
Each contains pinned sources/inputs, resource guard and immutable result.

Publication check: Git reports one trailing space on the same declaration in
each executed pilot source. Preserve those exact hashed sources; the cosmetic
warnings do not change parsing or numerical results and do not justify reruns.
