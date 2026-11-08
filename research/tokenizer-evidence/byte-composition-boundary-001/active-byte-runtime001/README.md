# Active byte-row prototype: closed, not adopted

The unchanged deterministic byte backward was measured at about0.195seconds in
the current profile. Existing-data analysis found54 active rows in the first1024
training tokens, and a median58 (range50–63) in261 complete nonoverlapping windows,
versus259 possible rows. These are descriptive counts, not paired-sampler expectations.

The isolated prototype uses sorted CPU unique IDs, computes only those gradient
rows, and copies the result into the complete259x15 tensor. It preserves original
2048-position chunks, chunk order, CPU behavior, and all validation predicates.
It does not remove padded positions or change the adopted model.

The prospective comparison ran seven fixed cases on CPU and MPS. All forward
outputs and all CPU gradients matched exactly. MPS gradients matched in the three
real-input cases, the all259-row control, and the singleton. Two cases failed:

- One active row over the full133120-position bound:14 gradient coordinates differ;
  maximum absolute difference0.0001678466796875.
- Four active rows over2049positions:48 coordinates differ;
  maximum absolute difference0.00002288818359375.

This is a numerical failure of the fixed exactness gate, not a harness defect.
The candidate is closed and not adopted. No tolerance, case or result was changed.
The original implementation remains active through bulk-hash-adoption001.

On the representative masked prefix, median candidate/reference primitive latency
was0.402490 (about59.75% lower), with7/7 paired wins. Selection cost had median
1.122ms. That useful component timing does not override the failed numerical gate
and is not a training-throughput or quality result. The real-input repeated
measurements were exact and repeatable; this does not establish all-input equivalence.
The changed matrix output dimension is a plausible explanation for the tiny-row
rounding changes, but kernel attribution has not been established.

No optimizer update, quality evaluation, new random role or weight archive was
performed. Weight bytes and global RNG were unchanged. All sources, case digests,
error magnitudes, timings and the original frozen plan remain available here.

Next: use the saved case evidence to assess whether a separate implementation can
avoid numerical regime changes. Do not replay this closed prototype or mask the
failing small-row inputs. Any new hypothesis must preserve those cases, exactness
and the full input domain, with a prospective rationale and resource budget.
No worker is active.
