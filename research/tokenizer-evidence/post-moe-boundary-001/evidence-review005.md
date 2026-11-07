# Diagnostic branch closed without a batch-selection conclusion

The purpose of the gradient diagnostic was to decide whether larger-batch
training deserved a direct quality experiment. It has not established a batch
prescription or any model-quality improvement. The raw signal estimate varies
substantially across independent sampled cohorts. The new CPU optimizer-response
component passed its generated arithmetic proof, but the subsequent strict MPS
transport proof failed. Close this bounded diagnostic branch as inconclusive;
keep the native dense incumbent and return to quality-facing direction selection.

## Original target-device failure is preserved

NEW root `research_runs/tokenizer-optimizer-response-mps-component-v1-001` is
CLOSED_FAILED with its original STOP, source plan and exception guard unchanged.
Session94578/PID11142 exited1. The exception was `Independent parameter/exposure
response differs on mps`. It occurred after1.4090s at RSS450052096B and MPS driver
17268736B, inside the original resource limits. This is a numerical agreement
failure, not a capacity failure or a negative model-quality result.

The full54-case agreement panel did not complete. Its final RNG/source invariance
checks were not reached. No partial panel or endpoint is accepted, and the exact
failing case/parameter was not saved by the original exception. Do not infer it
or rerun the closed cases to reconstruct it. The original tolerances remain
unchanged. All1050 successful predecessors and three new failed transport source
files are immutable, giving1053 bindings. Neither model nor RAW/corpus data was
loaded and no actual-model update occurred. The CPU-only component remains
valid only for its declared synthetic CPU scope.

## One distinct, bounded localization completed

A NEW one-file primitive probe used different inputs:2x4/3x4 rational matrices
and a length5 vector. It did not call an optimizer, execute the old proof, replay
its54 cases or load a model/checkpoint/corpus. It compared FP32 and bfloat16 CPU/MPS
norm,normalization,matmul,addmm and finite matrix-iteration stages. Four prospective
scalar precision models were all reported descriptively;none was used to rescue
the failed transport acceptance. Source and input/globalCPU-MPSRNG checks passed.
All1054 current source bindings are immutable.

On these inputs, FP32 addmm CPU/MPS discrepancies were at most2.3841858e-7, while
BF16 isolated addmm differences reached.0078125. Cast,norm and normalization agreed
in both BF16 examples;the first trace divergence was iteration1/polynomial.
Later stage maxima reached.25 for2x4 and.109375 for3x4. The sampled FP32 vector
first/second moment,quotient andnorm primitives agreed exactly. These are finite
primitive observations,not measurements of actual model updates or proof of a
training defect. The four scalar rounding candidates did not explain all MPS
BF16 results,so the precise complete backend arithmetic remains unresolved.

The exact installed PyTorch git revision5c4886908584029761b579af026dcfb627c84070
was retrieved from the primary source. In LinearAlgebra.mm,the MPSGraph addmm
branch uses input-dtype scalar constants and separate operation nodes;the
use_metal_mm conditions select other paths. This motivates the rounding-boundary
hypothesis but does not fully specify compiled backend arithmetic. See the exact
[primary implementation](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/LinearAlgebra.mm),
lines497–548 and1116–1204. No backend flag,runtime,optimizer algorithm or original
source was changed.

The distinct primitive probe completed1.4899s/RSS383057920B/MPS17252352B under
600s/4GiB/6GiB/reserve8GiB/2MiBmetadata bounds. Full floor9861394432B and target
12008878080B fit~80.6GB free. No storage blocker,cleanup or retirement arose.
Independent saved-metadata arithmetic review agreed exactly without backend or
optimizer replay. Seven closed failure/localization files were mirrored losslessly,
ZEROweights;see optimizer-precision-review001 and its metadata mirror manifest.
The new localization root is terminal and every command is closed. No worker
remains active. Original scientific model/validation evidence is unchanged;
this failure of a new CPU/MPS response criterion does not reopen past studies.

## Research decision

Using evidence-grounded-research and the human's model-quality objective, do not
extend this measurement pipeline merely to make its proof pass. Its immediate
value to choosing a quality-improving change is weak:raw ratios are uncertain,
stateful nonlinear response lacks a qualified device-equivalence instrument,and
no direct validation improvement has been measured. This does not prove larger
batches cannot help. It leaves that quality question untested and unadmitted.
McCandlish1812.06162v1's previously read local-step,conditioning,LR,ratio-bias and
generalization caveats remain;historical character batch reversals stay unpooled.

Next executable work is a focused direction review against the actual native
incumbent,original acceptance/final results/curves and architecture/attention
roadmaps. Compare remaining structural and recipe hypotheses using supporting
AND contrary local evidence and relevant primary-paper sections. Select only a
qualified falsifiable quality hypothesis with discriminating controls and a full
cost,then implement its smallest necessary prerequisite. If none qualifies,answer
one specific existing-data or literature question. No additional raw cohorts,
precision probes,optimizer-response tooling,actual endpoint-state consumer or
batch quality panel is admitted by this closure. Never use the old wide control
as a fallback adoption. No random grid or exploration-quota run.

See diagnostic-branch-close001.json. All current and historical sources,STOPs,
failures,metrics,receipts,required data/tokenizers and protected weights remain.
This work is diagnosis/preparation,not numerical quality exploration;70/20/10
remains proposed. No model,training policy or incumbent has been changed.
