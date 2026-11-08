# Scientific admission after the bounded pilot

The prospective pilot completed all 32 updates in 55.675 seconds, peak RSS
3,201,941,504 bytes, with no execution failure. It performed no validation
selection. Both backend baseline checks reproduced all 105 gradients, four
updates, optimizer states and EMA exactly. Analytic linear/tanh gradient checks
had zero maximum error. Both new arms exposed all 106 parameters and preserved
exact serialized weights. Maximum trained same-weight CPU/MPS batch CE error
was 8.327e-6 and maximum target CE error 1.3733e-4, below the predeclared 3e-5
and 3e-4 limits. Independent CPU and MPS training trajectories need not agree
bitwise; exactness refers to new-versus-official baseline on each backend.

Warm MPS step maxima: byte .496506s, linear .477308s, tanh .514722s. These are
three warm observations per arm, not sustained throughput estimates. Projected
serialized weights measured 65,701,675 bytes. The complete nine-run requirement
remains 18 retained files, 1.5GiB total output cap, 8GiB reserve plus 2GiB extra
headroom. Full time forecast uses 1.5 times each warm maximum for 1,200 updates,
plus 1,200 seconds per run for evaluations and serialization. This forecasts
about 5.23 hours for nine runs; the independent outer allowance is 10 hours,
with one-hour per-trial guards. The quality-plan records exact measured inputs
and calculations. No paid compute and no cleanup are required at admission.

Admit the fixed .0025, three-arm by three-fresh-bundle comparison specified in
BYTE_CONTEXT_TRANSFORM_PROTOCOL.md, subject to the prospective immutable
quality-plan, resource check and fresh-role census. This is a direct comparison
to the current byte baseline. No hyperparameter search or screen selection is
performed. All 1,200 updates and H200 rules, both controls, strict .02 gain and
simultaneous conditional-text bounds are required. All three arms share data
order and common initial parameters within each bundle; order rotates between
bundles. The separate output table and genuine byte composer are retained.

Relative priority: adding an automatic shared-table adapter lacks support from
the geometry result. Repeating probability mixing lacks a new local rationale
for its extra vocabulary projections. A single context transform addresses the
remaining feature-parameterization question with one small matrix, two controls
and a changed, adopted byte-input condition. Its evidence is limited and mixed,
which is why it receives a bounded fixed-recipe test rather than a tuning grid.
The negative outcome would reject this exact recipe; a positive outcome would
justify independent artifact/consumer proof before adoption, not claims about
softmax rank, morphology, unseen datasets or equal-time performance.

The independent saved-result reviewer is prepared before quality outcomes. It
checks source/assets and plan binding, full update/gradient coverage, paired
samplers, numerical reports, retained weight integrity and independent scalar
bootstrap arithmetic. A scientific gate failure is retained without rescue.
Engineering failures preserve their exact diagnostics before any separately
identified recovery. Historical campaigns and failed fixtures remain closed.
