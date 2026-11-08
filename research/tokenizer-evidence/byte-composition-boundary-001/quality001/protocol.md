# Byte-input composition: original full quality comparison

The machine-readable `plan.json` and pinned `contract.py` define this original
campaign. Existing fixtures, pilot scores and closed studies are not quality
observations for this panel. No human approval between stages is required under
the standing autonomous-research instruction.

## Evidence and falsifiable question

Use `../quality-evidence001/evidence.md`, the original input-capacity analysis and
input-dropout final results, and [Kim et al.1508.06615v4](https://arxiv.org/pdf/1508.06615v4),
equations5–8, Optimization and Tables2/7. The paper supplies a reason to test CNN
and highway composition, not a local gain. Its word/character/LSTM/SGD setting
and larger corpora differ substantially. Its negative MLP ablation and filters
that do not generally match morphemes limit simple causal explanations.

Locally, uniform input rank reduction is not supported by the covariance
analysis, and input-feature dropout lost all three fresh comparisons. This
campaign tests a different mechanism: nonlinear sharing of lexical byte
fragments at fixed GPT2 token IDs, segmentation, context and output CE. It is
structural exploration; preparation and resource pilots do not count as quality
exploration. All campaign screening and confirmation costs belong to this
hypothesis.

Three arms are native embeddings, raw-byte composition and length-matched
unique nonlexical codes using the identical composer. Composer initialization
is paired. The code control does not match every byte marginal, so a positive
raw-minus-control result supports usefulness of this lexical coding, not an
isolated proof of substring-sharing or morphology. A control-relative win with
a native-relative loss gives no practical adoption. A native-only win must not
be described as establishing lexical utility. Nonlexical codes are a mechanism
control, not a fallback winner chosen after seeing confirmation.

## Fixed design

Nine complete screening trajectories cover the same three learning rates
(.000625,.00125,.0025) in all three arms. The screen bundle's native initializer,
composer initializer, code key and train/validation samplers are paired across
all nine slots. Independently choose each arm's minimum complete H200 validation
CE; lower LR breaks an exact tie. Publish and bind all three choices before any
confirmation. No selection on pilots or intermediate diagnostic CE.

Then run nine confirmation trajectories: all three arms for each of three new,
distinct five-role bundles. Each bundle shares the native/common initialization
and sampled training/validation sequence across arms. Code-key variation is
part of the declared nonlexical control variation. Rotate execution order to
avoid always measuring the same arm first. No old scores, selective retries,
unfavorable-seed replacement, partial panels or rate-grid extensions.

Every trajectory has1200updates,1,228,800sampled targets,B8T128,original
paired_blocks512 training and train/validation identities,clip1,unchanged
Muon/AdamW policy and final native architecture. Cooldown starts800 and ends
at.1; spaced half-life200 averaging starts800 at interval10, giving41snapshots.
Keep diagnostic validation at1/301/601/901/1200,10batches each, without advancing
the training sampler or changing weights/global RNG. Final selection/statistics
use all34278validation targets, the102-target tail and111536target bytes.
Holdout remains unencoded and unscored.

## Outcome rules fixed before outcomes

Raw bytes is the predeclared practical candidate. Its mean fresh CE delta
against independently tuned native must be strictly below-.02nat/token; every
fresh paired delta must be negative; all three upper conditional-text bounds
must be negative. Failure of any gate retains the incumbent for this campaign.
All slots and saved-data/resource acceptance must also pass. A separate final
acceptance review precedes any adoption; the controller never edits the
incumbent automatically.

A second predeclared contrast compares raw bytes against nonlexical codes.
Lexical-content support requires negative mean, every fresh delta negative and
all three upper bounds negative. This is a separate claim from practical
native-relative quality. No cost-only adoption or control-to-candidate relabeling.

For both contrasts use paired text blocks2048/4096/8192,including each partial
last block,2000PCG64 resamples,linear quantiles and upper quantile1-.05/6.
The Bonferroni scope is the two contrasts times three block lengths. Resample
the same text block indices across the three trained seeds and average their
per-token differences. These are conditional reused-text intervals given these
three trained bundles, not holdout or population-seed confidence. A separate
scalar aggregation/quantile implementation checks every bootstrap result.

## Complete cost and retention

Retain all18 final RAW/H200 pairs, including losing screens: six native pairs
at512+128MiB and twelve composed pairs at320+80MiB =8640MiB. Add512MiB metadata
and704MiB for serialization overlap/oversized failed writes:9856MiB, inside a
10GiB output cap. Initial free-space requirement is20GiB including8GiB reserve
and2GiB extra headroom. No unperformed retirement credit is allowed. The
completed exact retirement audit `../capacity-recovery001` recovered obsolete
numerical fixture RAW files while preserving all original evidence, selected/
fresh scientific weights and the latest20-update pilot parents.

Pilot maximum update times project to54,460.8seconds for21600updates. Add600s
per trajectory for construction, curves, RAW/H200 publication, validation,
cached checks, serving measurement and saved review; a25% margin gives about
22.66hours. These are estimates, not guarantees. Hard limits remain7200s per
complete slot,4GiB RSS,6GiB sampled MPS driver allocation and8GiB free reserve.
The controller has130800s total (36h20m). Longer-run behavior must remain within
those guards; no fallback, truncation or changed comparisons may hide failure.

One live model is used at a time. Serving costs are descriptive: B1/B8,
full128,prefill96+decode32,two warmups and five measured repeats, with construction
and cache allocation outside timing. Numerical full/cached/reset checks precede
timing. Existing safety checks within each timed consumer remain included.

A failed slot closes that original attempt and stops dispatch. Preserve its
source, roles, logs and payloads. Diagnose engineering failures under standing
authorization with separately identified corrections where needed. A failed
scientific criterion is a result; never rewrite the panel to rescue it.
