# Tokenizer architecture research agenda

Human steering, October 26, 2026 (Los Angeles): research must explore model
architecture, including attention, MoE, activation, normalization and residual
layers, alongside hyperparameter tuning. The human clarified: "not only focus
on HP tuning." Both are first-class research directions; this is not a mandate
to replace tuning with architecture work. This governs future
rounds alongside the [attention roadmap](TOKENIZER_ATTENTION_ROADMAP.md).
The running, frozen LR transfer experiment finishes unchanged to establish a
reviewed subword baseline. Choose subsequent architecture or hyperparameter
experiments from evidence, while ensuring meaningful architectural exploration.

## Scheduling and decisions

The proposed architecture sequence is inference-aware GQA/MLA, followed by the Kimi
KDA/KDA/KDA/MLA hybrid. Then prioritize a finite sparse-versus-dense FFN study
and a finite activation/normalization/residual study, choosing each specific
hypothesis from reviewed evidence and implementation feasibility. Do not let
attention-only refinements indefinitely displace the other architecture families.
After each completed architecture round, explicitly review the next untested
family below. Record why it is selected, deferred or rejected; this agenda is
not permission to launch all candidates at once.

Each round needs a concrete hypothesis, exact configuration and, for architecture
changes, equations and layout,
matched controls, resource pilots, a frozen finite plan, fresh-seed confirmation
and an adoption rule declared before results. Use equally bounded optimization
tuning to avoid handicapping a family. Standalone hyperparameter rounds remain
valid when motivated by evidence, including LR, schedule, horizon and regularization
questions. No architecture-only ordering or extra permission is required. Review
both kinds of hypotheses at each round boundary and record the selection rationale.
Fix correctness or resource failures before continuing research;
preserve the failed evidence. Implementation tests do not establish quality.

## Architecture families and discriminating comparisons

These are local proposals, not claims of existing tokenizer results or paper
reproductions. Freeze exact candidate values separately before training.

| Family | First structural question | Essential controls and measurements |
|---|---|---|
| Attention | Can GQA or compressed MLA retain quality with cheaper inference? | Fresh MHA control; actual KV memory, prefill/decode timing and backend; use the attention roadmap's prospective neutrality and cost rules |
| Recurrent/global hybrids | Does three KDA layers plus one MLA layer preserve global recall better than all-KDA at useful state cost? | Full MLA and full KDA controls, explicit Linear/K3 semantics, layerwise caches; version/ratio ablations follow the initial 3:1 comparison |
| Sparse MoE | Does routed expert capacity improve subword quality for a comparable active FFN budget? | Dense SwiGLU control; start from a small four-expert/top-two proposal, derive expert widths before launch; report active and total parameters, router/shared-expert overhead, actual time/memory, routing entropy/load, unused experts and auxiliary loss separately from validation CE |
| Shared experts and layer placement | Is shared dense capacity useful alongside routed experts, or should only selected layers use MoE? | Separate follow-up to the sparse baseline; account for always-active shared work and vary one placement/design choice at a time |
| Activation and FFN structure | Does the tokenizer change the relative behavior of SwiGLU, a GELU gate or an ungated dense FFN? | Isolate gate nonlinearity with identical shapes first; a gated-versus-ungated comparison separately matches matrix budget as closely as possible and reports rounding/bias differences |
| Normalization | Does block RMSNorm versus LayerNorm, or additional Q/K or FFN normalization, improve stability and quality? | Separate block norm, final norm and internal norm changes; specify placement, epsilon, affine parameters and optimizer treatment; track activation/gradient scales without assuming they explain a CE gain |
| Residual topology | Does serial versus parallel attention/FFN composition help? | Specify which normalized stream each branch consumes and whether norms are shared; separate topology from fixed or learned branch gains |
| Residual scaling and depth | Can depth-aware branch scaling make additional layers useful? | First test scaling at fixed depth, then a bounded depth/width allocation; report total and non-embedding parameters and measured compute, not just equal token counts |
| Embedding/readout | Does tying the large BPE input/output tables improve the quality/memory tradeoff? | Untied control; explicit initialization, optimizer grouping and readout scaling contract; count saved parameters and measured memory; tying also changes expressivity |
| Position and local mixing | Do positions, local/global attention placement or short convolutions improve the selected architecture? | Separate structural ablations, declared context and cache semantics; no long-context claim from context128 results |

The existing repository contains GQA/MLA/KDA modules, routed/shared experts,
SwiGLU gate alternatives, norm and residual variants, and tying/depth helpers.
Their existence does not mean the inherited tokenizer policy permits combining
them. Validate and extend explicit model/optimizer/checkpoint contracts for each
new design. Do not disable historical guards or revive paused null-attention or
permanently stopped campaign006 to access a candidate.

## Fair comparisons and evidence

Retain the reviewed character learnings as priors. A failed character experiment
can motivate a targeted control or narrow a hypothesis; it does not establish a
subword result. Revisit it only with a stated tokenizer-related or structural
reason. Do not pool character and subword CE, or select the luckiest old seed.

Equal training tokens, active parameters, total parameters and wall time are
different budgets. Declare the main comparison axis and report the others.
MoE especially needs both active and total capacity accounting, including the
large shared vocabulary tables. Architecture changes alter RNG consumption;
paired seeds alone do not imply identical shared initial weights.

Keep the test holdout closed and record adaptive validation reuse. Confirm
selected candidates on fresh seeds and the claimed widths; report reversals,
failures and limitations. Quality-neutral GQA/MLA can qualify through the
predeclared measured inference-cost route. Other families need their own
prospective quality/cost rule; do not invent one after seeing favorable numbers.

Validate gradients, causality and the affected cache, optimizer, tokenizer,
checkpoint/resume and EMA behavior before serial MPS pilots. Freeze source
before each finite training round; preserve checkpoints and raw evidence.
Continue with one local worker and bounded disk, memory and time guards. Publish
completed iterations, substantive learnings and failures, then advance to the
next evidence-driven architecture or hyperparameter hypothesis until the human
stops the campaign.
