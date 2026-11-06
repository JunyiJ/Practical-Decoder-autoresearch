# After MoE: measured boundary and training-free diagnostic

The [MoE result](../../TOKENIZER_MOE_RESULTS.md) is CLOSED_NOT_ADOPTED. Closure
verified every original acceptance/guard/hash,36weights10555774716B andall1035
sources;106fresh/finalmetadatafiles mirroredZEROweights. No numerical worker or
controller is active. All200110screen/evaluation/fresh/bootstraproles areUSED.
Keep native dense incumbent; no wide fallback. OriginalSTOP closesALLcampaign
commands,which must never be replayed. No original rescoring/training/bootstrap.

## Local facts and candidate review

Existing-analysis001 examines only accepted savedmetadata. MoE is worse onall3
freshseeds versusbothcontrols andmosttextblocks. Noexpertwasunused;everyMoE
parameter received a gradient onall1200updates inthispanel. Assignmentfractions
.2111-.2953 do not implyspecialization. Freshmeantraining271.21/288.21/317.12s
(dense/wide/MoE),clippedupdates16/24/25,2/7/7,15/19/17. Sparsecurves andRAW/H200
CEdifferences do notidentify overfit,noise-limited training,EMAorhorizonoptima.
Thus sharedexperts/routerzloss/expertcount/longerhorizon rescue is deferred;
there is no observed starvation/instability mechanism demanding it.

The historical character [batch results](../../BATCH_SENSITIVITY_RESULTS.md)
provide a separate unresolvedtransfer question. B16/B32 failed all-width quality
neutrality,eachworsened someD256seed,despite D256 meanchanges−.002558/−.002290 and
lowertrainingcost. That MHA/character/3000-update recipe differsfromcurrent
nativeBPE/MLA/Muon-AdamW/1200schedule. It changedupdatecount,momenttime scales,
LRanddecay jointly. Never poolits scores orreopen its grid. Current sparsecurves/
clipping doNOTestablish that changingbatchsize helps. Defer a qualitypanel.

## Primary paper actually read

McCandlish etal.,[arXiv:1812.06162v1](https://arxiv.org/html/1812.06162v1),
§2.2equations2.1–2.10/independencefootnote,§2.4caveats,AppendixA.1equationsA.1–A.2
andratio-estimator discussion,AppendixE.3 were retrieved/read. The paper connects
raw gradient variability to local optimization efficiency;it warns about local
approximations,conditioning,tunedLR,correlatedsamples andgeneralization. Its
signal/noise component estimators canbeunbiased whiletheirratioisbiased.
Our paired512correlatedtokens,Muonorthogonalization,Adampreconditioning,small
reusedcorpus andendpoint-only sampling invalidate an automaticcritical-batch or
quality prediction. The measurement unit will be independent B8paired512draws,
notIIDtokens. This gap is explicit; publishedscaling is notlocalgain.

## Admit only a bounded fixed-weight gradient diagnostic

Unresolvedassumption: how variable areuntransformedgradientvectors under the
actual private paired512B8sampler at frozennative endweights? Capacity/routing
losses do not answerthis. A highreplicatedtrace-covariance relative tocorrected
mean-gradientnormwould motivate furtheroptimization feasibility,notbatchadoption;
smallnoise orundefined/nonpositivesignal would weaken that local rationale.
A rawgradient statistic isnot aMuon/Adamupdate statistic orageneralizationmetric.
Both outcomes add information without a fullcampaign orarbitrarygrid.

Prospective probe: all3RAWdensefresh endpoints fromtheclosednewMoEpanel,32NEW
privateB8trainingdrawseach,98304targetpositions total,nooptimizerupdate,validation
scoring,bootstrap,holdout orweightfiles. Sameprobe draws across the3weights;
newfixture roles must bechecked beforeexecution. Use ownmodel/codec/signature,
loadexactboundstate,prove model/source/data/fileandCPU-MPS RNGimmutability and
complete-gradientcoverage. Allfloatinggradientsstream toCPUFP64moments; do not
retain32fullgradienttrees. Reportperparameter/group/all moments,corrected
signal andbiasedratio;nonpositivesignal yieldsNone,neverclamp apositiveanswer.
Do not callthe ratio anoptimalbatchsize orselectbatch/beta/LRfromit.

Completeprospective outputbudget16MiBmetadata+8GiBreserve+1267904512external+
1458176entry=>9876074496Bfloor,target12023558144B;~80.5GBfree passes. No tensors
willbewritten;one~114MBlearnedmodel/gradient pair andone~228MBFP64mean tree,
per-largest-parameter temporary buffers andloader state must fitoriginal4GiB/
6GiB/600sactualguards. SavedRAWpayload~445MBmustbereleasedbeforegradientwork;
these areforecasts,not anactualMPSmemorymeasurement. Full3endpointprobe and
allcorrectness/source/privateRNGcontracts precede execution;noqualitypanelbudget,
scientificseedassignment orfreeze exists. Allweightsoriginalpathsprotected.

## Concrete completed prerequisite

New tokenizer_gradient_noise_v1(twofiles) plusgradient-component001.py preserve
1035predecessors=>1038currentbindings. GeneratedCPUFP64/FP32ordinary,constant and
zero-mean cases passed independent stacked-vector covariance/mean arithmetic
andA.2identities,maximumerror3.55e-15. Thirty malformedupdates rejected before
mutation;undefinedsignalhandledwithoutforcedpositiveratio.GlobalCPURNGunchanged.
Fixture600490only,no corpus/model/optimizerstep/weight/qualityseed. Originalguard
passed;source/plan/resultnowimmutable.Neverrerun/editthis proof.

NEXT implement/prove actual ownfixed-weight gradient sampling/streaming model
integration and CPU/MPS equations before the3endpointtrain-only probe. This is
a bounded diagnostic,notqualityexploration orbatchpanel. No automaticbeta1,
batch,dropout,decay,horizon,rank/routergrid;70/20/10remainsPROPOSED. Continue
usefulnumerical prerequisites underoriginalauthority,withoutredundantconsent.
