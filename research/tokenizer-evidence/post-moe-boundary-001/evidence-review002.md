# Fixed-weight native gradient integration and first diagnostic complete

The new four-file tokenizer_gradient_probe_v1 namespace andgradient-model001.py
preserve1038predecessors=>1043currentbindings. Exactroleaudit foundno prior
standalone600491–600494matches beforecreating newcode. Integration uses600491
forgeneratedinputs/train-samplerproof,600494forCPUinitializer,probe600492forprivate
trainingdraws/600493forCPUinitializer. AllareUSEDfixtures,notqualityseeds.
CPUinitializer usesonlydefaultCPUgenerator withintransactionalCPURNGfork;
there is no implicit torch.manual_seed reseeding ofMPS. Allsources nowimmutable.

## Integration actually passed

One original600s/RSS4GiB/MPS6GiB/reserve8GiBguard coversallgeneratedandrealchecks.
GeneratedfullCPU/MPSgradients agreed(max1.04308e-7),andtwoB1microgradients averaged
totheB2gradient(max1.64146e-8),all88learnedgradients. Teninvalidinput/mode/type
calls rejectedbeforeexistinggradientmutation. EighteenRAWmetadataforgeries
rejected;the newexplicitconsumer extractsweightsONLYfromboundacceptedownRAW,
neverrestoresoptimizer/EMA/sampler/globalRNG. Trainloader encodesonlyoriginal
trainingprefix,withonlytrainIDs/generator;no validationorholdoutencoding.

All3actualRAWdensefreshcontrols09/14/16 passedonefullB8train-gradientCPU/MPS
pair andindependentprivatepaired512replay. Maximumrealgradientdifference
1.6173348e-5withinpredeclared4e-5atol/4e-4rtol;CEdifferences<=2.55108e-5within3e-5.
Afull227939328BFP64meantree stress-testedtheactualbufferlayout;repeatedcopiesof
onegradientwereexplicitlyNOTindependentdraws/noise evidence. Fixedmodelstate,
files,data andglobalCPU/MPSRNGunchanged. RAWpayloadreleasedbeforegradients.
Integration10.20s/peakRSS2706522112/MPS2468397056Bpassed. ZEROoptimizerupdates,
weightfiles,validation/holdout,qualityseed ororiginalclosedcommandreplay.

## Full first diagnostic actually completed

NEWrootresearch_runs/tokenizer-gradient-noise-diagnostic-v1-001 terminalSTOP
complete_training_free_gradient_diagnostic closesALLitscommands.32privateB8draws
onEACHof3fixedRAWmodels,98304targetevaluations,all88gradients,CPUFP64Welford
moments andperinput/readout/bodygroups. All3weights were measured withsameprivate
input/targetstream;these arecontrolledsharedprobeinputs,not3independentprobe
cohorts. No clipping/optimizer/EMA/validation/holdout/bootstrap/tensorwrites.
GlobalCPU/MPSRNG,models,3originalRAWhashesandall1043sources unchanged.
Guard27.43s/RSS2122596352/MPS2528002048Bpassed. No worker/session active.

| RAW control slot | Empirical trace covariance | Corrected squared signal | Biased ratio,B8drawunits |
|---|---:|---:|---:|
|09|.697247634|.010048972|69.384970|
|14|.723057415|.015949105|45.335296|
|16|.731820760|.010092514|72.511245|

Independentmetadatareview recomputedperparameter covariance via savedperdraw
squarednorms andfinalmean norms,maximumidentityerror2.78e-17. All3complete32draw
streams andaggregate/groupinventories verified. Sevenclosedmetadatafiles905403B
mirroredZEROweights. No gradients/model/scoring/bootstrapreplayed;originalSTOP
andallpreviousscientificevidence remainunchanged. Seegradient-probe001-review001.

## Interpretation and next discriminating work

The raw-gradient covariance is large comparedtothecorrectedmean-squaredsignal
atthese endpoints. ItdoesNOTidentifyanoptimalbatchsize,Muon/Adamupdatenoise or
quality/generalizationgain. RawEuclideanbodygradientmoments dominate;theirscale
andgrouping arenotacausalcapacity/bottleneckdecomposition. No batch/beta/LR
change ornewqualitypanel isadmitted. Originalhistoricalbatchcontraryresults and
McCandlish1812.06162v1§2.2/2.4/A.1/E.3 scope/ratio-biascaveats remain.

The correction subtracts58.6–69.4percentofthefinite32drawmean-gradientnorm.
Thedenominatorandratio maythusvarywithnewprobesamples;the current3models share
probeinputs anddonotresolveindependentcohortstability. SelectONLYaNEWbounded
cohortdiagnostic,notabatchcampaign. Predeclare4NEWindependentprivatecohorts×32
B8draws perall3fixedRAWmodels,combined128draws,393216targetevaluations. Keepall
cohorts/reportnegativeorundefinedsignalswithoutrescue;excludefirst600492probe
fromnewcombinedstatistics. Ifstronglyvariableorundefined,thenoisebasisremains
weak;stablecombinedmoments justifyreviewingoptimizationfeasibilityonly.

See cohort-design001.json. NEWcohort/initializerfixture rolesUNASSIGNED;newsource
namespace/root/sourcecontracts/unusedroleaudit andLIVEbudget required. Sources/
STOPs/weights fromthefirstdiagnostic/component/integration mustNEVERbererun,
editedorrestored. TwoCPUFP64meantrees455878656B,sequentialendpoints,projected
metadata5718764Bwithin16MiB;conservative2×(4×27.43s+4×27.43sallowanceforsecond
momentpass)=438.84s beloworiginal600s,notaguarantee. Fullmetadata/reserve/external/
entryfloor9876074496B,target12023558144Bfits~80.5GBfree;RSS4GiB/MPS6GiBguards
remainmandatory. No storageblocker/cleanup,partialendpoint/cohortpanel orcredit
fromold/futuredeletions. This remainsfixedweightdiagnosis,notqualityexploration.
