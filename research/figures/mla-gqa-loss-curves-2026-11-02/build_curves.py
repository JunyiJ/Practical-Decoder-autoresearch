"""Read historical losses; plot MLA beside GQA without training or checkpoint scoring."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rsi-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
EVIDENCE=ROOT/"research/tokenizer-evidence"
sources={}


def read(path):
    raw=path.read_bytes()
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return raw.decode()


specs=[
    (256,"gqa",1200,[2811,2812,2813],"horizon-001"),
    (256,"gqa",3000,[2811,2812,2813],"horizon-001"),
    (256,"mla",1200,[2911,2912,2913],"mla-001"),
    (512,"gqa",1200,[3411,3412,3413],"wide-attention-001"),
    (512,"mla",1200,[3411,3412,3413],"wide-attention-001"),
]
for campaign in ["horizon-001","mla-001","wide-attention-001"]:
    plan=json.loads(read(ROOT/f"research_runs/tokenizer-{campaign}/plan.json"))
    commit=plan.get("git_commit",plan.get("source_commit"))
    raw=subprocess.check_output(["git","show",f"{commit}:src/train/train.py"],cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest()==plan["source"]["src/train/train.py"]
    source=raw.decode()
    assert "loader.generators[split].manual_seed(loader.eval_seed" in source
    assert "avg_eval_loss, avg_eval_aux_loss = _evaluate(model, loader, 'val', eval_iters)" in source
    sources[f"git:{commit}:src/train/train.py"]=hashlib.sha256(raw).hexdigest()

groups={}; csv_rows=[]; diagnostics=[]; configs={}
for width,family,horizon,seeds,campaign in specs:
    key=f"D{width}-{family}-{horizon}"
    runs=[]
    for seed in seeds:
        stem=f"confirm-{family}-w{width}-s{seed}-"+(f"n{horizon}" if campaign=="horizon-001" else "lr0.00125")
        folder=EVIDENCE/campaign/stem
        rows=[json.loads(line) for line in read(folder/"metrics.jsonl").splitlines()]
        req=json.loads(read(folder/"request.json"))
        result=json.loads(read(folder/"result.json"))
        cfg=req["config"]; t=cfg["training"];m=cfg["model"]
        assert t["max_iters"]==horizon and t["seed"]==seed and t["learning_rate"]==.00125
        assert t["eval_iters"]==10 and t["eval_seed"]==2026 and t["eval_every"]==300
        assert t["batch_size"]==8 and m["block_size"]==128 and m["dim"]==width and m["attn_type"]==family
        assert t["lr_schedule"]["start_step"]==t["weight_averaging"]["start_step"]==2*horizon//3
        assert result["holdout_consumed"] is False
        assert [r["step"] for r in rows]==list(range(0,horizon,300))+[horizon-1]
        original=ROOT/f"research_runs/tokenizer-{campaign}"/stem/"metrics.jsonl"
        assert original.read_bytes()==(folder/"metrics.jsonl").read_bytes()
        configs[width,family,horizon,seed]=cfg
        run=dict(seed=seed,updates=[r["step"]+1 for r in rows],
            train=[r["train_ce_loss"] for r in rows],val=[r["val_ce_loss"] for r in rows],
            raw_final_ce=result["raw"]["mean_ce"],h200_final_ce=result["h200"]["mean_ce"])
        assert np.isfinite(run["train"]).all() and np.isfinite(run["val"]).all()
        runs.append(run)
        k=int(np.argmin(run["val"]))
        diagnostics.append(dict(group=key,seed=seed,lowest_logged_validation_update=run["updates"][k],
            sampled_validation_min=run["val"][k],sampled_validation_final=run["val"][-1],
            validation_strictly_decreases_at_all_logs=bool(np.all(np.diff(run["val"])<0))))
        if horizon==1200:assert np.all(np.diff(run["val"])<0)
        for r in rows:
            csv_rows.append(dict(width=width,family=family,horizon=horizon,seed=seed,log_step=r["step"],
                completed_updates=r["step"]+1,train_minibatch_ce=r["train_ce_loss"],validation_sample_ce=r["val_ce_loss"]))
    groups[key]=dict(width=width,family=family,horizon=horizon,seeds=seeds,campaign=campaign,runs=runs,
        updates=runs[0]["updates"],
        train_mean=np.mean([r["train"] for r in runs],axis=0).tolist(),
        val_mean=np.mean([r["val"] for r in runs],axis=0).tolist(),
        raw_final_mean=float(np.mean([r["raw_final_ce"] for r in runs])),
        h200_final_mean=float(np.mean([r["h200_final_ce"] for r in runs])))

# The direct D512 comparison shares all training settings apart from output paths.
for seed in [3411,3412,3413]:
    a=configs[512,"gqa",1200,seed];b=configs[512,"mla",1200,seed]
    ignore={"checkpoint_dir","metrics_file","summary_file"}
    assert {k:v for k,v in a["training"].items() if k not in ignore}=={k:v for k,v in b["training"].items() if k not in ignore}
    assert a["data"]==b["data"]

BG="#FBFAF6"; INK="#192D3B"; MUTED="#64717A"; GRID="#DDE3E5"
colors={"mla":"#087F78","gqa1200":"#4269A0","gqa3000":"#C15D3C"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False,
    "svg.fonttype":"none","pdf.fonttype":42})


def appearance(g):
    name=g["family"].upper()
    return colors["mla" if name=="MLA" else f"gqa{g['horizon']}"],f"{name} · {g['horizon']:,} updates"


def figure(full=False):
    fig=plt.figure(figsize=(14,10.2))
    fig.text(.06,.956,"MLA / GQA · HISTORICAL LOSS CURVES",size=10,color=colors["mla"],weight="bold")
    fig.text(.06,.905,"MLA validation keeps improving through 1,200 updates" if not full else
             "Complete histories: startup and all individual runs",size=21,weight="bold")
    fig.text(.06,.864,"Peak LR 0.00125 · online model losses · three fresh seeds per recipe · no smoothing or new training",size=11,color=MUTED)
    for col,width in enumerate([256,512]):
        x0=.08+col*.48
        subgroups=[g for g in groups.values() if g["width"]==width]
        fig.text(x0,.812,f"Width {width}"+(" · separate campaigns" if width==256 else " · direct paired comparison"),size=13,weight="bold")
        fig.text(x0,.784,"GQA seeds 2811–2813; MLA 2911–2913" if width==256 else "Both families: seeds 3411–3413",size=10,color=MUTED)
        handles=[]
        for g in subgroups:
            color,label=appearance(g)
            handles.append(Line2D([],[],color=color,lw=2.3,marker="o",ms=4,label=label))
        fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(x0-.008,.765),ncol=1,frameon=False,fontsize=9.5,handlelength=2,borderaxespad=0,labelspacing=.35)
        for row,metric in enumerate(["val","train"]):
            ax=fig.add_axes([x0,.435 if row==0 else .155,.37,.235])
            for g in subgroups:
                color,label=appearance(g)
                sl=slice(None) if full else slice(1,None)
                x=np.array(g["updates"])[sl];arr=np.array([r[metric] for r in g["runs"]])[:,sl]
                if full:
                    for values in arr:ax.plot(x,values,color=color,lw=.8,alpha=.3)
                else:ax.fill_between(x,arr.min(0),arr.max(0),color=color,alpha=.08)
                ax.plot(x,arr.mean(0),color=color,lw=2.2,marker="o" if metric=="val" else "s",ms=4,
                    ls="-" if metric=="val" else "--",label=label)
            ax.axvline(800,color=MUTED,lw=1,ls=":",alpha=.7)
            if width==256:ax.axvline(2000,color=colors["gqa3000"],lw=1,ls=":",alpha=.7)
            ax.set_xlim((0 if full else 250),3220 if width==256 else 1270)
            ax.set_xticks(([0,600,1200,1800,2400,3000] if full else [300,1200,2100,3000]) if width==256 else ([0,300,600,900,1200] if full else [300,600,900,1200]))
            ax.set_ylim((.75,11.2) if full else ((4.4,5.6) if metric=="val" else (1.0,5.3)))
            ax.set_ylabel(("Validation" if metric=="val" else "Training minibatch")+" CE (nats/token)")
            ax.grid(color=GRID,lw=.7);ax.set_axisbelow(True);ax.tick_params(length=0)
            if row==1:ax.set_xlabel("Completed optimizer updates")
            if row==0 and not full:
                if width==256:
                    ax.annotate("GQA 3,000: validation reverses",xy=(2401,groups["D256-gqa-3000"]["val_mean"][-3]),
                        xytext=(1050,5.47),size=9,color=colors["gqa3000"],
                        arrowprops=dict(arrowstyle="->",color=colors["gqa3000"],lw=.9))
                    ax.text(1400,4.46,"MLA ends at 1,200;\nno continuation inferred",color=MUTED,size=9)
                else:
                    ax.annotate("Both still improving at 1,200",xy=(1200,groups["D512-mla-1200"]["val_mean"][-1]),
                        xytext=(690,4.98),size=9,color=colors["mla"],
                        arrowprops=dict(arrowstyle="->",color=colors["mla"],lw=.9))
    fig.text(.06,.035,
        ("Detail begins at update 301; page 2 includes startup. Shading = range across three seeds, not confidence intervals.\n" if not full else
         "Thin lines = individual seeds; bold lines = three-seed means. Startup CE is about 10.8. Lines connect the actual sparse observations.\n")+
        "Validation uses the same 10 sampled batches at each log; training uses one noisy minibatch. These are not full-validation EMA curves.\n"
        "Dotted lines: configured cooldown/EMA start, 800 for short runs and 2,000 for long GQA. D256 is descriptive, with unmatched MLA/GQA seeds.\n"
        "No MLA 3,000-update curve is available in these comparison campaigns. Improving through 1,200 does not establish resistance to later overfitting.",
        size=9,color=MUTED,linespacing=1.5)
    return fig

with PdfPages(OUT/"mla-gqa-training-validation.pdf") as pdf:
    for full,name in [(False,"mla-gqa-loss-comparison"),(True,"mla-gqa-full-histories")]:
        fig=figure(full)
        fig.savefig(OUT/f"{name}.png",dpi=180)
        fig.savefig(OUT/f"{name}.svg")
        pdf.savefig(fig);plt.close(fig)
with (OUT/"loss-history.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
(OUT/"analysis.json").write_text(json.dumps(dict(groups=groups,diagnostics=diagnostics,
    source_sha256=sources,scope="D256 comparison unmatched across campaigns; D512 paired seeds and common training settings. Online fixed-sample validation; holdout unscored.",
    interpretation="No logged validation reversal through 1200 in any of these MLA or short GQA runs. Long GQA reverses. No inference about MLA at 3000 or immunity to overfitting."),indent=2)+"\n")
for k,g in groups.items():
    print(k,"sampled val:",g["val_mean"],"train:",g["train_mean"],"raw:",g["raw_final_mean"],"H200:",g["h200_final_mean"])
print("Saved",len(csv_rows),"observations from",sum(len(g['runs']) for g in groups.values()),"runs")
