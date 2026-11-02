"""Plot historical GQA losses without loading checkpoints or running training."""
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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / "research/tokenizer-evidence/horizon-001"
SEEDS = [2811, 2812, 2813]
HORIZONS = [1200, 3000]
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


plan = json.loads(read(ROOT / "research_runs/tokenizer-horizon-001/plan.json"))
# Check the original producer, rather than inferring log semantics from current code.
trainer = subprocess.check_output(["git", "show", f"{plan['git_commit']}:src/train/train.py"], cwd=ROOT)
assert hashlib.sha256(trainer).hexdigest() == plan["source"]["src/train/train.py"]
producer = trainer.decode()
assert "loader.generators[split].manual_seed(loader.eval_seed" in producer
assert "avg_eval_loss, avg_eval_aux_loss = _evaluate(model, loader, 'val', eval_iters)" in producer
sources[f"git:{plan['git_commit']}:src/train/train.py"] = hashlib.sha256(trainer).hexdigest()
schedule = subprocess.check_output(["git", "show", f"{plan['git_commit']}:src/train/lr_schedule.py"], cwd=ROOT)
assert hashlib.sha256(schedule).hexdigest() == plan["source"]["src/train/lr_schedule.py"]
sources[f"git:{plan['git_commit']}:src/train/lr_schedule.py"] = hashlib.sha256(schedule).hexdigest()

runs = {}
records = []
for horizon in HORIZONS:
    for seed in SEEDS:
        stem = f"confirm-gqa-w256-s{seed}-n{horizon}"
        folder = BASE / stem
        metrics = [json.loads(line) for line in read(folder / "metrics.jsonl").splitlines()]
        request = json.loads(read(folder / "request.json"))
        result = json.loads(read(folder / "result.json"))
        cfg = request["config"]; t = cfg["training"]
        assert t["max_iters"] == horizon and t["seed"] == seed
        assert t["eval_every"] == 300 and t["eval_iters"] == 10 and t["eval_seed"] == 2026
        assert t["learning_rate"] == .00125 and t["batch_size"] == 8
        assert cfg["model"]["block_size"] == 128 and cfg["model"]["dim"] == 256
        assert t["lr_schedule"]["start_step"] == t["weight_averaging"]["start_step"] == 2*horizon//3
        assert result["holdout_consumed"] is False
        assert [r["step"] for r in metrics] == list(range(0,horizon,300))+[horizon-1]
        train = np.array([r["train_ce_loss"] for r in metrics])
        val = np.array([r["val_ce_loss"] for r in metrics])
        assert np.isfinite(train).all() and np.isfinite(val).all()
        # Match the portable evidence to the original retained run logs.
        original = ROOT / "research_runs/tokenizer-horizon-001" / stem / "metrics.jsonl"
        if original.exists():
            assert original.read_bytes() == (folder / "metrics.jsonl").read_bytes()
        x = np.array([r["step"]+1 for r in metrics])
        expected_lr = .00125*(1-.9*np.clip((x-1-2*horizon//3)/(horizon-2*horizon//3-1),0,1))
        assert np.allclose(expected_lr,[r["scheduled_learning_rate"] for r in metrics],rtol=0,atol=1e-15)
        run = dict(updates=x, train=train, val=val,
                   lr=np.array([r["scheduled_learning_rate"] for r in metrics]),
                   raw_final_ce=result["raw"]["mean_ce"], ema_final_ce=result["h200"]["mean_ce"])
        runs[horizon,seed] = run
        for j,row in enumerate(metrics):
            records.append(dict(horizon=horizon, seed=seed, log_step=row["step"],
                completed_updates=int(x[j]), train_minibatch_ce=float(train[j]),
                validation_sample_ce=float(val[j]), scheduled_lr=row["scheduled_learning_rate"]))

summary = {}
for h in HORIZONS:
    train = np.stack([runs[h,s]["train"] for s in SEEDS])
    val = np.stack([runs[h,s]["val"] for s in SEEDS])
    summary[h] = dict(updates=runs[h,SEEDS[0]]["updates"].tolist(),
        train_mean=train.mean(0).tolist(), val_mean=val.mean(0).tolist(),
        train_min=train.min(0).tolist(), train_max=train.max(0).tolist(),
        val_min=val.min(0).tolist(), val_max=val.max(0).tolist(),
        full_validation_raw_mean=float(np.mean([runs[h,s]["raw_final_ce"] for s in SEEDS])),
        full_validation_h200_mean=float(np.mean([runs[h,s]["ema_final_ce"] for s in SEEDS])))

diagnostics = []
for seed in SEEDS:
    r = runs[3000,seed]; k = int(np.argmin(r["val"]))
    assert int(r["updates"][k]) in [901,1201]
    assert np.all(np.diff(r["val"][r["updates"]>=1201])>0)
    assert r["train"][-1] < r["train"][k]
    assert np.all(np.diff(runs[1200,seed]["val"])<0)
    diagnostics.append(dict(seed=seed, long_min_logged_val_update=int(r["updates"][k]),
        long_min_logged_val_ce=float(r["val"][k]), long_final_logged_val_ce=float(r["val"][-1]),
        long_train_ce_at_val_min=float(r["train"][k]), long_final_train_ce=float(r["train"][-1])))

BG="#FBFAF6"; INK="#192D3B"; GRAY="#64717A"; GRID="#DDE3E5"
COLORS={1200:"#087F78",3000:"#C15D3C"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
    "figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":GRAY,"ytick.color":GRAY,
    "axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False,
    "svg.fonttype":"none","pdf.fonttype":42})


def style(ax):
    ax.grid(color=GRID,lw=.7); ax.set_axisbelow(True); ax.tick_params(length=0)


def heading(fig, title, subtitle):
    fig.text(.065,.955,"GQA / TRAINING DURATION",color=COLORS[1200],size=10,weight="bold")
    fig.text(.065,.900,title,size=21,weight="bold")
    fig.text(.065,.855,subtitle,size=10.5,color=GRAY)


def legend(fig, y):
    handles=[]
    for h in HORIZONS:
        for metric,ls in [("validation","-"),("training","--")]:
            handles.append(Line2D([],[],color=COLORS[h],ls=ls,lw=2.3,
                label=f"{h:,}-update recipe · {metric}"))
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.51,y),ncol=2,frameon=False,fontsize=10.5)


fig=plt.figure(figsize=(13,8.8))
heading(fig,"The 3,000-update runs show clear overfitting",
    "Width 256 · LR 0.00125 · fresh seeds 2811 / 2812 / 2813 · means with seed ranges; no smoothing")
ax=fig.add_axes([.09,.33,.83,.43])
for h in [3000,1200]:
    s=summary[h]; x=np.array(s["updates"])[1:]
    for key,ls,marker in [("train","--","s"),("val","-","o")]:
        ax.fill_between(x,np.array(s[key+"_min"])[1:],np.array(s[key+"_max"])[1:],color=COLORS[h],alpha=.08)
        ax.plot(x,np.array(s[key+"_mean"])[1:],color=COLORS[h],ls=ls,lw=2.5,marker=marker,ms=4.5)
for h in HORIZONS:
    start=2*h//3
    ax.axvline(start,color=COLORS[h],ls=":",lw=1.4,alpha=.7)
ax.set_xlim(250,3220); ax.set_ylim(1,5.9)
ax.set_xticks([300,600,900,1200,1500,1800,2100,2400,2700,3000])
ax.set_ylabel("Cross entropy (nats/token)"); style(ax)
ax.annotate("Validation rises",xy=(2701,summary[3000]["val_mean"][-2]),
    xytext=(2090,5.65),color=COLORS[3000],size=11,
    arrowprops=dict(arrowstyle="->",color=COLORS[3000],lw=1))
ax.annotate("Training loss generally falls",xy=(2401,summary[3000]["train_mean"][-3]),
    xytext=(1350,1.16),color=COLORS[3000],size=11,
    arrowprops=dict(arrowstyle="->",color=COLORS[3000],lw=1))
for x,y,text,color in [(3030,summary[3000]["val_mean"][-1],f"{summary[3000]['val_mean'][-1]:.3f}",COLORS[3000]),
    (3030,summary[3000]["train_mean"][-1],f"{summary[3000]['train_mean'][-1]:.3f}",COLORS[3000]),
    (1240,summary[1200]["val_mean"][-1]-.15,f"{summary[1200]['val_mean'][-1]:.3f}",COLORS[1200])]:
    ax.text(x,y,text,color=color,weight="bold",va="center",size=11)
legend(fig,.80)

lr_ax=fig.add_axes([.09,.16,.83,.10])
for h in HORIZONS:
    start=2*h//3
    # Exact LR applied at zero-based iteration i; display completed update i+1.
    updates=np.arange(1,h+1)
    mult=1-.9*np.clip((updates-1-start)/(h-start-1),0,1)
    lr_ax.plot(updates,.00125*mult*1000,color=COLORS[h],lw=2)
    lr_ax.axvline(start,color=COLORS[h],ls=":",lw=1.3)
    lr_ax.text(start+30,1.39,f"{start:,}: cooldown + EMA start",color=COLORS[h],size=9)
lr_ax.set_xlim(250,3220);lr_ax.set_ylim(.05,1.65)
lr_ax.set_yticks([.125,1.25]);lr_ax.set_ylabel("LR (×10⁻³)")
lr_ax.set_xticks([300,600,900,1200,1500,1800,2100,2400,2700,3000]);lr_ax.set_xlabel("Completed optimizer updates")
style(lr_ax)
fig.text(.065,.035,
    "Detail from update 301 onward; full startup and individual seeds are on page 2. Shading = three-seed min–max, not confidence intervals.\n"
    "Train: one minibatch before its update. Validation: the same 10 sampled batches after its update, using online weights (not EMA).\n"
    "Logged about every 300 updates. These curves differ from full-validation final H200 scores: 4.753 (short) vs 5.464 (long).",
    size=9,color=GRAY,linespacing=1.5)

fig2=plt.figure(figsize=(14,10))
heading(fig2,"The validation reversal appears in every long-run seed",
    "Top: detail from update 301 · bottom: complete logged curves including startup · straight lines connect observations")
for col,seed in enumerate(SEEDS):
    for row in [0,1]:
        a=fig2.add_axes([.075+col*.305, .49 if row==0 else .15, .26,.245])
        for h in [3000,1200]:
            r=runs[h,seed]; sl=slice(1,None) if row==0 else slice(None)
            for key,ls,marker in [("train","--","s"),("val","-","o")]:
                a.plot(r["updates"][sl],r[key][sl],color=COLORS[h],ls=ls,lw=1.7,marker=marker,ms=3)
            a.axvline(2*h//3,color=COLORS[h],ls=":",lw=.9,alpha=.65)
        a.set_xlim(250 if row==0 else 0,3100);a.set_ylim((.8,5.65) if row==0 else (.5,11.3))
        a.set_xticks([300,1200,2100,3000] if row==0 else [0,1000,2000,3000]);style(a)
        a.set_title(f"Seed {seed}"+(" · full history" if row else " · detail"),size=12,pad=12)
        a.set_xlabel("Completed updates")
        if col==0:a.set_ylabel("CE (nats/token)")
        if row==0:
            k=int(np.argmin(runs[3000,seed]["val"]))
            a.annotate("Lowest logged validation",xy=(runs[3000,seed]["updates"][k],runs[3000,seed]["val"][k]),
                xytext=(900,5.28),size=8.5,color=COLORS[3000],
                arrowprops=dict(arrowstyle="->",color=COLORS[3000],lw=.8))
legend(fig2,.795)
fig2.text(.065,.035,
    "Dotted lines mark each recipe's configured cooldown and EMA start (800 / 2,000). These are separate runs, not truncations of one trajectory.\n"
    "The 3,000-update sampled validation minimum occurs at update 901 (seed 2811) or 1,201 (2812 / 2813); exact onset is unresolved between logs.\n"
    "Training points are noisy minibatch losses. Validation is a fixed sampled subset. No historical checkpoints were re-scored and the test holdout stays closed.",
    size=9,color=GRAY,linespacing=1.5)

with PdfPages(OUT/"gqa-training-validation-curves.pdf") as pdf:
    for name,f in [("gqa-loss-curves",fig),("gqa-loss-curves-by-seed",fig2)]:
        f.savefig(OUT/f"{name}.png",dpi=180)
        f.savefig(OUT/f"{name}.svg")
        pdf.savefig(f)
        plt.close(f)
with (OUT/"loss-history.csv").open("w",newline="") as fh:
    writer=csv.DictWriter(fh,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
artifact=dict(seeds=SEEDS,summary=summary,diagnostics=diagnostics,source_sha256=sources,
    original_producer_commit=plan["git_commit"],
    interpretation="All three long runs show rising fixed-sample validation loss after their logged minimum, while training minibatch loss falls overall. This is strong evidence of overfitting for this recipe, not a duration-only causal estimate or an optimal-stopping claim.",
    time_semantics="x=logged step+1. Training loss is measured before that update; validation after. Configured cooldown/EMA start markers use recorded start_step.")
(OUT/"analysis.json").write_text(json.dumps(artifact,indent=2)+"\n")
print(json.dumps(dict(diagnostics=diagnostics,means=summary),indent=2))
