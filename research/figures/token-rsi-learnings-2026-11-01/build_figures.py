"""Render a sourced, descriptive summary; never train or score a model.

Run with Python + matplotlib + numpy. Outputs PNG, SVG, a four-page PDF and
the exact chart data/source hashes. Evidence is read from this repository.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rsi-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EVIDENCE = ROOT / "research/tokenizer-evidence"
SOURCES = {}


def read(relative):
    path = EVIDENCE / relative
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def interval4096(items):
    return next(x for x in items if x.get("text_block_tokens", x.get("block_tokens")) == 4096)


lr = read("quality-001/final-review.json")["comparisons"]
horizon = read("horizon-001/final-review.json")
gqa = read("gqa-001/final-review.json")
mla = read("mla-001/final-review.json")
wide = read("wide-attention-001/final-review.json")
table = read("table-001/final-review.json")
hybrid = read("hybrid-001/final-review.json")

attention = []
for i, width in enumerate([64, 256]):
    q = interval4096(gqa["quality"][i]["intervals"])
    t = next(v for v in gqa["inference"] if v["width"] == width and v["batch"] == 1)
    attention.append(dict(label=f"GQA vs MHA · D{width}\n3,000 updates", delta=q["mean_delta"],
        ci=q["conditional_text_bootstrap_95pct"], cache=50., decode=(t["decode_median_ratio"]-1)*100,
        adopted=width == 256))
attention.append(dict(label="MLA vs MHA · D256\n1,200 updates", delta=mla["mean_delta"],
    ci=interval4096(mla["intervals"])["conditional_text_bootstrap_95pct"], cache=81.25,
    decode=(mla["inference"][0]["decode_median_ratio"]-1)*100, adopted=True))
wq = interval4096(wide["quality"]["intervals"])
attention.append(dict(label="MLA vs GQA · D512\n1,200 updates", delta=wq["mean_delta"],
    ci=wq["conditional_text_bootstrap_95pct"], cache=68.75,
    decode=(wide["median_of_seed_median_ratios"]["1"]["mla_vs_gqa"]["decode_seconds"]-1)*100,
    adopted=True))

recent = []
activation = read("activation-preparation-001/inference-recovery-quality-prerequisite-001-analysis.json")
read("activation-preparation-001/inference-recovery002-closed-review.json")


def find_stats(obj):
    if isinstance(obj, dict):
        if "seed_deltas" in obj and "bounds" in obj and "mean_delta" in obj:
            return obj
        for value in obj.values():
            found = find_stats(value)
            if found is not None:
                return found
    return None


recent.append(dict(label="Exact GELU gate vs SiLU", **find_stats(activation)))
for name, label in [
    ("block-norm", "LayerNorm vs RMSNorm"),
    ("ema-horizon", "EMA half-life 100 vs 200"),
    ("ffn-structure", "Ungated SiLU1008 vs SwiGLU672"),
    ("mla-score", "Attention multiplier 1.0 vs 0.5"),
    ("ffn-norm", "FFN product RMSNorm vs none"),
    ("warmup", "100-update warmup vs none"),
    ("local-mix", "Causal FFN mixing vs none"),
    ("decay", "Zero weight decay vs 0.00006"),
]:
    d = read(f"{name}-preparation-001/quality001-final-review.json")
    assert d["quality_adopted"] is False
    assert abs(np.mean(d["statistics"]["seed_deltas"])-d["statistics"]["mean_delta"]) < 1e-10
    recent.append(dict(label=label, **d["statistics"]))

# The wider hybrid's closed review is summarized in this append-only record.
# Retain the exact quoted section alongside its source hash for provenance.
path = ROOT / "research/TOKENIZER_WIDE_HYBRID_PREPARATION.md"
raw = path.read_bytes()
SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
section = raw.decode().split("### Campaign005 closed: hybrid fails both adoption routes", 1)[1].split("### Next FFN direction", 1)[0]
hybrid_wide = dict(hybrid_ce=4.985189831510325, gated_mla_ce=4.8429360153472585,
    kda_ce=4.96329048568082, hybrid_state_bytes=399360, gated_mla_state_bytes=327680,
    kda_state_bytes=335872, source_excerpt=section)
for v in [hybrid_wide["hybrid_ce"], hybrid_wide["gated_mla_ce"], hybrid_wide["kda_ce"]]:
    assert str(v) in section

BG = "#FBFAF6"
INK = "#192D3B"
MUTED = "#5F6C76"
GRID = "#DCE2E4"
GREEN = "#087F78"
BLUE = "#4269A0"
RED = "#BE5A43"
GOLD = "#A47625"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": INK, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "svg.fonttype": "none", "pdf.fonttype": 42})


def title(fig, num, headline, subtitle):
    fig.text(.055, .956, f"{num:02d}  /  TOKEN-BASED LLM RSI", color=GREEN, size=10, weight="bold")
    fig.text(.055, .900, headline, size=21, weight="bold")
    fig.text(.055, .858, subtitle, size=10.5, color=MUTED)


def footer(fig, text):
    fig.text(.055, .035, text, size=9, color=MUTED, va="bottom", linespacing=1.5)
    fig.text(.945, .956, "Through Nov 1, 2026 · Los Angeles", size=9, color=MUTED, ha="right")


def clean(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=.7)
    ax.tick_params(length=0)


figures = []

# 1: Matched within-experiment gains, not a pooled trajectory.
fig = plt.figure(figsize=(13, 7.7))
title(fig, 1, "The largest early gains came from the training recipe",
      "Three fresh paired seeds per comparison · GPT-2 BPE on Tiny Shakespeare · lower validation CE is better")
axes = [fig.add_axes([.08, .28, .23, .45]), fig.add_axes([.40, .28, .23, .45]), fig.add_axes([.72, .28, .23, .45])]
for ax, comp in zip(axes[:2], lr):
    vals = [comp["baseline_mean_ce"], comp["candidate_mean_ce"]]
    ax.plot([0, 1], vals, color=GREEN, lw=3, marker="o", ms=9)
    for x, val in enumerate(vals):
        ax.annotate(f"{val:.3f}", (x,val), xytext=(0,12), textcoords="offset points", ha="center", weight="bold")
    ax.set_xticks([0,1], ["LR 0.01", "LR 0.0025"])
    ax.set_xlim(-.25,1.25); ax.set_ylim(4.5, 7.4)
    ax.set_ylabel("Mean validation CE (nats/token)")
    ax.set_title(f"MHA · width {comp['width']} · 3,000 updates", size=11, pad=18)
    clean(ax, "y")
    ax.text(.5, -.28, f"Δ CE = {comp['mean_delta']:+.3f}", transform=ax.transAxes, ha="center", color=GREEN, size=15, weight="bold")
ax = axes[2]
vals = [horizon["mean_h200_ce"]["3000"], horizon["mean_h200_ce"]["1200"]]
ax.plot([0,1], vals, color=GREEN, lw=3, marker="o", ms=9)
for x, val in enumerate(vals):
    ax.annotate(f"{val:.3f}", (x,val), xytext=(0,12), textcoords="offset points", ha="center", weight="bold")
ax.set_xticks([0,1], ["3,000 updates", "1,200 updates"])
ax.set_xlim(-.25,1.25); ax.set_ylim(4.5,7.4)
ax.set_ylabel("Mean validation CE (nats/token)")
ax.set_title("GQA · width 256 · LR 0.00125", size=11, pad=18)
clean(ax,"y")
ax.text(.5, -.28, f"Δ CE = {horizon['mean_delta']:+.3f}", transform=ax.transAxes, ha="center", color=GREEN, size=15, weight="bold")
fig.text(.835, .105, "Training time: 504 → 202 s (−60%)", ha="center", color=GREEN, size=10)
footer(fig, "Separate controlled comparisons; do not add their gains. Horizon, cooldown and EMA start changed together.\nH200 = EMA half-life 200 updates. Means use reused validation text; test holdout remains unscored. Sources: quality-001; horizon-001.")
figures.append(("01-training-recipe", fig))

# 2: Show the independent axes of quality, memory, and latency.
fig = plt.figure(figsize=(13, 7.8))
title(fig, 2, "A smaller attention cache did not mean faster decoding",
      "Each row is a separate fresh comparison; GQA's quality result also changed with model width")
axs = [fig.add_axes([.285,.25,.255,.48]), fig.add_axes([.59,.25,.14,.48]), fig.add_axes([.79,.25,.16,.48])]
for ax in axs:
    ax.set_ylim(3.6,-.6); clean(ax)
    ax.set_yticks(range(4)); ax.tick_params(axis="y", labelleft=False)
    ax.axvline(0, color=MUTED, lw=1)
axs[0].tick_params(axis="y", labelleft=True, pad=14)
axs[0].set_yticklabels([x["label"] for x in attention], size=10.5)
for i,r in enumerate(attention):
    c = GREEN if r["adopted"] else RED
    lo,hi = r["ci"]
    axs[0].plot([lo,hi],[i,i], color=c,lw=2)
    axs[0].scatter(r["delta"], i, color=c, marker="D" if r["adopted"] else "X", s=65,zorder=3)
    axs[0].annotate(f"{r['delta']:+.4f}",(r["delta"],i),xytext=(0,14),textcoords="offset points",ha="center",color=c,size=10)
    for ax,key,fmt,col in [(axs[1],"cache","{:.2f}%",BLUE),(axs[2],"decode","+{:.1f}%",RED)]:
        ax.barh(i,r[key],color=col,alpha=.85,height=.28)
        ax.text(r[key]+1.3,i,fmt.format(r[key]),va="center",size=10)
axs[0].set_xlim(-.085,.052); axs[0].set_xticks([-.08,-.04,0,.04])
axs[0].set_title("Quality change",size=12,pad=22)
axs[0].set_xlabel("Δ CE (nats/token)\n← better quality",labelpad=12)
axs[1].set_xlim(0,111); axs[1].set_xticks([0,50,100]); axs[1].set_title("Cache saved",size=12,pad=22)
axs[1].set_xlabel("Allocated KV bytes (%)",labelpad=12)
axs[2].set_xlim(0,30); axs[2].set_xticks([0,10,20,30]); axs[2].set_title("Decode latency increase",size=12,pad=22)
axs[2].set_xlabel("Batch 1 (%)\nlower is better",labelpad=12)
fig.legend(handles=[Line2D([],[],marker="D",color=GREEN,ls="",label="Scoped adoption"),Line2D([],[],marker="X",color=RED,ls="",label="Not adopted")],loc="lower left",bbox_to_anchor=(.27,.12),ncol=2,frameon=False)
footer(fig,"Whiskers: conditional 95% text-block intervals (4,096-token blocks), not seed-population uncertainty.\nFP32 MPS, context 128, 32-token prefill / 64-token decode; two models resident. Cache ≠ total memory. Sources: GQA, MLA, wide-attention final reviews.")
figures.append(("02-attention-tradeoffs",fig))

# 3: Every seed and the practical threshold, with conditional-text uncertainty.
fig = plt.figure(figsize=(13,9.4))
title(fig,3,"Later refinements found small gains, but no new adoption",
      "Width 256 RoPE MLA · 1,200 updates · nine completed follow-ups, shown in research order")
ax = fig.add_axes([.34,.23,.51,.55])
ax.set_xlim(-.024,.020); ax.set_ylim(8.7,-.9)
ax.axvspan(-.024,-.020,color=GREEN,alpha=.07)
ax.axvline(-.020,color=GREEN,ls="--",lw=1.5)
ax.axvline(0,color=MUTED,lw=1)
for i,r in enumerate(recent):
    m = r["mean_delta"]; c = GREEN if m<0 else RED
    lo,hi = r["bounds"]["4096"]["descriptive_95pct"]
    ax.plot([lo,hi],[i,i],color=c,lw=2,zorder=2)
    ax.scatter(r["seed_deltas"], np.array([i]*3)+[-.18,0,.18],s=26,facecolors=BG,edgecolors=MUTED,zorder=3)
    ax.scatter(m,i,s=65,marker="D",color=c,zorder=4)
    ax.text(.0215,i,f"{m:+.4f}",va="center",color=c,weight="bold",clip_on=False)
ax.set_yticks(range(9),[r["label"] for r in recent]); ax.tick_params(axis="y",pad=12)
ax.set_xticks([-.02,-.01,0,.01,.02]); clean(ax)
ax.set_xlabel("Candidate − control validation CE (nats/token)    ← better | worse →",labelpad=13)
ax.text(-.020, -.99,"Required mean < −0.020",ha="center",size=10,color=GREEN)
ax.text(.0215,-.99,"Mean Δ",size=10,color=MUTED,weight="bold")
fig.legend(handles=[Line2D([],[],marker="D",ls="",color=INK,label="Three-seed mean"),
    Line2D([],[],marker="o",ls="",mfc=BG,color=MUTED,label="Individual paired seed"),
    Line2D([],[],color=INK,lw=2,label="Conditional 95% text interval")],
    loc="lower left",bbox_to_anchor=(.30,.135),frameon=False,ncol=3,fontsize=9.5)
footer(fig,"Attention scale, EMA half-life and LayerNorm improved all three seeds, but missed the predeclared minimum gain. No effects are added together.\nWhiskers use 4,096-token blocks; adoption also required every seed and all simultaneous text bounds to improve. Intervals are not holdout confidence.\nEqual per-arm LR searches where applicable; EMA uses the same online trajectory, and decay uses fixed common LR 0.00125. Sources: nine final quality reviews.")
figures.append(("03-later-refinements",fig))

# 4: Two distinct architecture lessons, deliberately retaining matched controls.
fig = plt.figure(figsize=(13,8.4))
title(fig,4,"Compression and hybridization still had to pass quality gates",
      "Neither vocabulary-table sharing nor the tested 3:1 KDA / gated-MLA hybrids earned adoption")
left = fig.add_axes([.31,.50,.61,.23])
keys = ["matched_untied_vs_legacy_untied","scaled_tied_vs_legacy_untied","scaled_tied_vs_matched_untied"]
labels = ["Matched untied − legacy untied","Scaled tied − legacy untied","Scaled tied − matched untied"]
for i,key in enumerate(keys):
    r = interval4096(table["quality"]["comparisons"][key]["intervals"])
    m=r["mean_delta"]; lo,hi=r["conditional_text_bootstrap_95pct"]
    left.barh(i,m,color=RED,height=.36,alpha=.85)
    left.plot([lo,hi],[i,i],color=INK,lw=1.6)
    left.text(hi+.004,i,f"{m:+.4f}",va="center",size=10,weight="bold")
left.set_yticks(range(3),labels); left.set_ylim(2.6,-.7); left.set_xlim(0,.18)
left.set_title("A. Table sharing · width 256 GQA · 1,200 updates",loc="left",size=12,pad=17)
left.set_xlabel("Δ validation CE (nats/token) · positive is worse",labelpad=7); clean(left)
fig.text(.31,.407,"45.0% less parameter storage; +0.1332 CE vs legacy. Only +0.0305 is the matched tying contrast.",size=10,color=MUTED)
bottom=fig.add_axes([.31,.17,.61,.16])
vals=[hybrid["mean_delta"],hybrid_wide["hybrid_ce"]-hybrid_wide["kda_ce"],hybrid_wide["hybrid_ce"]-hybrid_wide["gated_mla_ce"]]
labels=["D64 hybrid − all-KDA","D512 hybrid − all-KDA","D512 hybrid − gated MLA"]
bottom.barh(range(3), vals, color=GOLD,height=.37)
for i,v in enumerate(vals): bottom.text(v+.004,i,f"{v:+.4f}",va="center",weight="bold",size=10)
bottom.set_yticks(range(3),labels); bottom.set_xlim(0,.18); bottom.set_ylim(2.6,-.7)
bottom.set_title("B. NoPE hybrids · three fresh seeds per width",loc="left",size=12,pad=14)
bottom.set_xlabel("Δ validation CE (nats/token) · positive is worse",labelpad=7); clean(bottom)
footer(fig,"D512 hybrid also used 21.9% more state than gated MLA and 18.9% more than KDA at context 128. These recipes are local adaptations.\nPanel A whiskers: conditional 95% text intervals; Panel B: means only. No general rejection of tying/hybrids. Sources: table-001; hybrid-001; wider-hybrid closure.")
figures.append(("04-architecture-lessons",fig))

with PdfPages(OUT / "token-rsi-major-learnings.pdf") as pdf:
    for name,fig in figures:
        fig.savefig(OUT / f"{name}.png",dpi=180)
        fig.savefig(OUT / f"{name}.svg")
        pdf.savefig(fig)
        plt.close(fig)

data = dict(as_of="2026-11-01 America/Los_Angeles", lr_comparisons=lr,
    horizon={k:horizon[k] for k in ["mean_h200_ce","mean_delta","mean_training_seconds","paired_seed_deltas"]},
    attention=attention, later_refinements=recent, hybrid_wide=hybrid_wide,
    table_comparisons=table["quality"]["comparisons"], hybrid_small_mean_delta=hybrid["mean_delta"],
    source_sha256=SOURCES, scope="GPT-2-tokenized Tiny Shakespeare, reused validation, three fresh seeds per comparison; no test scoring.")
(OUT / "figure-data.json").write_text(json.dumps(data,indent=2)+"\n")
print(f"Wrote {len(figures)} figures (PNG/SVG), four-page PDF, and sourced data to {OUT}")
