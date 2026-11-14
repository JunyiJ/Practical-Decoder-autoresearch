"""Render retained fixed-probe curves with a separate plotting environment."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'tokenizer-evidence/general-rsi-cross-size-quality-001'
rows=json.loads((P/'probe-curves.json').read_text())['rows']
fig,axes=plt.subplots(3,3,figsize=(13,10),sharey=True,layout='constrained')
colors={'native':'#2369a1','raw_bytes':'#c65c19'}
styles={1:'-',2:'--',3:':'}
for i,depth in enumerate([4,8,16]):
    for j,(field,scale,label) in enumerate([('targets',1e6,'Training targets (millions)'),('estimated_accounted_flops',1e15,'Partial matrix + Muon FLOPs (10¹⁵)'),('training_seconds',3600,'Synchronized training time (hours)')]):
        ax=axes[i,j]
        for arm in ['native','raw_bytes']:
            for bundle in [1,2,3]:
                rr=[r for r in rows if r['depth']==depth and r['arm']==arm and r['bundle']==bundle]
                assert len(rr)==9
                ax.plot([r[field]/scale for r in rr],[r['probe_ce'] for r in rr],color=colors[arm],linestyle=styles[bundle],linewidth=1.6,label=f"{'Byte' if arm=='raw_bytes' else 'Native'} · seed {bundle}")
        ax.set_title(f'Depth {depth}');ax.set_xlabel(label);ax.grid(alpha=.2)
        if j==0:ax.set_ylabel('Fixed-probe CE (nats/token)')
axes[0,2].legend(fontsize=8,ncol=2)
fig.suptitle('Cross-size equal-token confirmation: fixed-probe learning curves\n2 FineWeb-Edu documents · intermediate states · 3 paired seeds',fontsize=14)
for suffix in ['png','pdf']:
    q=P/f'probe-curves.{suffix}';assert not q.exists();fig.savefig(q,dpi=160)
plt.close(fig)
with (P/'plot-receipt.json').open('x') as f:json.dump(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_sha256=hashlib.sha256((P/'probe-curves.json').read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__,outputs={x:hashlib.sha256((P/f'probe-curves.{x}').read_bytes()).hexdigest() for x in ['png','pdf']},test_opened=False),f,indent=2);f.write('\n')
