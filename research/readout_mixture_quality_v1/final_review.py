"""Independent explicit draw/group/linear-quantile arithmetic for both contrasts."""
import math
import numpy as np
from research.body_allocation_model_v1.contract import require
from .statistics import report_contract

def verify(panel,seeds,bootstrap,primary):
    require(type(panel) is list and len(panel)==3 and [p['seed'] for p in panel]==seeds and len(set(seeds))==3,'All original fresh triples required')
    for p in panel:
        require(set(p)=={'seed','native','context_mix2','probability_mix2'},'Complete three-arm fresh inventory required')
        for arm in ('native','context_mix2','probability_mix2'):report_contract(p[arm],arm)
    values={}
    for control in ('native','context_mix2'):
        ds=[(p['probability_mix2']['total_loss']-p[control]['total_loss'])/34278 for p in panel];mean=math.fsum(ds)/3
        expected=primary['contrasts'][control]
        require(max(abs(a-b) for a,b in zip(ds,expected['candidate_minus_control_ce']))<1e-12 and abs(mean-expected['mean_ce_delta'])<1e-12,'Independent complete paired mean differs')
        values[control]=dict(deltas=ds,mean=mean,upper={})
    for width in (2048,4096,8192):
        counts=[min(width,34278-i) for i in range(0,34278,width)]
        # Group directly from128-target windows; primary groups2048 blocks.
        grouped=[{a:[math.fsum(p[a]['window_loss_sums'][i:i+width//128]) for i in range(0,268,width//128)] for a in ('native','context_mix2','probability_mix2')} for p in panel]
        rng=np.random.Generator(np.random.PCG64(bootstrap));samples={a:[] for a in ('native','context_mix2')}
        for _ in range(2000):
            draw=rng.integers(0,len(counts),len(counts));denominator=sum(counts[int(i)] for i in draw)
            for a in ('native','context_mix2'):
                samples[a].append(math.fsum(math.fsum(g['probability_mix2'][int(i)]-g[a][int(i)] for i in draw)/denominator for g in grouped)/3)
        for a in ('native','context_mix2'):
            ordered=sorted(samples[a]);position=(1-.05/6)*(len(ordered)-1);lo=math.floor(position);hi=math.ceil(position);fraction=position-lo
            upper=ordered[lo]*(1-fraction)+ordered[hi]*fraction;values[a]['upper'][str(width)]=upper
            row=primary['contrasts'][a]['intervals'][str(width)]
            require(row['token_counts']==counts and row['draws']==2000 and row['draw_seed']==bootstrap and row['quantile']==1-.05/6 and row['pcg64_reinitialized_each_width'] is True and row['same_indices_across_seeds_and_contrasts'] is True,'Original six-comparison interval contract differs')
            require(abs(upper-row['upper'])<1e-12,'Independent grouped bootstrap/linear quantile differs')
    gates=dict(both_mean_ce_below_minus_point02=all(v['mean']<-.02 for v in values.values()),every_paired_seed_delta_negative=all(d<0 for v in values.values() for d in v['deltas']),all_six_simultaneous_conditional_text_upper_bounds_negative=all(u<0 for v in values.values() for u in v['upper'].values()))
    require(primary['practical_threshold']==-.02 and gates==primary['gates'] and all(gates.values())==primary['quality_gates_pass'],'Independent strict decisions differ')
    return dict(status='independent_complete_panel_statistics_agreed',contrasts=values,gates=gates,quality_gates_pass=all(gates.values()),atol=1e-12,adopted=False,scope='Conditional on trained seeds/reused validation text;not holdout or population-seed confidence.')
