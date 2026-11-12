"""Verify narrow new bindings against preserved qualified source by AST. No numerical reruns."""
import ast
from pathlib import Path
p=Path('research/general_rsi_native_horizon_worker_v1.py');s=p.read_text()
s=s.replace('Native horizon binding of the qualified time worker; training and publication unchanged.','Pseudo-repetition binding of qualified durable time worker and new qualified Context.')
s=s.replace('general-rsi-native-horizon-001','general-rsi-pseudo-repetition-001').replace('research.general_rsi_native_horizon_worker_v1','research.general_rsi_pseudo_worker_v1')
s=s.replace('from research.general_rsi_equal_time_training_v1 import Context,write_checkpoint,read_checkpoint,clone_tree,sha','from research.general_rsi_pseudo_training_v1 import Context\n    from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint,clone_tree,sha')
s=s.replace('    from research.byte_input_model_v6 import build_codebook\n','')
a=s.index('    raw=[]\n');b=s.index('    latest=None;',a)
s=s[:a]+"    context=Context(config['recipe'],config['budget_ns'],plan['ema_half_life_ns'],config['roles'],config['pseudo_seed'],config['parent'],binding)\n"+s[b:]
s=s.replace("phase='native_horizon_confirmation'","phase='native_pseudo_repetition_confirmation'").replace('native_horizon_panel_launched=True','pseudo_repetition_panel_launched=True').replace('native horizon','pseudo-repetition')
assert ast.dump(ast.parse(s)) == ast.dump(ast.parse(Path('research/general_rsi_pseudo_worker_v1.py').read_text())), 'worker adaptation'
s=Path('research/general_rsi_native_horizon_review_v1.py').read_text().replace('native horizon','pseudo-repetition').replace('general-rsi-native-horizon-001','general-rsi-pseudo-repetition-001').replace('research.general_rsi_native_horizon_worker_v1','research.general_rsi_pseudo_worker_v1')
s=s.replace("    if slot%2:\n", "    natural=[r for r in canonical if r['objective']=='ce']\n    for r in canonical:\n        objective='pseudo' if expected['recipe']=='native_pseudo_repetition' and r['step']%2==0 else 'ce'\n        assert r['objective']==objective\n        assert 0<r['valid_loss_targets']<=1024\n        if objective=='ce':assert r['valid_loss_targets']==1024\n    if slot%2:\n")
s=s.replace("n=min(len(previous),len(canonical));assert previous[:n]==[r['batch_sha256'] for r in canonical[:n]]", "n=min(len(previous),len(natural));assert previous[:n]==[r['batch_sha256'] for r in natural[:n]]")
s=s.replace("batch_prefix=[r['batch_sha256'] for r in canonical],test_opened=False", "batch_prefix=[r['batch_sha256'] for r in natural],\n        objective_counts={k:sum(r['objective']==k for r in canonical) for k in ['ce','pseudo']},\n        valid_loss_targets={k:sum(r['valid_loss_targets'] for r in canonical if r['objective']==k) for k in ['ce','pseudo']},\n        saturated_targets=sum(r['saturated_targets'] for r in canonical),test_opened=False")
assert ast.dump(ast.parse(s)) == ast.dump(ast.parse(Path('research/general_rsi_pseudo_review_v1.py').read_text())), 'review adaptation'

from pathlib import Path
s=Path('research/general_rsi_fp32_quality_analysis_v1.py').read_text()
s=s.replace('FP32 quality','pseudo-repetition quality').replace('general-rsi-composer-fp32-quality-001','general-rsi-pseudo-repetition-001')
s=s.replace("BINDING = 'bcf4b43bf68e5137f5a04fe36c24a7baa99a7a2c552c474ba2225e4507e82f79'",'BINDING = None')
s=s.replace('PAIRS = ((0, 1), (3, 2), (4, 5))','PAIRS = ((0, 1), (2, 3), (4, 5))')
s=s.replace("def analyze():\n", "def analyze():\n    global BINDING\n    BINDING = read(PANEL/'admission.json')['plan_sha256']\n")
s=s.replace('byte_composer_fp32','native_pseudo_repetition').replace('existing_byte','native_ce').replace('fp32_minus_existing','pseudo_minus_ce').replace('FP32-minus-existing','pseudo-minus-CE').replace("'h200'","'time_ema'").replace('h200_delta','time_ema_delta').replace('existing_slot','ce_slot').replace('fp32_slot','pseudo_slot')
a=s.index("        assert a['arm']");b=s.index('        pairs.append(',a)
s=s[:a]+"        assert a['recipe'] == 'native_ce' and b['recipe'] == 'native_pseudo_repetition'\n        assert a['parent'] == b['parent'] and a['pseudo_seed'] == b['pseudo_seed']\n        assert a['budget_ns'] == b['budget_ns'] == 1200*10**9\n"+s[b:]
s=s.replace("('native_ce',[0,3,4]), ('native_pseudo_repetition',[1,2,5])", "('native_ce',[0,2,4]), ('native_pseudo_repetition',[1,3,5])")
s=s.replace('Composer Muon FP32 versus existing BF16 arithmetic at10240matched updates and fixed LR.0025 on this model/corpus. RAW secondary. No native superiority, equal-time advantage, equivalence or transfer claim.','Paired native continuation recipes from three selected parents at1200s each. Ordinary CE versus alternating CE/pseudo repetition; natural exposure differs. RAW secondary. No fresh-pretraining, semantic-generation or transfer claim.')
s=s.replace('research.general_rsi_fp32_quality_worker_v1','research.general_rsi_pseudo_worker_v1')
assert ast.dump(ast.parse(s)) == ast.dump(ast.parse(Path('research/general_rsi_pseudo_analysis_v1.py').read_text())), 'analysis adaptation'

print('Worker/review/decision bindings match the enumerated adaptations; original lifecycle and symmetric decision arithmetic preserved.')
