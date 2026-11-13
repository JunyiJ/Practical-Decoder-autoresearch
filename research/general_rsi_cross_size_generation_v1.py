"""Fixed uncached generation for depth models; same qualified decoder and journal."""
import json
from pathlib import Path
import random
import time

from research.general_rsi_time_journal_v1 import append
from research.general_rsi_training_v2 import sha
from research.general_rsi_generation_metrics_v1 import score


def select_token(logits, decoder, generator):
    import torch
    assert logits.ndim == 1 and bool(torch.isfinite(logits).all())
    if decoder=='greedy':return int(logits.argmax())
    assert decoder=='temperature_0.8_top_k_40'
    values,indices=torch.topk(logits/.8,40)
    return int(indices[torch.multinomial(torch.softmax(values,dim=-1),1,generator=generator)])


def generate(model, protocol, directory, checkpoint, binding):
    import torch
    import tiktoken
    from research.general_rsi_pseudo_update_diagnostic_v1 import check_stop
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    directory=Path(directory)
    # Exclusive per-model folder: retries require explicit preserved-prefix recovery.
    directory.mkdir()
    assert protocol['seed_reset']=='each prompt and decoder' and protocol['rerolls']==0
    assert protocol['seed']==42 and protocol['max_new_tokens']==64 and protocol['eos_token_id']==50256
    assert protocol['decoders']==['greedy','temperature_0.8_top_k_40']
    assert len(protocol['prompts'])==12 and len({p['id'] for p in protocol['prompts']})==12
    assert sha(checkpoint['path'])==checkpoint['sha256']
    enc=tiktoken.get_encoding('gpt2'); assert enc.eot_token==protocol['eos_token_id']
    prior=model.training;model.eval();start=time.perf_counter();rows=[]
    try:
        for decoder in protocol['decoders']:
            for prompt in protocol['prompts']:
                generated=[];ids=enc.encode(prompt['text']);assert 0<len(ids) and len(ids)+64<=128
                item_id=f"{prompt['id']}-{decoder}"
                journal=directory/f'{item_id}.jsonl'
                assert not journal.exists()
                generator=torch.Generator().manual_seed(protocol['seed'])
                append(journal,dict(event='start',id=item_id,prompt=prompt,decoder=decoder,seed=42,
                    checkpoint=checkpoint,binding=binding,torch_version=torch.__version__,
                    generator_source_sha256=sha(__file__),context=128,depth=model.spec.depth,inference='uncached_full_prefix'))
                inputs=torch.tensor([ids],dtype=torch.long,device=model.validate())
                for position in range(64):
                    check_stop()
                    with torch.no_grad(),FP32RMSNormInference():
                        logits=model.primary_logits(inputs).values[0,-1].cpu()
                    token=select_token(logits,decoder,generator)
                    generated.append(token)
                    append(journal,dict(event='token',position=position,token=token))
                    if token==enc.eot_token:break
                    inputs=torch.tensor([ids+generated],dtype=torch.long,device=model.validate())
                metrics=score(generated)
                assert not metrics['unexpected_early_stop']
                row=dict(id=item_id,prompt=prompt,decoder=decoder,seed=42,generated_token_ids=generated,
                         continuation=enc.decode(generated),stop_reason='eos' if metrics['stopped_eos'] else 'length_limit',
                         checkpoint_sha256=checkpoint['sha256'],metrics=metrics)
                append(journal,dict(event='complete',row=row));rows.append(row)
    finally:model.train(prior)
    assert len(rows)==24
    with (directory/'result.json').open('x') as f:
        json.dump(dict(rows=rows,seconds=time.perf_counter()-start,checkpoint=checkpoint,binding=binding),f,indent=2)
        f.write('\n')
    return rows

