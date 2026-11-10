"""Fixed first-output generation with per-token evidence and per-prompt RNG."""
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
    from research.byte_input_cache_v4 import Session
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
                    generator_source_sha256=sha(__file__),cache_source_sha256=sha(Path(__file__).with_name('byte_input_cache_v4.py'))))
                session=Session(model,1)
                inputs=torch.tensor([ids],dtype=torch.long,device=model.validate())
                for position in range(64):
                    with torch.no_grad(),FP32RMSNormInference():
                        logits=session.logits(inputs,session.position)[0,-1].cpu()
                    token=select_token(logits,decoder,generator)
                    generated.append(token)
                    append(journal,dict(event='token',position=position,token=token))
                    if token==enc.eot_token:break
                    inputs=torch.tensor([[token]],dtype=torch.long,device=model.validate())
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


def blind(rows):
    """Call after gathering all144 outputs; mapping saved separately by caller."""
    assert len(rows)==144
    shuffled=list(rows);random.Random(62110500998).shuffle(shuffled)
    sheets=[];mapping=[]
    for i,row in enumerate(shuffled):
        identity=f'g{i:03d}'
        sheets.append(dict(id=identity,prompt=row['prompt'],continuation=row['continuation'],
                           relevance=None,consistency=None,evidence_quote=None,reason=None,annotator_identity=None))
        mapping.append(dict(id=identity,original_id=row['id'],checkpoint_sha256=row['checkpoint_sha256'],decoder=row['decoder']))
    return sheets,mapping
