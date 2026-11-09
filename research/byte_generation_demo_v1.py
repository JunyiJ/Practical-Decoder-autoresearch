"""Bounded qualitative demo of the adopted checkpoint; no training or selection."""
import fcntl
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/byte-generation-demo-001'
RECIPE = ROOT / 'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json'
PROMPTS = [
    'KING RICHARD:\nMy friends, the kingdom is in danger, and we must',
    'JULIET:\nIf love be true, why must we part tonight?\nROMEO:\n',
    'First Citizen:\nWe have no bread, and our children are hungry.\nSecond Citizen:\n',
    'Question: Why does the sun rise in the morning?\nAnswer:',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(record):
    path = Path(record['path'])
    assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
    return json.loads(path.read_text())


def save(name, data):
    with (OUT / name).open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run():
    assert sha(RECIPE) == '1a5029445e36f2701fa3e4b9d5bcf6af29ce21fbb07c54b0366bb027aa1d3823'
    recipe = json.loads(RECIPE.read_text())
    loader = recipe['loader']
    implementation = bound(loader['inference_implementation_record'])
    for record in implementation['sources']:
        assert sha(record['path']) == record['sha256']
    for path, digest in bound(implementation['source_inventory'])['sources'].items():
        assert sha(ROOT / path) == digest, path
    for record in bound(recipe['source_inventory']['record'])['sources']:
        assert sha(record['path']) == record['sha256'], record['path']
    import torch
    import tiktoken
    from research.byte_input_corrected_inference_v1 import load
    torch.set_num_threads(4)
    backend = 'mps' if torch.backends.mps.is_available() else 'cpu'
    started = time.monotonic()
    consumer = load(loader['implementation_record'], loader['plan_record'],
                    loader['acceptance_record'], backend)
    enc = tiktoken.get_encoding('gpt2')
    result = dict(recipe_sha256=sha(RECIPE), checkpoint=recipe['representative']['h200'],
                  backend=backend, temperature=0.8, top_k=40, max_new_tokens=64,
                  seed=42, prompts=PROMPTS, samples=[], training_updates=0)
    generator = torch.Generator(device='cpu').manual_seed(42)
    for prompt in PROMPTS:
        tokens = enc.encode(prompt)
        assert len(tokens) + 64 <= 128
        session = consumer.session(1)
        inputs = torch.tensor([tokens], dtype=torch.long, device=consumer.device)
        generated = []
        for step in range(64):
            scores = session.logits(inputs, session.position)[0, -1].cpu()
            values, indices = torch.topk(scores / 0.8, 40)
            choice = torch.multinomial(torch.softmax(values, dim=-1), 1, generator=generator)
            token = int(indices[choice].item())
            generated.append(token)
            if token == enc.eot_token:
                break
            inputs = torch.tensor([[token]], dtype=torch.long, device=consumer.device)
        sample = dict(prompt=prompt, continuation=enc.decode(generated), token_ids=generated,
                      stop_reason='eos' if generated[-1] == enc.eot_token else 'length_limit')
        result['samples'].append(sample)
        save(f'sample-{len(result["samples"]):02d}.json', sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    consumer.check()
    for record in recipe['representative'].values():
        if isinstance(record, dict):
            assert sha(record['path']) == record['sha256']
    result['elapsed_seconds'] = time.monotonic() - started
    result['selected_weights_unchanged'] = True
    save('result.json', result)


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir()
        save('attempt.json', dict(prompts=PROMPTS, seed=42, temperature=0.8, top_k=40,
                                 max_new_tokens=64, source_sha256=sha(__file__)))
        try:
            with CpuPrerequisiteGuard(OUT / 'guard.json', ROOT, timeout_seconds=180,
                                      rss_limit_bytes=4*1024**3, reserve_bytes=10*1024**3):
                run()
        except BaseException:
            save('failure.json', dict(traceback=traceback.format_exc()))
            raise
