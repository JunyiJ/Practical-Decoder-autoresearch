"""Small deterministic repetition/length scorer for the prospective protocol.

Input is generated token IDs only, excluding the prompt. Semantic rubric
annotations are separate; these statistics cannot measure relevance/coherence.
No torch dependency and no model/data access.
"""

import argparse
import json
from pathlib import Path


def score(tokens, *, eos_token_id=50256, max_new_tokens=64):
    if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('expected a list of nonnegative integer token IDs')
    if len(tokens) > max_new_tokens:
        raise ValueError('generation exceeds the fixed cap')
    stopped_eos = bool(tokens and tokens[-1] == eos_token_id)
    if eos_token_id in tokens[:-1]:
        raise ValueError('tokens follow EOS; preserve and diagnose invalid output')
    body = tokens[:-1] if stopped_eos else tokens
    trigrams = [tuple(body[i:i + 3]) for i in range(max(0, len(body) - 2))]
    loop_lengths = [n for n in range(2, len(body) // 3 + 1)
                    if body[-n:] == body[-2*n:-n] == body[-3*n:-2*n]]
    return {
        'generated_tokens_including_eos': len(tokens),
        'content_tokens': len(body),
        'trigram_opportunities': len(trigrams),
        'repeated_trigram_fraction': ((len(trigrams) - len(set(trigrams))) / len(trigrams)
                                     if trigrams else None),
        'terminal_loop': bool(loop_lengths),
        'terminal_loop_phrase_lengths': loop_lengths,
        'stopped_eos': stopped_eos,
        'hit_token_cap': len(tokens) == max_new_tokens and not stopped_eos,
        'empty_content': not body,
        'unexpected_early_stop': not stopped_eos and len(tokens) < max_new_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL rows: id, generated_token_ids')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate generation ID')
    scored = [dict(id=row['id'], **score(row['generated_token_ids'])) for row in rows]
    with args.output.open('x') as handle:
        for row in scored:
            handle.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
