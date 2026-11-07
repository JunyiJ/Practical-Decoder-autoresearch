"""Own future-only nearest-occurrence targets; SOURCE ONLY, runtime unqualified.

This is an explicit local contract inspired by TOP, not an upstream reproduction.
No model, tokenizer, sampler, loss kernel, tensor or checkpoint dependencies.
"""
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(sequence, *, context, window, vocabulary):
    """Return immutable (token_id, first_future_offset) pairs for each input.

    Inputs are sequence[:context]. Row t sees labels only at t+1 through
    t+window, inclusive. Repeated future types appear once, at their first
    offset. Complete validation precedes construction; caller input is untouched.
    """
    _require(type(context) is int and 1 <= context <= 128, 'Context must be 1..128')
    _require(type(window) is int and 1 <= window <= 128, 'Window must be 1..128')
    _require(type(vocabulary) is int and 2 <= vocabulary <= 50257, 'Vocabulary must be 2..50257')
    _require(type(sequence) in (list, tuple) and len(sequence) == context + window,
             'An exact split-local context+window sequence is required')
    _require(all(type(token) is int and 0 <= token < vocabulary for token in sequence),
             'Every ID must be an ordinary in-vocabulary integer; no padding')
    result = []
    for t in range(context):
        seen, row = set(), []
        for offset in range(1, window + 1):
            token = sequence[t + offset]
            if token not in seen:
                seen.add(token)
                row.append((token, offset))
        result.append(tuple(row))
    return tuple(result)


def probabilities(row):
    """Sparse target probabilities proportional to exp(1-first_offset).

    Removing the common window score avoids a large positive score offset.
    Python floats/math.fsum only; this is not native CE or a model backend.
    """
    _require(type(row) is tuple and 1 <= len(row) <= 128, 'Nonempty immutable row required')
    seen, previous = set(), 0
    for pair in row:
        _require(type(pair) is tuple and len(pair) == 2, 'Exact ID/offset pair required')
        token, offset = pair
        _require(type(token) is int and 0 <= token < 50257 and token not in seen,
                 'Distinct ordinary vocabulary IDs required')
        _require(type(offset) is int and previous < offset <= 128,
                 'Strictly increasing positive offsets required')
        seen.add(token)
        previous = offset
    _require(row[0][1] == 1, 'The immediate future token must be present')
    masses = tuple(math.exp(1 - offset) for _, offset in row)
    total = math.fsum(masses)
    return tuple((token, mass / total) for (token, _), mass in zip(row, masses))
