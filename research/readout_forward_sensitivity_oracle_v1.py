"""Independent scalar binary64 oracle and new dyadic diagnostic literals."""
import math

CASES = ('regular_norm', 'regular_bypass', 'narrow_centered', 'narrow_shifted',
         'narrow_gain16_centered', 'narrow_gain16_shifted',
         'narrow_gain64_centered', 'narrow_gain64_shifted', 'constant_norm')


def fixture(name):
    if name not in CASES:
        raise ValueError('New declared literal fixture required')
    narrow = name.startswith('narrow')
    x = [[(((17*j + 29*i) % 257)-128)/128 for j in range(256)] for i in range(4)]
    if narrow:
        x = [[(((j+3*i) % 16)-7.5)/4096 for j in range(256)] for i in range(4)]
    if name.endswith('shifted'):
        x = [[v+1024 for v in row] for row in x]
    if name == 'constant_norm':
        x = [[.75]*256 for _ in range(4)]
    gain = 64 if 'gain64' in name else 16 if 'gain16' in name else 1
    gamma = [gain*(1+((j % 9)-4)/16) for j in range(256)]
    beta = [((j % 7)-3)/64 for j in range(256)]
    a = [[(.5 if i == j else 0.) + (((5*i+3*j) % 17)-8)/4096
          for j in range(256)] for i in range(256)]
    w = [[(((23*i+7*j) % 41)-20)/128 for j in range(256)] for i in range(31)]
    return dict(x=x, w=w, a=a, gamma=gamma, beta=beta,
                targets=[1, 9, 18, 29], normalize=name != 'regular_bypass')


def evaluate(case, arm):
    if arm not in ('native', 'linear_projection1', 'tanh_projection1'):
        raise ValueError('Declared diagnostic arm required')
    result = {k: [] for k in ('features', 'projected', 'transformed', 'logits', 'nll')}
    for row, target in zip(case['x'], case['targets']):
        if case['normalize']:
            mean = math.fsum(row)/256
            centered = [v-mean for v in row]
            variance = math.fsum(v*v for v in centered)/256
            denom = math.sqrt(variance+1e-5)
            z = [g*v/denom+b for g, v, b in zip(case['gamma'], centered, case['beta'])]
        else:
            z = list(row)
        u = z if arm == 'native' else [math.fsum(v*t for v, t in zip(r, z)) for r in case['a']]
        h = [math.tanh(v) for v in u] if arm == 'tanh_projection1' else u
        logits = [math.fsum(v*t for v, t in zip(r, h))/4 for r in case['w']]
        top = max(logits)
        loss = top+math.log(math.fsum(math.exp(v-top) for v in logits))-logits[target]
        for k, v in zip(result, (z, u, h, logits, loss)):
            result[k].append(v)
    return result
