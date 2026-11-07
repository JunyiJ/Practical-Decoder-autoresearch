"""New literal fixtures and independent scalar LayerNorm forward/backward."""
import math

CASES = ('wide_signed', 'flat_zero', 'flat_offset', 'narrow_low', 'narrow_high',
         'narrow_gain8_low', 'narrow_gain8_high', 'permuted_narrow_high',
         'boundary_rows16', 'alternating_sign', 'one_spike')


def fixture(name):
    if name not in CASES:
        raise ValueError('Declared new literal fixture required')
    n = 16 if name == 'boundary_rows16' else 5
    x = [[(((19*j+13*i) % 67)-33)/16 for j in range(256)] for i in range(n)]
    gamma = [((j % 13)-6)/8 for j in range(256)]
    beta = [((j % 5)-2)/32 for j in range(256)]
    upstream = [[(((7*j+3*i) % 23)-11)/32 for j in range(256)] for i in range(n)]
    if name == 'flat_zero':
        x = [[0.]*256 for _ in range(n)]
        gamma = [1.]*256
    elif name == 'flat_offset':
        x = [[.625]*256 for _ in range(n)]
    elif 'narrow' in name:
        x = [[(((9*j+5*i) % 32)-15.5)/2048 for j in range(256)] for i in range(n)]
        if name.endswith('high'):
            x = [[v+2048 for v in row] for row in x]
        if 'gain8' in name:
            gamma = [v*8 for v in gamma]
    elif name == 'alternating_sign':
        x = [[(-1 if (j+i) % 2 else 1)*(1+j % 11)/8 for j in range(256)] for i in range(n)]
    elif name == 'one_spike':
        x = [[(1/256 if j == 17*i else 0.) for j in range(256)] for i in range(n)]
    if name == 'permuted_narrow_high':
        order = [(73*j) % 256 for j in range(256)]
        x = [[row[j] for j in order] for row in x]
        upstream = [[row[j] for j in order] for row in upstream]
        gamma, beta = [gamma[j] for j in order], [beta[j] for j in order]
    return dict(x=x, gamma=gamma, beta=beta, upstream=upstream)


def oracle(case):
    """L=sum(output*upstream), biased variance, epsilon fixed at1e-5."""
    output, gx = [], []
    ggamma = [[] for _ in range(256)]
    gbeta = [[] for _ in range(256)]
    for row, upstream in zip(case['x'], case['upstream']):
        mean = math.fsum(row)/256
        c = [v-mean for v in row]
        variance = math.fsum(v*v for v in c)/256
        inv = 1/math.sqrt(variance+1e-5)
        normalized = [v*inv for v in c]
        y = [v*g+b for v, g, b in zip(normalized, case['gamma'], case['beta'])]
        v = [g*t for g, t in zip(upstream, case['gamma'])]
        mean_v = math.fsum(v)/256
        mean_vc = math.fsum(g*t for g, t in zip(v, c))/256
        gx.append([inv*(g-mean_v-t*inv*inv*mean_vc) for g, t in zip(v, c)])
        for j, (g, t) in enumerate(zip(upstream, normalized)):
            ggamma[j].append(g*t)
            gbeta[j].append(g)
        output.append(y)
    return dict(output=output, gx=gx, ggamma=[math.fsum(v) for v in ggamma],
                gbeta=[math.fsum(v) for v in gbeta])
