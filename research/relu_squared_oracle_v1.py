"""New exact-binary literals; independent scalar FFN forward/analytic backward.

No Torch, autograd, RNG, component imports, model, or checkpoint access.
Matrices are output-major. L=sum(output*upstream).
"""
import math

CASES = ('mixed_d8', 'all_zero_x', 'strict_negative_up', 'strict_positive_up',
         'relu_kink_triplet', 'zero_down', 'zero_up', 'signed_large',
         'boundary_n16_d6', 'small_d2', 'minimum_n1_d4')
KINDS = ('native_swiglu', 'ungated_silu', 'ungated_relu_squared')


def fixture(name, kind):
    if name not in CASES or kind not in KINDS:
        raise ValueError('New declared literal fixture only')
    d = 2 if name == 'small_d2' else 6 if name == 'boundary_n16_d6' else 4 if name == 'minimum_n1_d4' else 8
    n = 16 if name == 'boundary_n16_d6' else 1 if name == 'minimum_n1_d4' else 3
    h = d if kind == 'native_swiglu' else 3*d//2
    x = [[(((11*i+7*j) % 29)-14)/16 for j in range(d)] for i in range(n)]
    up = [[(((5*k+3*j) % 17)-8)/32 for j in range(d)] for k in range(h)]
    down = [[(((7*j+9*k) % 19)-9)/32 for k in range(h)] for j in range(d)]
    gate = [[(((13*k+5*j) % 23)-11)/32 for j in range(d)] for k in range(h)]
    upstream = [[(((3*i+5*j) % 13)-6)/16 for j in range(d)] for i in range(n)]
    if name == 'all_zero_x':
        x = [[0.]*d for _ in range(n)]
    elif name in ('strict_negative_up', 'strict_positive_up'):
        x = [[(i+j+1)/16 for j in range(d)] for i in range(n)]
        sign = -1 if name == 'strict_negative_up' else 1
        up = [[sign*(1+(k+j) % 5)/32 for j in range(d)] for k in range(h)]
    elif name == 'relu_kink_triplet':
        x = [[(0., 1/4096, -1/4096)[(i+j) % 3] for j in range(d)] for i in range(n)]
        up = [[float(j == k % d) for j in range(d)] for k in range(h)]
    elif name == 'zero_down':
        down = [[0.]*h for _ in range(d)]
    elif name == 'zero_up':
        up = [[0.]*d for _ in range(h)]
    elif name == 'signed_large':
        x = [[(-1 if (i+j) % 2 else 1)*(j+1) for j in range(d)] for i in range(n)]
        gate = [[float(j == k % d)*2 for j in range(d)] for k in range(h)]
    return dict(x=x, up=up, down=down, gate=gate if kind == 'native_swiglu' else None,
                upstream=upstream)


def silu_pair(t):
    s = 1/(1+math.exp(-t)) if t >= 0 else math.exp(t)/(1+math.exp(t))
    return t*s, s*(1+t*(1-s))


def oracle(case, kind):
    x, up, down, gate, upstream = [case[k] for k in ('x', 'up', 'down', 'gate', 'upstream')]
    n, d, h = len(x), len(x[0]), len(up)
    u = [[math.fsum(a*b for a, b in zip(row, w)) for w in up] for row in x]
    g = [[math.fsum(a*b for a, b in zip(row, w)) for w in gate] for row in x] if gate else None
    z, du, dg = [], [], []
    for i in range(n):
        zi, dui, dgi = [], [], []
        for k in range(h):
            dz = math.fsum(upstream[i][j]*down[j][k] for j in range(d))
            if kind == 'native_swiglu':
                v, derivative = silu_pair(g[i][k])
                zi.append(v*u[i][k]); dui.append(dz*v); dgi.append(dz*u[i][k]*derivative)
            elif kind == 'ungated_silu':
                v, derivative = silu_pair(u[i][k])
                zi.append(v); dui.append(dz*derivative)
            elif kind == 'ungated_relu_squared':
                v = max(u[i][k], 0.)
                zi.append(v*v); dui.append(dz*2*v)
            else:
                raise ValueError('Unknown scalar equation')
        z.append(zi); du.append(dui); dg.append(dgi)
    output = [[math.fsum(w*v for w, v in zip(row, zi)) for row in down] for zi in z]
    gx = [[math.fsum(du[i][k]*up[k][j] + (dg[i][k]*gate[k][j] if gate else 0.)
                     for k in range(h)) for j in range(d)] for i in range(n)]
    gu = [[math.fsum(du[i][k]*x[i][j] for i in range(n)) for j in range(d)] for k in range(h)]
    gd = [[math.fsum(upstream[i][j]*z[i][k] for i in range(n)) for k in range(h)] for j in range(d)]
    result = dict(output=output, gx=gx, gup=gu, gdown=gd)
    if gate:
        result['ggate'] = [[math.fsum(dg[i][k]*x[i][j] for i in range(n)) for j in range(d)] for k in range(h)]
    return result
