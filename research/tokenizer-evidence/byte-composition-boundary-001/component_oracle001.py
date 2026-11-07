"""Independent represented-literal scalar forward/analytic gradient; no Torch."""
from array import array
import math
import struct

CASES = (
    ('short_nul', ((257, 0, 258),), 3, False),
    ('byte_extremes', ((257, 0, 255, 32, 65, 258),), 6, False),
    ('mixed_special_padding', ((257, 32, 65, 32, 66, 258), (257, 259, 258)), 10, False),
    ('repeated_valid_max', ((257, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 258),), 12, False),
    ('zero_relu_and_max_ties', ((257, 0, 0, 0, 258), (257, 259, 258)), 8, True),
    ('maximum_128_bytes', ((257, 0, 73, 146, 219, 36, 109, 182, 255, 72, 145, 218, 35, 108, 181, 254, 71, 144, 217, 34, 107, 180, 253, 70, 143, 216, 33, 106, 179, 252, 69, 142, 215, 32, 105, 178, 251, 68, 141, 214, 31, 104, 177, 250, 67, 140, 213, 30, 103, 176, 249, 66, 139, 212, 29, 102, 175, 248, 65, 138, 211, 28, 101, 174, 247, 64, 137, 210, 27, 100, 173, 246, 63, 136, 209, 26, 99, 172, 245, 62, 135, 208, 25, 98, 171, 244, 61, 134, 207, 24, 97, 170, 243, 60, 133, 206, 23, 96, 169, 242, 59, 132, 205, 22, 95, 168, 241, 58, 131, 204, 21, 94, 167, 240, 57, 130, 203, 20, 93, 166, 239, 56, 129, 202, 19, 92, 165, 238, 55, 258),), 130, False),
)
SHAPES = {'byte_weight': (259, 15), 'highway_weight': (525, 525),
          'highway_bias': (525,), 'transform_weight': (525, 525),
          'transform_bias': (525,), 'projection_weight': (256, 525)}
SHAPES.update({f'filters.{i}': (25*(i+1), 15*(i+1)) for i in range(6)})
SHAPES.update({f'filter_biases.{i}': (25*(i+1),) for i in range(6)})


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def literal_parameters(zero=False):
    result = {}
    for name, shape in SHAPES.items():
        values = array('d')
        for i in range(math.prod(shape)):
            r, c = (divmod(i, shape[1]) if len(shape) == 2 else (i, 0))
            if name == 'byte_weight':
                value = ((r % 251)+1)*2**-12 + (c+1)*2**-15
            elif name.startswith('filters.'):
                value = ((r % 13)+1)*2**-14 + ((c % 17)+1)*2**-17
            elif name.startswith('filter_biases.'):
                value = ((r % 7)+1)*2**-10
            elif name == 'highway_weight':
                value = ((r % 5)-2)*((c % 7)-3)*2**-11
            elif name == 'highway_bias':
                value = .03125 if r % 2 else -.03125
            elif name == 'transform_weight':
                value = ((r % 3)-1)*((c % 5)-2)*2**-12
            elif name == 'transform_bias':
                value = -2. + ((r % 3)-1)*2**-6
            else:
                value = ((r % 7)-3)*((c % 11)-5)*2**-12
            if zero and name not in ('byte_weight', 'projection_weight', 'transform_bias'):
                value = 0.
            values.append(f32(value))
        result[name] = values
    return result


def reference(case, parameters):
    """d(sum(output[n,j]*represented_coefficient[j]/N))/d(each parameter).

    Stable scalar sigmoid, math.fsum dots. ReLU'(0)=0. Valid amax ties share
    derivative equally; absent-width features and their derivatives are zero.
    """
    _, sequences, _, _ = case
    p = parameters
    gradients = {name: array('d', [0.])*len(value) for name, value in p.items()}
    outputs = array('d')
    N = len(sequences)
    coefficients = [f32((j % 5-2)*2**-8/N) for j in range(256)]
    tie_channels = 0
    for sequence in sequences:
        rows = [code-(code > 256) for code in sequence]
        emb = [p['byte_weight'][row*15:(row+1)*15] for row in rows]
        features, tapes = [], []
        for index in range(6):
            width, channels = index+1, 25*(index+1)
            W, bias = p[f'filters.{index}'], p[f'filter_biases.{index}']
            count = max(0, len(sequence)-width+1)
            maps = []
            for start in range(count):
                maps.append([math.tanh(math.fsum(W[c*15*width+e*width+d]*emb[start+d][e]
                             for e in range(15) for d in range(width))+bias[c]) for c in range(channels)])
            maxima = [max(row[c] for row in maps) if count else 0. for c in range(channels)]
            ties = [[start for start in range(count) if maps[start][c] == maxima[c]]
                    for c in range(channels)]
            tie_channels += sum(len(indices) > 1 for indices in ties)
            features.extend(maxima)
            tapes.append((width, channels, maps, ties))
        y = features
        pre = [math.fsum(p['highway_weight'][r*525+c]*y[c] for c in range(525))+
               p['highway_bias'][r] for r in range(525)]
        hidden = [max(0., value) for value in pre]
        logits = [math.fsum(p['transform_weight'][r*525+c]*y[c] for c in range(525))+
                  p['transform_bias'][r] for r in range(525)]
        gates = [(1./(1.+math.exp(-x)) if x >= 0 else math.exp(x)/(1.+math.exp(x))) for x in logits]
        z = [gates[c]*hidden[c]+(1.-gates[c])*y[c] for c in range(525)]
        outputs.extend(math.fsum(p['projection_weight'][j*525+c]*z[c] for c in range(525))
                       for j in range(256))
        dz = [math.fsum(p['projection_weight'][j*525+c]*coefficients[j] for j in range(256))
              for c in range(525)]
        dh = [dz[r]*gates[r]*(pre[r] > 0) for r in range(525)]
        dt = [dz[r]*(hidden[r]-y[r])*gates[r]*(1.-gates[r]) for r in range(525)]
        dy = [dz[c]*(1.-gates[c])+math.fsum(p['highway_weight'][r*525+c]*dh[r]+
              p['transform_weight'][r*525+c]*dt[r] for r in range(525)) for c in range(525)]
        for j in range(256):
            for c in range(525):
                gradients['projection_weight'][j*525+c] += coefficients[j]*z[c]
        for r in range(525):
            gradients['highway_bias'][r] += dh[r]
            gradients['transform_bias'][r] += dt[r]
            for c in range(525):
                gradients['highway_weight'][r*525+c] += dh[r]*y[c]
                gradients['transform_weight'][r*525+c] += dt[r]*y[c]
        de = [[0.]*15 for _ in sequence]
        offset = 0
        for index, (width, channels, maps, ties) in enumerate(tapes):
            W = p[f'filters.{index}']
            for c in range(channels):
                if not ties[c]:
                    continue
                for start in ties[c]:
                    df = dy[offset+c]*(1.-maps[start][c]**2)/len(ties[c])
                    gradients[f'filter_biases.{index}'][c] += df
                    for e in range(15):
                        for d in range(width):
                            q = c*15*width+e*width+d
                            gradients[f'filters.{index}'][q] += df*emb[start+d][e]
                            de[start+d][e] += df*W[q]
            offset += channels
        for pos, row in enumerate(rows):
            for e in range(15):
                gradients['byte_weight'][row*15+e] += de[pos][e]
    return {'output': outputs, 'gradients': gradients, 'tie_channels': tie_channels,
            'coefficients': coefficients, 'relu_zero_derivative': 0., 'amax_ties': 'equal_share_valid_only'}
