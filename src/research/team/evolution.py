"""Bounded configuration evolution; selection never reads transfer outcomes."""
from __future__ import annotations

import copy
import math
from statistics import mean

from ..protocol import digest


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"population_size", "max_mutations", "explore_every"}:
        raise ValueError("evolution requires population_size, max_mutations and explore_every")
    for key, upper in (("population_size", 16), ("max_mutations", 2), ("explore_every", 16)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"evolution.{key} must be an integer from 1 to {upper}")
    if settings['explore_every'] < 3:
        raise ValueError("evolution.explore_every must leave room for mutation and crossover")


def flattened(value, prefix=""):
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flattened(item, path))
        else:
            result[path] = item
    return result


def genotype(base, candidate):
    a, b = flattened(base), flattened(candidate)
    return {path: value for path, value in b.items() if path not in a or value != a[path]}


def population(state):
    settings = state['config']['evolution']
    local = f"width-{state['plan']['widths'][0]}"
    members = []
    for item in state['iterations']:
        report = item.get('report', {})
        if (item['stage'] != 'completed' or report.get('local_outcome') != 'supported'
                or not report.get('evidence_integrity', False)
                or item.get('idea', {}).get('change_kind') != 'config'):
            continue
        row = next((r for r in report.get('performance', [])
                    if r['target'] == local and r['metric'] == 'validation_ce_delta'), None)
        if not row or not row['complete'] or not math.isfinite(row['candidate_mean']):
            continue
        members.append({'iteration': item['id'], 'result_id': item['result_id'],
                        'genotype': genotype(state['baseline_config'], item['candidate_config']),
                        'local_ce': row['candidate_mean'], 'report_digest': digest(report)})
    members.sort(key=lambda x: (x['local_ce'], x['iteration']))
    unique = {}
    for member in members:
        unique.setdefault(digest(member['genotype']), member)
    return list(unique.values())[:settings['population_size']]


def plan_generation(state):
    settings = state['config']['evolution']; members = population(state)
    generation = len(state['iterations']) + 1
    slot = (generation - 1) % settings['explore_every']
    plan = {'generation': generation, 'population': members, 'parents': [],
            'inherited_genotype': {}, 'mode': 'explore',
            'max_mutations': settings['max_mutations'],
            'selection': 'small_width_validation_only',
            'parent_comparison': 'historical paired observations; not fresh randomized trials'}
    if not members or slot == settings['explore_every'] - 1:
        return plan
    parent = members[(generation - 2) % len(members)]
    plan.update(mode='mutate', parents=[copy.deepcopy(parent)], inherited_genotype=copy.deepcopy(parent['genotype']))
    if slot == settings['explore_every'] - 2:
        for donor in members:
            additions = {k: v for k, v in donor['genotype'].items() if k not in parent['genotype']}
            if additions:
                # Conflicting loci stay with the selected primary parent.
                additions = dict(sorted(additions.items())[:settings['max_mutations']])
                combined = {**parent['genotype'], **additions}
                if any(combined == member['genotype'] for member in members):
                    continue  # A copy of an existing parent is not a new crossover.
                plan['inherited_genotype'].update(additions)
                plan['parents'].append(copy.deepcopy(donor))
                plan.update(mode='crossover', max_mutations=0, crossed_paths=sorted(additions))
                break
    return plan


def validate_child(state, iteration, changes):
    plan = iteration.get('evolution')
    if plan is None:
        return
    if iteration['idea']['change_kind'] != 'config' or any(c['path'].startswith('src/') for c in changes):
        raise ValueError("Configuration evolution cannot inherit source edits; use config_overrides only")
    child = genotype(state['baseline_config'], iteration['candidate_config'])
    parent = plan['inherited_genotype']; sentinel = object()
    paths = sorted(k for k in child.keys() | parent.keys() if child.get(k, sentinel) != parent.get(k, sentinel))
    if len(paths) > plan['max_mutations']:
        raise ValueError(f"Evolution mutation bound exceeded: {paths}")
    # A crossover alone is the intervention. Mutation/exploration must change something.
    if plan['mode'] != 'crossover' and not paths:
        raise ValueError("Evolution requires a mutation, not an unchanged parent")
    iteration['evolution_child'] = {'genotype': child, 'mutated_paths': paths}


def compare_parents(state, iteration, report):
    """Describe saved parent observations; never change the frozen local choice."""
    plan = iteration.get('evolution')
    if plan is None:
        return None
    comparisons = []
    for parent in plan['parents']:
        old = next(i['report'] for i in state['iterations'] if i['id'] == parent['iteration'])
        if digest(old) != parent['report_digest']:
            raise ValueError("Frozen evolutionary parent evidence changed")
        for width in state['plan']['widths']:
            target = {'name': f'width-{width}'}
            pairs = []
            for seed in state['plan']['seeds']:
                def score(r):
                    rows = [x for x in r['records'] if x['arm'] == 'candidate' and x['target'] == target['name']
                            and x['seed'] == seed and x['status'] == 'completed']
                    return rows[0]['metric'] if len(rows) == 1 and math.isfinite(rows[0]['metric']) else None
                a, b = score(old), score(report)
                if a is not None and b is not None:
                    pairs.append({'seed': seed, 'parent': a, 'child': b, 'delta': b-a})
            complete = (len(pairs) == len(state['plan']['seeds']) and report.get('evidence_integrity', False)
                        and old.get('evidence_integrity', False))
            comparisons.append({'parent_result_id': parent['result_id'], 'target': target['name'],
                                'pairs': pairs, 'complete': complete,
                                'mean_child_minus_parent': mean(p['delta'] for p in pairs) if complete else None})
    return {'plan': plan, 'child': iteration['evolution_child'], 'comparisons': comparisons,
            'scope': 'Adaptive historical comparison within one frozen campaign; parents are not retrained controls. '
                     'Joint configuration success does not establish individual component benefit or synergy.'}
