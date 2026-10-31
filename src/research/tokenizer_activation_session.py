"""Bounded immutable activation controller ledger; no subprocess or deletion API.

The future executable entry must independently validate its frozen plan, runtime,
source, inputs, child audits and current full-panel capacity. This module verifies
receipt provenance and ordering, not the scientific truth of an audit verdict.
It deliberately cannot turn a self-declared receipt into launch authorization.
"""
import copy
import json
import re
import stat

from . import tokenizer_activation_rules as rules
from .tokenizer_activation_config import config
from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _path, _sha

KIND = 'activation_session_v1'
MAX_EVENTS = 40
AUDITS = ('training', 'evaluation', 'tensors', 'resources')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(root, name):
    path = _path(root, name)
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'Regular unaliased artifact required')
    value = dict(path=name, size=st.st_size, sha256=file_digest(path))
    after = path.stat()
    require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Artifact changed during hashing')
    return value


def ref_schema(root, ref):
    require(type(ref) is dict and set(ref) == {'path', 'size', 'sha256'}, 'Exact artifact reference required')
    require(type(ref['size']) is int and ref['size'] >= 0, 'Exact artifact size required')
    _path(root, ref['path']); _sha(ref['sha256'])
    return ref


class Session:
    def __init__(self, writer, plan_ref):
        require(isinstance(writer, MetadataWriter), 'Bounded writer and two worker locks required')
        self.writer, self.root = writer, writer.root
        self.plan = copy.deepcopy(ref_schema(self.root, plan_ref))
        require(self.plan['path'] == 'evidence/meta/activation-plan.json', 'Dedicated activation plan required')

    def verify(self, ref):
        ref_schema(self.root, ref)
        require(reference(self.root, ref['path']) == ref, 'Artifact hash/size changed: ' + ref['path'])
        return self.root / ref['path']

    def document(self, ref):
        require(ref['size'] <= 1024**2, 'Receipt exceeds read bound')
        return json.loads(self.verify(ref).read_text())

    def _events(self):
        snapshot(self.root)  # Reject symlinks, aliases, undeclared paths and metadata overflow.
        directory = self.root/'evidence/meta/session'
        names = sorted(directory.iterdir()) if directory.exists() else []
        require(len(names) <= MAX_EVENTS, 'Finite session event cap exceeded')
        events, previous = [], None
        for i, path in enumerate(names):
            require(path.name == f'{i:04d}.json', 'Partial or noncontiguous ledger; no automatic retry')
            ref = reference(self.root, str(path.relative_to(self.root)))
            event = self.document(ref)
            require(set(event) == {'kind', 'index', 'previous', 'plan', 'action', 'payload'}
                    and event['kind'] == KIND and type(event['index']) is int and event['index'] == i
                    and event['previous'] == previous and event['plan'] == self.plan,
                    'Session event chain or plan differs')
            events.append((event, ref)); previous = ref
        return events

    @staticmethod
    def _action(state):
        return rules.next_action(state['pilots'], state['records'], state['selection'],
            pilots_released=state['pilot_boundary'], losers_released=state['screen_boundary'])

    def _eligible(self, state, boundary):
        if boundary == 'pilot':
            indices = [0, 1]
        else:
            keep = rules.screening_retention(state['records'], state['selection'])
            indices = [i+2 for i in keep['losing_indices']]
        return {ref['path']: ref for i in indices for ref in state['pairs'][i]}

    def inspect(self):
        """Reconstruct from immutable events. Never trusts a mutable state cache."""
        return self._replay(self._events())

    def _replay(self, events, *, verify_pins=True):
        self.verify(self.plan)
        state = dict(pilots=[], records=[], selection=None, selection_ref=None,
                     pilot_boundary=False, screen_boundary=False, pending=None,
                     pairs=[], reviews=[], pins={self.plan['path']: self.plan}, deleted={}, failed=False)
        def pin(ref):
            ref_schema(self.root, ref)
            old = state['pins'].get(ref['path'])
            require(old is None or old == ref, 'Conflicting artifact identities')
            state['pins'][ref['path']] = ref
        for event, event_ref in events:
            require(not state['failed'], 'No continuation after a failed attempt')
            action, payload = event['action'], event['payload']
            if action == 'reserve':
                expected = self._action(state)
                require(state['pending'] is None and expected['kind'] in ('pilot', 'quality'), 'Unexpected trial reservation')
                request = self.document(payload)
                stage = 'pilots' if expected['kind'] == 'pilot' else 'quality'
                slot = len(state['pilots']) if stage == 'pilots' else len(state['records'])
                stem = f'evidence/{stage}/{slot:02d}'
                exact = dict(kind=KIND, plan=self.plan, spec=expected['spec'],
                             config=config(self.root/stem, expected['spec']))
                require(payload['path'] == stem+'/request.json' and digest(request) == digest(exact),
                        'Request differs from finite next slot/config')
                state['pending'] = dict(ref=payload, request=request, stem=stem)
                pin(payload)
            elif action == 'accept':
                pending = state['pending']
                require(pending is not None, 'Acceptance requires one reserved attempt')
                stem = pending['stem']; review = self.document(payload)
                require(not any(p.name.endswith('.tmp') or 'failure' in p.name
                                for p in (self.root/stem).rglob('*')), 'Failed or partial producer output')
                require(payload['path'] == stem+'/acceptance.json'
                        and set(review) == {'kind', 'status', 'request', 'result', 'audits', 'weights', 'outputs'}
                        and review['kind'] == 'activation_acceptance_v1' and review['status'] == 'accepted'
                        and review['request'] == pending['ref'], 'Acceptance provenance differs')
                require(type(review['audits']) is dict and set(review['audits']) == set(AUDITS),
                        'Complete independent audit bundle required')
                result = self.document(review['result'])
                require(review['result']['path'] == stem+'/result.json'
                        and set(result) == {'kind', 'request', 'record', 'holdout_consumed'}
                        and result['kind'] == KIND and result['request'] == pending['ref']
                        and result['holdout_consumed'] is False, 'Result provenance/holdout differs')
                spec = pending['request']['spec']; row = result['record']
                rules.checked([row], [spec], quality=spec['stage'] != 'pilot')
                weights = review['weights']
                require(type(weights) is dict and set(weights) == {'raw', 'h200'}, 'Both endpoint weights required')
                raw, avg = weights['raw'], weights['h200']
                for ref in (raw, avg): ref_schema(self.root, ref)
                require(re.fullmatch(re.escape(stem) + rf'/checkpoints/final_{spec["steps"]:06d}_[0-9]+\.pt', raw['path']) is not None
                        and avg['path'] == stem+'/checkpoints/h200.pt' and raw['path'] != avg['path'],
                        'Checkpoint endpoint location differs')
                for name, ref in review['audits'].items():
                    audit = self.document(ref)
                    require(ref['path'] == stem+'/'+name+'-audit.json'
                            and audit.get('status') == 'passed' and audit.get('request') == pending['ref']
                            and audit.get('result') == review['result'] and audit.get('weights') == weights,
                            'Audit does not bind the accepted request, result and pair')
                    pin(ref)
                require(type(review['outputs']) is list, 'Complete immutable output inventory required')
                inventory = {r['path']: ref_schema(self.root, r) for r in review['outputs']}
                require(len(inventory) == len(review['outputs'])
                        and all(name.startswith(stem+'/') and name != payload['path'] for name in inventory),
                        'Unique slot-local output inventory required')
                current = {str(p.relative_to(self.root)) for p in (self.root/stem).rglob('*') if p.is_file()}
                require(current <= set(inventory) | {payload['path']}, 'Unreviewed trial output')
                for ref in (pending['ref'], review['result'], raw, avg, *review['audits'].values()):
                    require(inventory.get(ref['path']) == ref, 'Inventory omitted a reviewed artifact')
                # Missing files are checked at the end of the complete ledger;
                # only a later valid boundary can explain removed weights.
                for ref in review['outputs']: pin(ref)
                for ref in (payload, review['result'], raw, avg): pin(ref)
                state['pairs'].append([raw, avg]); state['reviews'].append(payload)
                state['pilots' if spec['stage'] == 'pilot' else 'records'].append(row)
                state['pending'] = None
            elif action == 'selection':
                expected = self._action(state)
                selection = self.document(payload)
                require(state['pending'] is None and expected['kind'] == 'save_selection'
                        and payload['path'] == 'evidence/meta/activation-selection.json'
                        and selection == dict(kind=KIND, choices=expected['value'], reviews=state['reviews'][2:]),
                        'Selection is not bound to the complete accepted screen')
                state['selection'], state['selection_ref'] = selection['choices'], payload
                pin(payload)
            elif action == 'boundary':
                expected = self._action(state)
                require(state['pending'] is None and expected['kind'] in ('review_pilot_boundary', 'review_screen_boundary'),
                        'Retention boundary out of order')
                boundary = 'pilot' if expected['kind'] == 'review_pilot_boundary' else 'screen'
                receipt = self.document(payload)
                require(payload['path'] == f'evidence/.retention/{"pilot" if boundary == "pilot" else "lr"}/activation-complete.json'
                        and set(receipt) == {'kind', 'status', 'boundary', 'reviews', 'selection', 'deleted', 'preserved'}
                        and receipt['kind'] == 'activation_retention_v1' and receipt['status'] == 'completed'
                        and receipt['boundary'] == boundary and receipt['reviews'] == state['reviews']
                        and receipt['selection'] == state['selection_ref'], 'Retention receipt binding differs')
                eligible = self._eligible(state, boundary)
                require(type(receipt['deleted']) is list and type(receipt['preserved']) is list, 'Explicit inventory lists required')
                deleted = {r['path']: ref_schema(self.root, r) for r in receipt['deleted']}
                # This version requires the complete declared removal, never partial cleanup credit.
                require(len(deleted) == len(receipt['deleted']) and deleted == eligible,
                        'Cleanup must exactly match successful pilots or losing screen pairs')
                preserved = {r['path']: ref_schema(self.root, r) for r in receipt['preserved']}
                required = {k: v for k, v in state['pins'].items() if k not in deleted and k not in state['deleted']}
                require(len(preserved) == len(receipt['preserved']) and all(preserved.get(k) == v for k, v in required.items()),
                        'Boundary must preserve all pinned nonweights and protected pairs')
                for ref in receipt['preserved']: pin(ref)
                for name, ref in deleted.items():
                    require(not (self.root/name).exists(), 'Cleanup not actually complete')
                    state['deleted'][name] = ref
                pin(payload); state[boundary+'_boundary'] = True
            elif action == 'failed':
                failure = self.document(payload)
                require(state['pending'] is not None and payload['path'] == state['pending']['stem']+'/session-failure.json'
                        and failure.get('request') == state['pending']['ref'] and failure.get('retry_allowed') is False,
                        'Failure must bind the pending attempt')
                state['failed'] = True; pin(payload)
            else:
                raise ValueError('Unknown session event')
            pin(event_ref)
        if verify_pins:
            for name, ref in state['pins'].items():
                if name not in state['deleted']: self.verify(ref)
        # An orphan request is evidence of interrupted publication, not permission to retry.
        known = {p['path'] for p in state['pins'].values()}
        for stage in ('pilots', 'quality'):
            for p in (self.root/'evidence'/stage).glob('*/request.json'):
                require(str(p.relative_to(self.root)) in known, 'Unjournaled request; manual failure review required')
        state['events'] = [ref for _, ref in events]
        state['next'] = (dict(kind='failed') if state['failed'] else dict(kind='pending')
                         if state['pending'] is not None else self._action(state))
        state['dispatch_authorized'] = False
        return state

    def _append(self, state, action, payload):
        require(len(state['events']) < MAX_EVENTS, 'Finite event cap exceeded')
        require(not (self.root/'STOP').exists() and not (self.root/'STOP').is_symlink(), 'STOP forbids session progress')
        index = len(state['events'])
        event = dict(kind=KIND, index=index, previous=state['events'][-1] if index else None,
                     plan=self.plan, action=action, payload=payload)
        self.writer.publish_json(f'evidence/meta/session/{index:04d}.json', event)

    def reserve(self):
        with self.writer._serialized():
            state = self.inspect(); action = state['next']
            require(action['kind'] in ('pilot', 'quality'), 'No trial reservation available')
            require(not (self.root/'STOP').exists(), 'STOP forbids requests')
            stage = 'pilots' if action['kind'] == 'pilot' else 'quality'
            slot = len(state['pilots']) if stage == 'pilots' else len(state['records'])
            stem = f'evidence/{stage}/{slot:02d}'
            require(not (self.root/stem).exists() or not any((self.root/stem).iterdir()),
                    'Existing trial outputs cannot be retried')
            # Session owner holds both worker locks; exclusive publication rejects
            # competing calls, retaining any interrupted request as an orphan.
        request = dict(kind=KIND, plan=self.plan, spec=action['spec'], config=config(self.root/stem, action['spec']))
        self.writer.publish_json(stem+'/request.json', request)
        ref = reference(self.root, stem+'/request.json')
        self._append(state, 'reserve', ref)
        return ref

    def record(self, action, ref):
        require(action in ('accept', 'selection', 'boundary', 'failed'), 'Unsupported event mutation')
        events = self._events()
        # A completed removal is checked against the proposed boundary before
        # rehashing weights. No other mutation may explain missing artifacts.
        state = self._replay(events, verify_pins=action != 'boundary')
        require(not state['failed'], 'Failed sessions cannot resume')
        proposed = dict(kind=KIND, index=len(events), previous=events[-1][1] if events else None,
                        plan=self.plan, action=action, payload=ref)
        # Full dry replay validates the prospective receipt and every remaining
        # pin before exclusive publication; there is no mutable state shortcut.
        self._replay(events + [(proposed, self.plan)])
        self._append(state, action, ref)

    def save_selection(self):
        state = self.inspect(); action = state['next']
        require(action['kind'] == 'save_selection', 'Complete screen required')
        name = 'evidence/meta/activation-selection.json'
        self.writer.publish_json(name, dict(kind=KIND, choices=action['value'], reviews=state['reviews'][2:]))
        ref = reference(self.root, name); self.record('selection', ref)
        return ref
