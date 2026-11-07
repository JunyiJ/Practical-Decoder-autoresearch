"""Bounded ASCII-decimal role lookup, SHA and binary skip; no model/RNG imports."""
import hashlib
import re


def scan(chunks, needles):
    """Find complete digit runs, including chunk splits, without unbounded carry.

    A NUL anywhere marks the file binary and discards all matches. Callers decide
    inventory roots/exclusions and own locks. Numeric literals are not RNG calls.
    """
    if type(needles) is not set or not needles or any(type(v) is not bytes or not v or not v.isdigit() for v in needles):
        raise ValueError('Nonempty distinct decimal byte strings required')
    low=min(map(len,needles));high=max(map(len,needles))
    if high>19:raise ValueError('At most19decimal digits per bounded role required')
    pattern=re.compile(rb'[0-9]{'+str(low).encode()+rb',}')
    leading=re.compile(rb'[0-9]*');trailing=re.compile(rb'[0-9]+$')
    pending=None;overflow=False;start=0;position=0;found=[];digest=hashlib.sha256()
    for chunk in chunks:
        if type(chunk) is not bytes:raise ValueError('Byte chunks required')
        if not chunk:continue
        if b'\0' in chunk:return dict(binary=True,bytes=position+len(chunk),sha256=None,matches=[])
        digest.update(chunk);begin=0
        if pending is not None or overflow:
            begin=leading.match(chunk).end()
            if not overflow:
                if len(pending)+begin>high:pending=None;overflow=True
                else:pending+=chunk[:begin]
            if begin==len(chunk):position+=len(chunk);continue
            if not overflow and pending in needles:found.append(dict(offset=start,value=pending.decode('ascii')))
            pending=None;overflow=False
        for match in pattern.finditer(chunk,begin):
            if match.end()<len(chunk) and match.end()-match.start()<=high:
                value=match.group()
                if value in needles:found.append(dict(offset=position+match.start(),value=value.decode('ascii')))
        # Only a final run can cross into the next chunk. high+1 distinguishes
        # an overlong run; its entire byte string never has to remain in memory.
        tail=trailing.search(chunk[-(high+1):])
        if tail is not None:
            length=tail.end()-tail.start();start=position+len(chunk)-length
            overflow=length>high;pending=None if overflow else tail.group()
        position+=len(chunk)
    if not overflow and pending is not None and pending in needles:
        found.append(dict(offset=start,value=pending.decode('ascii')))
    return dict(binary=False,bytes=position,sha256=digest.hexdigest(),matches=found)
