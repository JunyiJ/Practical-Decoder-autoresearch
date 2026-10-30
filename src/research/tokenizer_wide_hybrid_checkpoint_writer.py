"""Exclusive, bounded streaming checkpoint producer for prospective v2 workers.

The caller supplies the independently frozen trajectory allowance and holds both
worker locks. This does not replace full campaign storage, time, memory or audit
gates. No old checkpoint saver or closed controller is changed here.
"""
import hashlib
import os
from pathlib import Path
import shutil
import threading
from time import sleep as _handoff
from time import monotonic as _monotonic

from .tokenizer_wide_hybrid_cleanup import _sync_directory
from .tokenizer_wide_hybrid_metadata import snapshot, MAX_FILES
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter, _write
from .tokenizer_wide_hybrid_retention import RESERVE
from .tokenizer_wide_hybrid_retention_receipts import _path

CHUNK_BYTES=1024**2
# Leave two metadata-lock polling intervals outside the namespace lock after
# each MiB or 20ms between handoffs. Small serializer callbacks can monopolize
# repeated short lock scopes without accumulating a MiB before the deadline.
# This is cooperative scheduling, not a relaxation of the two-second timeout.
HANDOFF_SECONDS=.02
HANDOFF_WORK_SECONDS=.02


class CheckpointWriter:
    def __init__(self,writer,stage,slot,allowance_bytes):
        if (not isinstance(writer,MetadataWriter) or stage not in ('pilots','quality')
                or type(slot) is not int or not 0<=slot<(3 if stage=='pilots' else 24)
                or type(allowance_bytes) is not int or allowance_bytes<=0):
            raise ValueError('Bounded metadata writer, exact slot and positive byte allowance required')
        self.writer=writer
        self.stem=f'evidence/{stage}/{slot:02d}/checkpoints'
        self.group=f'evidence/{stage}/{slot:02d}/weights'
        self.allowance=allowance_bytes

    def _capacity(self,additional):
        if (self.writer.root/'STOP').exists() or (self.writer.root/'STOP').is_symlink():
            raise ValueError('Campaign STOP forbids checkpoint writes')
        observed=snapshot(self.writer.root)
        used=sum(v['size'] for v in observed['files'].values() if v['group']==self.group)
        if used+additional>self.allowance:
            raise ValueError('Aggregate trajectory checkpoint allowance exceeded')
        if shutil.disk_usage(self.writer.root).free<RESERVE+additional:
            raise ValueError('Checkpoint write would cross disk reserve')
        return observed

    def open(self,name):
        """Create one new temporary stream; any unfinished slot weight blocks entry."""
        if (type(name) is not str or Path(name).name!=name or not name.endswith('.pt')
                or name in ('.pt','..pt') or '.tmp' in name):
            raise ValueError('Single final checkpoint basename ending in .pt required')
        relative=self.stem+'/'+name
        with self.writer._serialized():
            final=_path(self.writer.root,relative);temporary=_path(self.writer.root,relative+'.tmp')
            before=self._capacity(0)
            if final.exists() or any(k.endswith('.tmp') and v['group']==self.group
                                     for k,v in before['files'].items()):
                raise FileExistsError('Existing checkpoint attempt cannot be retried or overwritten')
            if len(before['files'])+2>MAX_FILES:raise ValueError('Checkpoint publication exceeds file-count cap')
            final.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try:
                os.fsync(fd);_sync_directory(final.parent)
                return CheckpointStream(self,relative,fd)
            except BaseException:
                os.close(fd);raise

    def save(self,name,payload):
        """torch.save directly to a bounded stream; never stages an archive in RAM."""
        import torch
        stream=self.open(name)
        try:
            torch.save(payload,stream)
            return stream.publish()
        finally:
            stream.close()


class CheckpointStream:
    def __init__(self,owner,name,fd):
        self.owner,self.name,self._fd=owner,name,fd
        st=os.fstat(fd);self._inode=(st.st_dev,st.st_ino)
        self._stamp=(st.st_mtime_ns,st.st_ctime_ns)
        self._size=0;self._digest=hashlib.sha256();self._mutex=threading.RLock()
        self._handoff_bytes=0
        self._handoff_started=_monotonic()
        self.closed=False

    def _yield_if_due(self,additional=0):
        """Called with the stream mutex, always outside the namespace lock.

        Wall time includes census/lock waits and time between callbacks. A
        single operation is not preempted; this is not a hard hold-time bound.
        """
        self._handoff_bytes+=additional
        if (self._handoff_bytes>=CHUNK_BYTES or
                _monotonic()-self._handoff_started>=HANDOFF_WORK_SECONDS):
            self._handoff_bytes%=CHUNK_BYTES
            _handoff(HANDOFF_SECONDS)
            self._handoff_started=_monotonic()

    def _identity(self):
        if self.closed:raise ValueError('Closed or failed checkpoint stream cannot be reused')
        path=_path(self.owner.writer.root,self.name+'.tmp')
        saved=path.stat();actual=os.fstat(self._fd)
        if ((saved.st_dev,saved.st_ino)!=self._inode or (actual.st_dev,actual.st_ino)!=self._inode
                or saved.st_nlink!=1 or actual.st_nlink!=1
                or saved.st_size!=self._size or actual.st_size!=self._size
                or (saved.st_mtime_ns,saved.st_ctime_ns)!=self._stamp
                or (actual.st_mtime_ns,actual.st_ctime_ns)!=self._stamp):
            raise ValueError('Checkpoint stream identity/size changed')

    def write(self,data):
        with self._mutex:
            try:
                # PyTorch supplies borrowed memoryviews as well as bytes. Copy
                # at most one chunk so hashing and writing see identical bytes.
                # The single worker must keep the payload stable during save.
                if type(data) not in (bytes,bytearray,memoryview):
                    raise TypeError('Checkpoint stream requires a contiguous byte buffer')
                view=memoryview(data).cast('B')
                with self.owner.writer._serialized():
                    self._identity();self.owner._capacity(len(view))
                self._yield_if_due()
                for start in range(0,len(view),CHUNK_BYTES):
                    piece=bytes(view[start:start+CHUNK_BYTES])
                    # Release the namespace lock between bounded writes so the
                    # independent resource recorder can sample/publish promptly.
                    with self.owner.writer._serialized():
                        self._identity();self.owner._capacity(len(piece))
                        _write(self._fd,piece)
                        self._digest.update(piece);self._size+=len(piece)
                        current=os.fstat(self._fd)
                        self._stamp=(current.st_mtime_ns,current.st_ctime_ns)
                    self._yield_if_due(len(piece))
                return len(view)
            except BaseException:
                self.close();raise

    def flush(self):
        with self._mutex:
            try:
                with self.owner.writer._serialized():
                    self._identity();self.owner._capacity(0);os.fsync(self._fd)
                self._yield_if_due()
            except BaseException:
                self.close();raise

    def tell(self):
        with self._mutex:
            if self.closed:raise ValueError('Closed checkpoint stream')
            return self._size

    def publish(self):
        with self._mutex:
            try:
                with self.owner.writer._serialized():
                    self._identity();observed=self.owner._capacity(0)
                    if len(observed['files'])+1>MAX_FILES:raise ValueError('Checkpoint publication exceeds file-count cap')
                    final=_path(self.owner.writer.root,self.name)
                    temporary=_path(self.owner.writer.root,self.name+'.tmp')
                    os.fsync(self._fd)
                    # Exclusive publication: no replace of a concurrently created
                    # final file. Both names exist only while holding the census lock.
                    os.link(temporary,final);_sync_directory(final.parent)
                    temporary.unlink();_sync_directory(final.parent)
                    self.close();self.owner._capacity(0)
                    return dict(path=str(final),size=self._size,sha256=self._digest.hexdigest())
            except BaseException:
                self.close();raise

    def close(self):
        with self._mutex:
            if not self.closed:
                self.closed=True
                try:os.fsync(self._fd)
                finally:os.close(self._fd)

    def __enter__(self):return self
    def __exit__(self,*args):self.close();return False
