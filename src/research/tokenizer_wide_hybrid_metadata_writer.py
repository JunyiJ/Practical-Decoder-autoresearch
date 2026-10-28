"""Bounded v2 metadata producers, not a dispatcher or checkpoint writer.

Callers must already own inherited global/campaign locks and all launch gates.
An additional metadata lock serializes parent/child writers sharing those locks.
Partial files remain after failure; no automatic retry or evidence overwrite.
Unconverted producers still need monitoring and cannot claim these write bounds.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import time

from .tokenizer_wide_hybrid_cleanup import _locks, _sync_directory
from .tokenizer_wide_hybrid_metadata import budgets, classify, snapshot, MAX_FILES
from .tokenizer_wide_hybrid_retention import RESERVE
from .tokenizer_wide_hybrid_retention_receipts import _path

LOCK_NAME='control/.metadata.lock'
LOCK_SECONDS=2.
CHUNK_BYTES=65536


def _write(fd, data):
    view=memoryview(data)
    while view:
        size=os.write(fd,view[:CHUNK_BYTES])
        if size<=0:raise OSError('Metadata write made no progress')
        view=view[size:]


class MetadataWriter:
    def __init__(self, campaign, lock_fds):
        self.root=Path(campaign).absolute()
        if self.root!=self.root.resolve() or not self.root.is_dir():
            raise ValueError('Canonical existing campaign required')
        self.lock_fds=tuple(lock_fds)
        _locks(self.root/'evidence',self.lock_fds)
        with self._serialized():snapshot(self.root)

    @contextmanager
    def _serialized(self):
        _locks(self.root/'evidence',self.lock_fds)
        path=_path(self.root,LOCK_NAME)
        path.parent.mkdir(exist_ok=True)
        # The empty synchronization file is itself in the fixed control budget.
        if not path.exists():
            before=snapshot(self.root)
            if len(before['files'])>=MAX_FILES:
                raise ValueError('Metadata lock would exceed file-count cap')
        fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            start=time.monotonic()
            while True:
                try:
                    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()-start>=LOCK_SECONDS:
                        raise TimeoutError('Metadata producer lock timeout; no retry')
                    time.sleep(.01)
            st=os.fstat(fd);current=path.stat()
            if (not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size!=0
                    or (st.st_dev,st.st_ino)!=(current.st_dev,current.st_ino)):
                raise ValueError('Metadata lock identity/contents changed')
            _locks(self.root/'evidence',self.lock_fds)
            yield
        finally:
            os.close(fd)

    def _target(self,name):
        path=_path(self.root,name);group=classify(name)
        if (name in ('.lock',LOCK_NAME) or name.endswith('.tmp') or group.endswith('/weights')):
            raise ValueError('Reserved or checkpoint output requires a separate producer')
        return path,group

    def _capacity(self,group,additional):
        observed=snapshot(self.root)
        if observed['groups'][group]+additional>budgets()[group]:
            raise ValueError('Metadata producer group cap exceeded')
        if shutil.disk_usage(self.root).free<RESERVE+additional:
            raise ValueError('Metadata write would cross disk reserve')
        return observed

    def publish_json(self,name,value,*,replace=False):
        """Exclusive immutable JSON, or atomic replacement only under control/.

        Encoding streams to a fresh bounded .tmp; failed partial .tmp files are
        retained. Old control contents remain until the complete fsynced payload
        fits alongside them. Existing evidence and failed attempts are refused.
        """
        with self._serialized():
            path,group=self._target(name)
            if replace and not name.startswith('control/'):
                raise ValueError('Only mutable control documents may be replaced')
            temporary=_path(self.root,name+'.tmp')
            if temporary.exists() or path.exists() and not replace:
                raise FileExistsError('Existing metadata attempt cannot be retried or overwritten')
            before=self._capacity(group,0)
            # Reserve both names for exclusive hard-link publication, including
            # the transient interval before unlinking the temporary name.
            added=1 if path.exists() else 2
            if len(before['files'])+added>MAX_FILES:raise ValueError('Metadata file-count cap exceeded')
            path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try:
                remaining=budgets()[group]-before['groups'][group]
                encoder=json.JSONEncoder(sort_keys=True,allow_nan=False,indent=2,ensure_ascii=True)
                for piece in encoder.iterencode(value):
                    # ensure_ascii guarantees this byte count before allocation.
                    if len(piece)>remaining:raise ValueError('JSON payload exceeds metadata group cap')
                    data=piece.encode('ascii')
                    if shutil.disk_usage(self.root).free<RESERVE+len(data):
                        raise ValueError('Metadata write would cross disk reserve')
                    _write(fd,data);remaining-=len(data)
                if remaining<1:raise ValueError('JSON newline exceeds metadata group cap')
                if shutil.disk_usage(self.root).free<RESERVE+1:raise ValueError('Disk reserve crossed')
                _write(fd,b'\n');os.fsync(fd)
            finally:
                os.close(fd)
            if replace:os.replace(temporary,path)
            else:
                os.link(temporary,path)
                _sync_directory(path.parent)
                temporary.unlink()
            _sync_directory(path.parent)
            snapshot(self.root)
        return path

    def open_append(self,name):
        """New append-only byte/JSONL stream; never resumes an existing file."""
        with self._serialized():
            path,group=self._target(name)
            if name=='STOP':raise ValueError('STOP is an immutable marker, not a stream')
            before=self._capacity(group,0)
            if len(before['files'])+1>MAX_FILES:raise ValueError('Metadata file-count cap exceeded')
            path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try:
                os.fsync(fd);_sync_directory(path.parent)
                stream=BoundedAppend(self,name,group,fd)
            except BaseException:
                os.close(fd);raise
        return stream


class BoundedAppend:
    """Owns a private FD; each append is checked before any payload byte is written."""
    def __init__(self,writer,name,group,fd):
        self.writer,self.name,self.group,self._fd=writer,name,group,fd
        st=os.fstat(fd);self._inode=(st.st_dev,st.st_ino);self._size=0
        self.closed=False

    def write(self,data):
        if self.closed:raise ValueError('Closed/failed metadata stream cannot be reused')
        try:
            if type(data) is not bytes:raise TypeError('Metadata stream accepts exact bytes')
            with self.writer._serialized():
                path=_path(self.writer.root,self.name);saved=path.stat();actual=os.fstat(self._fd)
                if ((saved.st_dev,saved.st_ino)!=self._inode or (actual.st_dev,actual.st_ino)!=self._inode
                        or saved.st_nlink!=1 or saved.st_size!=self._size or actual.st_size!=self._size):
                    raise ValueError('Append-only metadata identity/size changed')
                self.writer._capacity(self.group,len(data))
                _write(self._fd,data);os.fsync(self._fd);self._size+=len(data)
                snapshot(self.writer.root)
        except BaseException:
            self.close();raise
        return len(data)

    def close(self):
        if not self.closed:
            self.closed=True
            try:os.fsync(self._fd)
            finally:os.close(self._fd)

    def __enter__(self):return self
    def __exit__(self,*args):self.close();return False
