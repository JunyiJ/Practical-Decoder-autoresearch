"""Child argument decoding only; no worker body, process or model launches."""
import sys
import pytest
from src.research import tokenizer_wide_hybrid_quality_entry_v2 as entry
from test_tokenizer_wide_hybrid_metadata_writer import writer


def args(writer,slot=0):
    startup=dict(path='meta/quality-first-startup.json' if slot==0 else f'meta/quality-lr-startup-{slot:02d}.json',sha256='a'*64)
    ready=dict(path=f'meta/quality-readiness-{slot:02d}.json',sha256='b'*64)
    return entry.worker_argv(writer,slot,startup,ready)


@pytest.mark.parametrize('slot',[0,1,8])
def test_exact_cli_passes_both_hashes_and_inherited_descriptors(writer,monkeypatch,slot):
    argv=args(writer,slot);seen=[]
    def worker(w,index,startup,ready,actual):
        assert w.root==writer.root and w.lock_fds==writer.lock_fds and index==slot and actual==argv
        assert startup['sha256']=='a'*64 and ready['sha256']=='b'*64
        seen.append(True);return 'mocked only'
    monkeypatch.setattr(entry,'run_worker',worker)
    assert entry.main(argv[3:])=='mocked only' and seen==[True]


@pytest.mark.parametrize('bad',['missing','extra','abbreviation','subcommand','slot','sha','fd_alias','duplicate','order','leading_zero'])
def test_noncanonical_or_incomplete_cli_never_reaches_body(writer,monkeypatch,bad):
    argv=args(writer)[3:]
    if bad=='missing':argv=argv[:-2]
    elif bad=='extra':argv+=['--unknown','1']
    elif bad=='abbreviation':argv[argv.index('--startup-sha256')]='--startup'
    elif bad=='subcommand':argv[0]='run'
    elif bad=='slot':argv[argv.index('--slot')+1]='9'
    elif bad=='sha':argv[-1]='A'*64
    elif bad=='fd_alias':argv[argv.index('--campaign-fd')+1]=str(writer.lock_fds[0])
    elif bad=='duplicate':argv+=['--slot','0']
    elif bad=='order':argv=argv[:1]+argv[-2:]+argv[1:-2]
    else:argv[argv.index('--slot')+1]='00'
    monkeypatch.setattr(entry,'run_worker',lambda *a,**kw:pytest.fail('Body must not execute'))
    with pytest.raises((SystemExit,ValueError)):entry.main(argv)


def test_default_argv_uses_actual_command(writer,monkeypatch):
    argv=args(writer);monkeypatch.setattr(sys,'argv',['module.py']+argv[3:]);seen=[]
    monkeypatch.setattr(entry,'run_worker',lambda *a:seen.append(a[-1]))
    entry.main();assert seen==[argv]
