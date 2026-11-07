"""Linux supervisor-only handoff; never signal a training process group."""
import os
from pathlib import Path
import signal
import time


def stat(pid, proc=Path('/proc')):
    text = (proc / str(pid) / 'stat').read_text()
    tail = text[text.rindex(')') + 2:].split()
    return dict(pid=int(text.split(' ', 1)[0]), state=tail[0],
                ppid=int(tail[1]), start_ticks=int(tail[19]), wait_status=int(tail[49]))


def same_process(identity):
    current = stat(identity['pid'])
    if current['start_ticks'] != identity['start_ticks']:
        raise RuntimeError('PID identity changed; refusing to inspect or signal it')
    return current


def freeze(identity):
    current = same_process(identity)
    if current['state'] in ('T', 't', 'Z', 'X'):
        raise RuntimeError('Expected a live, unsuspended source controller')
    os.kill(identity['pid'], signal.SIGSTOP)
    for _ in range(100):
        if same_process(identity)['state'] == 'T':
            return
        time.sleep(.02)
    raise RuntimeError('Controller did not suspend')


def assert_frozen(identity):
    if same_process(identity)['state'] != 'T':
        raise RuntimeError('Source controller is no longer suspended')


def rollback_freeze(identity):
    assert_frozen(identity)
    os.kill(identity['pid'], signal.SIGCONT)


def adopted_poll(identity):
    current = same_process(identity)
    if current['ppid'] != identity['ppid']:
        raise RuntimeError('Adopted worker parent changed; exit evidence may be lost')
    if current['state'] == 'Z':
        return os.waitstatus_to_exitcode(current['wait_status'])
    if current['state'] in ('X', 'T', 't'):
        raise RuntimeError('Adopted worker is unexpectedly dead or stopped')
    return None


def retire(identity, children):
    """Retire only a suspended supervisor whose recorded children all exited."""
    assert_frozen(identity)
    if not children or any(adopted_poll(child) is None for child in children):
        raise RuntimeError('Cannot retire a source controller with live workers')
    os.kill(identity['pid'], signal.SIGKILL)


def stop_worker_for_storage(identity):
    """Emergency only: preserve the original 5GB guard for adopted workers."""
    if adopted_poll(identity) is not None:
        return adopted_poll(identity)
    if os.getpgid(identity['pid']) != identity['pid']:
        raise RuntimeError('Adopted worker is not its own process-group leader')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        same_process(identity)
        os.killpg(identity['pid'], sig)
        for _ in range(200):
            rc = adopted_poll(identity)
            if rc is not None:
                return rc
            time.sleep(.1)
    raise RuntimeError('Owned adopted worker failed to exit after storage stop')
