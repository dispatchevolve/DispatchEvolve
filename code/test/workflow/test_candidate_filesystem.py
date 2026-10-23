"""Regression tests for OS-enforced worker filesystem boundaries.

Long-term coverage of candidate_worker and candidate_filesystem: native and
Python I/O outside scratch is denied, including symlink traversal.
"""
from pathlib import Path
import pandas as pd
import pytest
from dispatchevolve.tasks.full_dispatch.candidate_executor import LocalProcessCandidateRunner


@pytest.mark.parametrize('operation', ['read', 'write', 'native_read', 'native_write', 'symlink'])
def test_candidate_cannot_access_external_files(tmp_path: Path, operation: str):
    secret = tmp_path / 'outside.txt'
    secret.write_text('private sentinel')
    engine = tmp_path / 'engine'
    (engine / 'policies').mkdir(parents=True)
    (engine / 'policies/__init__.py').write_text('')
    (engine / 'README.md').write_text('Filesystem boundary test engine.')
    if operation == 'symlink':
        (engine / 'link').symlink_to(secret)
    target = engine / 'link' if operation == 'symlink' else secret
    if operation.startswith('native'):
        flags = 1 if operation == 'native_write' else 0
        probe = f"fd = ctypes.CDLL(None).open({str(target).encode()!r}, {flags}); blocked = fd < 0\n    if fd >= 0: os.close(fd)"
    else:
        mode = 'w' if operation == 'write' else 'r'
        probe = f"blocked = False\n    try:\n        with open({str(target)!r}, {mode!r}) as f: f.read() if {mode!r} == 'r' else f.write('changed')\n    except PermissionError: blocked = True"
    # Link traversal is tested directly against the OS boundary. Codebase
    # validation also rejects symlinks; create it from inside the candidate.
    if operation == 'symlink':
        (engine / 'link').unlink()
        target = Path('scratch-link')
        probe = f"os.symlink({str(secret)!r}, 'scratch-link')\n    blocked = False\n    try:\n        open('scratch-link').read()\n    except PermissionError: blocked = True"
    (engine / 'engine.py').write_text(
        'import ctypes\nimport os\ndef run_batch(frame, *, trace=True):\n    '
        + probe + "\n    frame['weight'] = float(blocked)\n    return frame, {}\n"
    )
    runner = LocalProcessCandidateRunner(workspace=tmp_path / 'runner')
    output, _ = runner(engine, pd.DataFrame({'batch_id': [1], 'weight': [0.]}), trace=True, progress_callback=None)
    assert output['weight'].tolist() == [1.]
    assert secret.read_text() == 'private sentinel'
