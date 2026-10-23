"""Irreversible OS filesystem restrictions for candidate workers."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys


def restrict_filesystem(engine_dir: Path) -> None:
    """Allow engine/runtime reads and scratch writes; fail if unavailable."""
    scratch = Path.cwd().resolve()
    reads = {engine_dir.resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    reads.update(Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    # Python extension modules may depend on system libraries outside the prefix.
    reads.update(Path(p) for p in ('/usr/lib', '/usr/share/zoneinfo', '/lib', '/lib64',
                                  '/System/Library', '/etc/localtime', '/etc/ld.so.cache', '/dev/null', '/dev/urandom')
                 if Path(p).exists())
    if sys.platform == 'darwin':
        library = ctypes.CDLL('/usr/lib/libsandbox.dylib', use_errno=True)
        library.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
        library.sandbox_init.restype = ctypes.c_int
        error = ctypes.c_char_p()
        def subpath(path):
            return '(subpath ' + json.dumps(str(path.resolve())) + ')'
        profile = '(version 1)(allow default)(deny file-read*)(deny file-write*)(deny network*)'
        profile += '(allow file-read* ' + ' '.join(subpath(p) for p in reads | {scratch}) + ')'
        profile += '(allow file-write* ' + subpath(scratch) + ')'
        if library.sandbox_init(profile.encode(), 0, ctypes.byref(error)):
            raise RuntimeError('candidate filesystem sandbox initialization failed')
    elif sys.platform == 'linux':
        _landlock(reads, scratch)
    else:
        raise RuntimeError('candidate filesystem isolation requires macOS sandbox or Linux Landlock')


def _landlock(reads: set[Path], scratch: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    abi = library.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError('Linux Landlock ABI 3 or newer is required for candidate filesystem isolation')
    # Filesystem access rights through REFER (ABI 2), TRUNCATE (ABI 3).
    rights = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    attr = Ruleset(rights)
    descriptor = library.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), 'Landlock ruleset creation failed')
    try:
        for path in reads | {scratch}:
            fd = os.open(path.resolve(), os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = rights if path == scratch else (1 << 0) | (1 << 2) | (1 << 3)
                if not path.is_dir():
                    allowed &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
                rule = Beneath(allowed, fd)
                if library.syscall(445, descriptor, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'Landlock path rule failed')
            finally:
                os.close(fd)
        if library.prctl(38, 1, 0, 0, 0) or library.syscall(446, descriptor, 0):
            raise OSError(ctypes.get_errno(), 'Landlock restriction failed')
    finally:
        os.close(descriptor)
