"""ctypes bindings for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-prophet.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mop_fourier_series": ([I, I, I, F, I], I),
    "mop_piecewise_linear": ([I, I, I, I, I, I, F, F], I),
    "mop_piecewise_logistic": ([I, I, I, I, I, I, I, I, F, F], I),
    "mop_ridge_fit": ([I, I, I, I, I, I, I], I),
    "mop_ridge_fit_parallel": ([I, I, I, I, I, I, I, I, I], I),
    "mop_matvec": ([I, I, I, I, I], I),
    "mop_component": ([I, I, I, I, I, I, I], I),
}

_LIBRARY: ctypes.CDLL | None = None


def build() -> None:
    subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        check=True,
    )


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        if not os.path.exists(LIB):
            build()
        _LIBRARY = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype != np.float64 or not value.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous float64 arrays")
    return int(value.ctypes.data)
