"""C ABI for Prophet's additive-model numeric kernels."""

from std.algorithm import parallelize
from std.math import cos, exp, sin, sqrt
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_MATVEC_MIN_VALUES = 1_000_000
comptime MATVEC_CHUNKS = 32
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def axpy(alpha: Float64, x: Ptr, y: Ptr, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.store(i, y.load[width=W](i) + va * x.load[width=W](i))
        i += W
    while i < n:
        y[i] += alpha * x[i]
        i += 1


def cholesky(a: Ptr, d: Int) -> Bool:
    for i in range(d):
        for j in range(i + 1):
            var acc = a[i * d + j]
            for k in range(j):
                acc -= a[i * d + k] * a[j * d + k]
            if i == j:
                if acc <= 0.0:
                    return False
                a[i * d + i] = sqrt(acc)
            else:
                a[i * d + j] = acc / a[j * d + j]
    return True


def cholesky_solve(l: Ptr, b: Ptr, d: Int):
    for i in range(d):
        var acc = b[i]
        for k in range(i):
            acc -= l[i * d + k] * b[k]
        b[i] = acc / l[i * d + i]
    for ri in range(d):
        var i = d - 1 - ri
        var acc = b[i]
        for k in range(i + 1, d):
            acc -= l[k * d + i] * b[k]
        b[i] = acc / l[i * d + i]


def ridge_accumulate(
    x: Ptr,
    y: Ptr,
    coef: Ptr,
    work: Ptr,
    first: Int,
    last: Int,
    d: Int,
):
    for r in range(first, last):
        var row = x + r * d
        axpy(y[r], row, coef, d)
        for i in range(d):
            if row[i] != 0.0:
                axpy(row[i], row, work + i * d, i + 1)


def ridge_finish(penalty: Ptr, coef: Ptr, work: Ptr, d: Int) -> Int:
    for i in range(d):
        work[i * d + i] += penalty[i]
        for j in range(i + 1, d):
            work[i * d + j] = work[j * d + i]
    if not cholesky(work, d):
        return 0
    cholesky_solve(work, coef, d)
    return 1


@export("mop_fourier_series")
def mop_fourier_series(
    days_addr: Int,
    dst_addr: Int,
    n: Int,
    period: Float64,
    order: Int,
) abi("C") -> Int:
    if n < 0 or order <= 0 or period <= 0.0:
        return 0
    if n == 0:
        return 1
    if days_addr == 0 or dst_addr == 0:
        return 0
    var days = p(days_addr)
    var dst = p(dst_addr)
    var tau = 6.283185307179586476925286766559
    for r in range(n):
        var base = tau * days[r] / period
        for j in range(order):
            var angle = Float64(j + 1) * base
            dst[r * 2 * order + 2 * j] = sin(angle)
            dst[r * 2 * order + 2 * j + 1] = cos(angle)
    return 1


@export("mop_piecewise_linear")
def mop_piecewise_linear(
    t_addr: Int,
    delta_addr: Int,
    cp_addr: Int,
    dst_addr: Int,
    n: Int,
    s: Int,
    k: Float64,
    m: Float64,
) abi("C") -> Int:
    if n < 0 or s < 0:
        return 0
    if n == 0:
        return 1
    if t_addr == 0 or dst_addr == 0:
        return 0
    if s > 0 and (delta_addr == 0 or cp_addr == 0):
        return 0
    var t = p(t_addr)
    var dst = p(dst_addr)
    if s == 0:
        for r in range(n):
            dst[r] = k * t[r] + m
        return 1
    var deltas = p(delta_addr)
    var cps = p(cp_addr)
    for r in range(n):
        var rate = k
        var offset = m
        for j in range(s):
            if cps[j] <= t[r]:
                rate += deltas[j]
                offset -= deltas[j] * cps[j]
        dst[r] = rate * t[r] + offset
    return 1


@export("mop_piecewise_logistic")
def mop_piecewise_logistic(
    t_addr: Int,
    cap_addr: Int,
    delta_addr: Int,
    cp_addr: Int,
    dst_addr: Int,
    gamma_addr: Int,
    n: Int,
    s: Int,
    k: Float64,
    m: Float64,
) abi("C") -> Int:
    if n < 0 or s < 0:
        return 0
    if n == 0:
        return 1
    if t_addr == 0 or cap_addr == 0 or dst_addr == 0:
        return 0
    if s > 0 and (delta_addr == 0 or cp_addr == 0 or gamma_addr == 0):
        return 0
    var t = p(t_addr)
    var cap = p(cap_addr)
    var dst = p(dst_addr)
    if s == 0:
        for r in range(n):
            dst[r] = cap[r] / (1.0 + exp(-k * (t[r] - m)))
        return 1
    var deltas = p(delta_addr)
    var cps = p(cp_addr)
    var gammas = p(gamma_addr)
    var cumulative_delta = 0.0
    var cumulative_gamma = 0.0
    for j in range(s):
        var old_rate = k + cumulative_delta
        cumulative_delta += deltas[j]
        gammas[j] = (cps[j] - m - cumulative_gamma) * (
            1.0 - old_rate / (k + cumulative_delta)
        )
        cumulative_gamma += gammas[j]
    for r in range(n):
        var rate = k
        var offset = m
        for j in range(s):
            if t[r] >= cps[j]:
                rate += deltas[j]
                offset += gammas[j]
        dst[r] = cap[r] / (1.0 + exp(-rate * (t[r] - offset)))
    return 1


@export("mop_ridge_fit")
def mop_ridge_fit(
    x_addr: Int,
    y_addr: Int,
    penalty_addr: Int,
    coef_addr: Int,
    work_addr: Int,
    n: Int,
    d: Int,
) abi("C") -> Int:
    if n <= 0 or d <= 0:
        return 0
    if (
        x_addr == 0
        or y_addr == 0
        or penalty_addr == 0
        or coef_addr == 0
        or work_addr == 0
    ):
        return 0
    var x = p(x_addr)
    var y = p(y_addr)
    var penalty = p(penalty_addr)
    var coef = p(coef_addr)
    var work = p(work_addr)
    for i in range(d * d):
        work[i] = 0.0
    for i in range(d):
        coef[i] = 0.0
    ridge_accumulate(x, y, coef, work, 0, n, d)
    return ridge_finish(penalty, coef, work, d)


@export("mop_ridge_fit_parallel")
def mop_ridge_fit_parallel(
    x_addr: Int,
    y_addr: Int,
    penalty_addr: Int,
    coef_addr: Int,
    work_addr: Int,
    scratch_addr: Int,
    n: Int,
    d: Int,
    chunks: Int,
) abi("C") -> Int:
    if n <= 0 or d <= 0 or chunks <= 0 or chunks > n:
        return 0
    if (
        x_addr == 0
        or y_addr == 0
        or penalty_addr == 0
        or coef_addr == 0
        or work_addr == 0
        or scratch_addr == 0
    ):
        return 0
    var x = p(x_addr)
    var y = p(y_addr)
    var penalty = p(penalty_addr)
    var coef = p(coef_addr)
    var work = p(work_addr)
    var scratch = p(scratch_addr)
    var scratch_coefs = scratch + chunks * d * d

    def accumulate_chunk(chunk: Int) capturing:
        var local_work = scratch + chunk * d * d
        var local_coef = scratch_coefs + chunk * d
        for i in range(d * d):
            local_work[i] = 0.0
        for i in range(d):
            local_coef[i] = 0.0
        var first = n * chunk // chunks
        var last = n * (chunk + 1) // chunks
        ridge_accumulate(x, y, local_coef, local_work, first, last, d)

    parallelize[accumulate_chunk](chunks, chunks)

    for i in range(d * d):
        work[i] = 0.0
    for i in range(d):
        coef[i] = 0.0
    for chunk in range(chunks):
        axpy(1.0, scratch_coefs + chunk * d, coef, d)
        var local_work = scratch + chunk * d * d
        for i in range(d):
            axpy(1.0, local_work + i * d, work + i * d, i + 1)
    return ridge_finish(penalty, coef, work, d)


@export("mop_matvec")
def mop_matvec(
    x_addr: Int,
    coef_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
) abi("C") -> Int:
    if n < 0 or d <= 0:
        return 0
    if n == 0:
        return 1
    if x_addr == 0 or coef_addr == 0 or dst_addr == 0:
        return 0
    var x = p(x_addr)
    var coef = p(coef_addr)
    var dst = p(dst_addr)

    def evaluate_chunk(chunk: Int) capturing:
        var first = n * chunk // MATVEC_CHUNKS
        var last = n * (chunk + 1) // MATVEC_CHUNKS
        for r in range(first, last):
            dst[r] = dot(x + r * d, coef, d)

    if n * d >= PARALLEL_MATVEC_MIN_VALUES:
        parallelize[evaluate_chunk](MATVEC_CHUNKS, MATVEC_CHUNKS)
    else:
        for r in range(n):
            dst[r] = dot(x + r * d, coef, d)
    return 1


@export("mop_component")
def mop_component(
    x_addr: Int,
    coef_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
    first: Int,
    width: Int,
) abi("C") -> Int:
    if n < 0 or d <= 0 or first < 0 or width <= 0 or first + width > d:
        return 0
    if n == 0:
        return 1
    if x_addr == 0 or coef_addr == 0 or dst_addr == 0:
        return 0
    var x = p(x_addr)
    var coef = p(coef_addr)
    var dst = p(dst_addr)
    for r in range(n):
        dst[r] = dot(x + r * d + first, coef + first, width)
    return 1
