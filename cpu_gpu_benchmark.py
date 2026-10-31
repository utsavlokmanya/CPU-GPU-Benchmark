"""
CPU vs GPU benchmark for the paper "CPU vs GPU Computing: Faster Scientific Applications".

Run in Google Colab (Runtime > Change runtime type > T4 GPU):
    !pip install cupy-cuda12x scipy matplotlib pandas
    !python cpu_gpu_benchmark.py

Outputs: results.csv (fill Table II from this) and speedup.png (use as Fig. 3).
Use --quick for a fast smoke test.
"""
import argparse, platform, statistics, time
import numpy as np
import pandas as pd
from scipy import ndimage as cpu_ndimage

try:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    GPU = True
except Exception as e:           # no CuPy / no GPU -> CPU-only run
    cp = None; gpu_ndimage = None; GPU = False
    print("GPU not available, running CPU only:", e)

REPEATS = 5

def sync():
    if GPU: cp.cuda.Device().synchronize()

def timeit(fn):
    fn(); sync()                                  # warm-up (excluded)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)                  # ms

# ---------------- workloads (xp = numpy or cupy) ----------------
def matmul(xp, n):
    rng = np.random.default_rng(0)
    a = rng.random((n, n), dtype=np.float32); b = rng.random((n, n), dtype=np.float32)
    return (a, b), lambda d: d[0] @ d[1]

def conv2d(xp, n):
    rng = np.random.default_rng(0)
    img = rng.random((n, n), dtype=np.float32); k = np.ones((5, 5), np.float32) / 25
    mod = gpu_ndimage if xp is cp else cpu_ndimage
    return (img, k), lambda d: mod.convolve(d[0], d[1])

def nbody(xp, n):
    rng = np.random.default_rng(0)
    pos = rng.random((n, 3), dtype=np.float32)
    def f(d):
        p = d[0]; diff = p[None, :, :] - p[:, None, :]
        r2 = (diff ** 2).sum(-1) + 1e-3
        return (diff * (r2 ** -1.5)[..., None]).sum(1)
    return (pos,), f

def analytics(xp, n):
    rng = np.random.default_rng(0)
    x = rng.random(n, dtype=np.float32)
    return (x,), lambda d: (d[0].mean(), d[0].std(), xp.sort(d[0]))

WORKLOADS = {
    "Matrix multiplication": (matmul,   [256, 512, 1024, 2048, 4096]),
    "Image filtering":       (conv2d,   [256, 512, 1024, 2048, 4096]),
    "N-body (molecular proxy)": (nbody, [512, 1024, 2048, 4096]),
    "Data analytics":        (analytics, [10**6, 10**7, 5 * 10**7]),
}
QUICK = {"Matrix multiplication": [128, 256], "Image filtering": [128, 256],
         "N-body (molecular proxy)": [256, 512], "Data analytics": [10**5, 10**6]}

def to_host(r):
    if isinstance(r, tuple): return [to_host(i) for i in r]
    return cp.asnumpy(r) if (GPU and isinstance(r, cp.ndarray)) else np.asarray(r)

def main(quick):
    rows = []
    for name, (build, sizes) in WORKLOADS.items():
        for n in (QUICK[name] if quick else sizes):
            data_h, f = build(np, n)
            cpu_ms = timeit(lambda: f(data_h))
            row = dict(workload=name, size=n, cpu_ms=cpu_ms,
                       gpu_kernel_ms=np.nan, gpu_total_ms=np.nan,
                       speedup_kernel=np.nan, speedup_total=np.nan, max_err=np.nan)
            if GPU:
                _, fg = build(cp, n)
                data_d = tuple(cp.asarray(a) for a in data_h)
                row["gpu_kernel_ms"] = timeit(lambda: fg(data_d))
                def total():
                    d = tuple(cp.asarray(a) for a in data_h)      # host -> device
                    return to_host(fg(d))                          # kernel + device -> host
                row["gpu_total_ms"] = timeit(total)
                a = np.asarray(f(data_h)[0] if name == "Data analytics" else f(data_h), dtype=np.float64)
                b = np.asarray(to_host(fg(data_d))[0] if name == "Data analytics" else to_host(fg(data_d)), dtype=np.float64)
                row["max_err"] = float(np.max(np.abs(a - b)))      # validation
                row["speedup_kernel"] = cpu_ms / row["gpu_kernel_ms"]
                row["speedup_total"] = cpu_ms / row["gpu_total_ms"]
            rows.append(row); print(row)
    df = pd.DataFrame(rows); df.to_csv("results.csv", index=False)
    print("\nCPU:", platform.processor() or platform.machine(), "| logical cores:", __import__("os").cpu_count())
    if GPU:
        print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
        for name, g in df.groupby("workload", sort=False):
            ax.plot(g["size"], g["speedup_total"], marker="o", ms=3, lw=1, label=name)
        ax.axhline(1, color="k", ls="--", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Problem size"); ax.set_ylabel("Speed-up (CPU time / GPU time)")
        ax.legend(fontsize=5); ax.tick_params(labelsize=6); fig.tight_layout(); fig.savefig("speedup.png")
    print(df.round(3).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
