#!/usr/bin/env python3
"""
nisar_probe.py - measures the link to ASF and traces every request while h5py opens one granule.
Run it in the same folder as nisar_region.py (same login as nisar_browser.py):

    python nisar_probe.py

Ctrl+C whenever you like and paste everything printed so far. Optional: python nisar_probe.py <url of any NISAR .h5>
"""
import os
import sys
import time

import nisar_region as nr

URL = sys.argv[1] if len(sys.argv) > 1 else (
    "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GUNW_PROVISIONAL_V1/"
    "NISAR_L2_PR_GUNW_021_013_D_071_022_4000_SH_20260522T024038_20260522T024116_20260603T024038_20260603T024115_P05023_N_F_J_001/"
    "NISAR_L2_PR_GUNW_021_013_D_071_022_4000_SH_20260522T024038_20260522T024116_20260603T024038_20260603T024115_P05023_N_F_J_001.h5")
GRID = "/science/LSAR/GUNW/grids/frequencyA"


def P(*a):
    print(*a, flush=True)


user, pw = nr.credentials()
os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"] = user, pw
import earthaccess
import h5py

for m in ("h5py", "fsspec", "aiohttp", "earthaccess", "requests", "numpy"):
    try:
        P(f"{m:12}", __import__(m).__version__)
    except Exception as ex:
        P(f"{m:12}", "n/a", ex)
P("python      ", sys.version.split()[0], sys.platform)
t = time.time()
earthaccess.login(strategy="environment")
P(f"login {time.time() - t:.1f}s")

P("\n== 1. plain HTTP range requests (this is the raw speed of the link)")
s = earthaccess.get_requests_https_session()
size = None


def rng(off, n, label):
    global size
    t = time.time()
    try:
        r = s.get(URL, headers={"Range": f"bytes={off}-{off + n - 1}"}, stream=True, timeout=(30, 120))
        first, got = time.time() - t, 0
        for blk in r.iter_content(256 * 1024):
            got += len(blk)
        dt = time.time() - t
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            size = int(cr.split("/")[-1])
        P(f"{label:22} status {r.status_code}  first byte {first:5.1f}s  total {dt:5.1f}s  {got / 1e6:6.2f} MB  {got / 1e6 / max(dt, 1e-9):5.2f} MB/s")
        return r
    except Exception as ex:
        P(f"{label:22} FAILED after {time.time() - t:.1f}s: {type(ex).__name__}: {ex}")


r = rng(0, 4096, "4 KB (with login)")
if r is not None:
    P("   headers:", {k: r.headers.get(k) for k in ("Server", "Via", "X-Cache", "X-Amz-Cf-Pop", "Accept-Ranges", "Content-Range") if r.headers.get(k)})
rng(0, 4096, "4 KB again")
rng(0, 1 << 20, "1 MB at start")
rng(0, 4 << 20, "4 MB at start")
rng(0, 8 << 20, "8 MB at start")
if size:
    P(f"   file size {size / 1e9:.2f} GB")
    rng(size - (8 << 20), 8 << 20, "8 MB at end")
    rng(size // 2, 8 << 20, "8 MB in the middle")


P("\n== 1b. parallel requests: do more connections give more speed? (decides how to fix the slowness)")
import concurrent.futures as cf
sessions = [earthaccess.get_requests_https_session() for _ in range(8)]


def fetch(i, off, n):
    t = time.time()
    r = sessions[i].get(URL, headers={"Range": f"bytes={off}-{off + n - 1}"}, stream=True, timeout=(30, 180))
    got = sum(len(b) for b in r.iter_content(256 * 1024))
    return got, time.time() - t


with cf.ThreadPoolExecutor(8) as ex:                      # warm every session (login redirect) outside the timing
    list(ex.map(lambda i: fetch(i, 0, 4096), range(8)))
for n in (1, 2, 4, 8):
    t = time.time()
    try:
        with cf.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda i: fetch(i, 100_000_000 + i * 40_000_000, 4 << 20), range(n)))
        dt, tot = time.time() - t, sum(g for g, _ in res)
        P(f"{n} at once, 4 MB each: wall {dt:5.1f}s  total {tot / 1e6:5.1f} MB  aggregate {tot / 1e6 / dt:5.2f} MB/s  slowest {max(d for _, d in res):5.1f}s")
    except Exception as ex:
        P(f"{n} at once FAILED: {type(ex).__name__}: {ex}")

fs = earthaccess.get_fsspec_https_session()


def open_traced(block):
    f = fs.open(URL, mode="rb", cache_type="blockcache", block_size=block)
    fetch = f.cache.fetcher

    def traced(a, b):
        t = time.time()
        P(f"    GET {a / 1e6:8.1f} .. {b / 1e6:8.1f} MB  ({(b - a) / 1e6:.1f} MB) started")
        d = fetch(a, b)
        P(f"        done in {time.time() - t:.1f}s")
        return d
    f.cache.fetcher = traced
    return f


for block in (8 << 20, 1 << 20):
    P(f"\n== 2. h5py open + first reads, block size {block >> 20} MB, page buffer 16 MB")
    f = open_traced(block)
    t = time.time()
    h = h5py.File(f, "r", page_buf_size=16 << 20, rdcc_nbytes=4 << 20)
    P(f"open took {time.time() - t:.1f}s")
    try:
        P("HDF5 file-space page size:", h.id.get_create_plist().get_file_space_page_size())
    except Exception as ex:
        P("page size not readable:", type(ex).__name__, ex)
    for name in ("unwrappedInterferogram/HH", "wrappedInterferogram/HH"):
        t = time.time()
        g = h[GRID + "/" + name]
        P(f"{name}: opened group, {len(list(g.keys()))} children, {time.time() - t:.1f}s")
    t = time.time()
    d = h[GRID + "/unwrappedInterferogram/HH/unwrappedPhase"]
    P(f"unwrappedPhase shape {d.shape} chunks {d.chunks} compression {d.compression} dtype {d.dtype}  ({time.time() - t:.1f}s)")
    t = time.time()
    ny, nx = d.shape
    a = d[ny // 2:ny // 2 + 64, nx // 2:nx // 2 + 64]
    P(f"read a 64x64 piece in {time.time() - t:.1f}s")
    h.close()
    f.close()
P("\nfinished")
