#!/usr/bin/env python3
"""
nisar_fast.py - parallel companion to nisar_region.py (keep both files in the same folder).

Differences from nisar_region.py:
  * --workers N   : N files at a time (downloads: threads, clips: separate processes,
                    because h5py serialises HDF5 calls inside one process)
  * clip also keeps the 3-D lookup cubes (tides, troposphere, geometry) and their height axis
  * --layers REGEX: only read datasets whose HDF5 path matches (big speed-up, e.g. skip 20 m wrapped data)
  * clip retries transient network errors

Examples:
  python nisar_fast.py --bbox -118.5 33.9 -118.0 34.2 --level GUNW --clip ./clips --workers 6
  python nisar_fast.py --bbox ... --level GUNW --download ./files --workers 4
"""
import argparse
import multiprocessing as mp
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import nisar_region as nr


# ------------------------------------------------------------------ clip (one file)
def clip_h5(fileobj, bbox, layer_re=None, verbose=False):
    """{name: array}: every gridded layer (2-D, or 3-D cube with trailing y,x) cut to the box."""
    import h5py
    import numpy as np
    from pyproj import Transformer

    w, s, e, n = bbox
    pat = re.compile(layer_re) if layer_re else None
    out, seen, grids, fallback = {}, set(), [], None

    with h5py.File(fileobj, "r", **nr.H5PY_PARAMS) as h5:
        def find_grids(name, obj):
            if isinstance(obj, h5py.Group) and "xCoordinates" in obj and "yCoordinates" in obj:
                grids.append(obj)
        h5.visititems(find_grids)
        if not grids:
            raise KeyError("no xCoordinates/yCoordinates found; is this a gridded (L2/L3) product?")

        for g in grids:
            x, y = g["xCoordinates"][:], g["yCoordinates"][:]
            try:
                epsg = nr.find_epsg(h5, g)
                fallback = fallback or epsg
            except KeyError:
                epsg = fallback               # e.g. radarGrid without its own projection
            if epsg is None or len(x) == 0 or len(y) == 0:
                continue
            ny, nx = len(y), len(x)
            layers = []

            def pick(name, obj):
                if (isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3)
                        and obj.shape[-2:] == (ny, nx) and obj.name not in seen
                        and not (pat and not pat.search(obj.name))):
                    layers.append(obj)
            g.visititems(pick)
            if not layers:
                continue

            tr = Transformer.from_crs(4326, epsg, always_xy=True)
            mx, my = (w + e) / 2, (s + n) / 2
            px, py = tr.transform([w, e, w, e, mx, mx, w, e], [s, s, n, n, s, n, my, my])
            dx = abs(float(x[1] - x[0])) if nx > 1 else 0.0   # one-cell margin (also gives the
            dy = abs(float(y[1] - y[0])) if ny > 1 else 0.0   # bracketing nodes for interpolation)
            xi = np.nonzero((x >= min(px) - dx) & (x <= max(px) + dx))[0]
            yi = np.nonzero((y >= min(py) - dy) & (y <= max(py) + dy))[0]
            if xi.size == 0 or yi.size == 0:
                continue
            xs, ys = slice(int(xi[0]), int(xi[-1]) + 1), slice(int(yi[0]), int(yi[-1]) + 1)
            base = g.name.strip("/").replace("/", "__")
            out[base + "__xCoordinates"], out[base + "__yCoordinates"] = x[xs], y[ys]
            out[base + "__epsg"] = np.array(epsg)
            if any(d.ndim == 3 for d in layers):      # keep the cube's height axis etc.
                for k, v in g.items():
                    if (isinstance(v, h5py.Dataset) and v.ndim == 1
                            and k not in ("xCoordinates", "yCoordinates") and v.size <= 4096):
                        out[base + "__" + k] = v[:]
            for d in layers:
                seen.add(d.name)
                if verbose:
                    print(f"    {d.name.strip('/')}  {d.dtype} {d.shape}", file=sys.stderr, flush=True)
                idx = (slice(None),) * (d.ndim - 2) + (ys, xs)
                out[d.name.strip("/").replace("/", "__")] = d[idx]
    if not seen:
        raise ValueError("the box does not overlap this file's grid (or --layers matched nothing)")
    return out


# ------------------------------------------------------------------ clip (parallel)
_FS = None


def _fs():
    global _FS
    if _FS is None:
        import earthaccess
        earthaccess.login(strategy="environment")
        _FS = earthaccess.get_fsspec_https_session()
    return _FS


def _init_worker():
    time.sleep(random.uniform(0, 3))     # avoid N simultaneous Earthdata logins
    _fs()


def clip_one(job):
    url, name, bbox, out_dir, layer_re = job
    dest = os.path.join(out_dir, os.path.splitext(name)[0] + "_clip.npz")
    if os.path.exists(dest):
        return name, "have", 0.0
    t0, err = time.time(), None
    for attempt in range(1, 4):
        try:
            with _fs().open(url, mode="rb", **nr.FSSPEC_PARAMS) as f:
                layers = clip_h5(f, bbox, layer_re)
            nr.atomic_savez(dest, bbox=list(bbox), source=name, **layers)
            return name, "ok", time.time() - t0
        except (ValueError, KeyError) as ex:           # permanent: no overlap / wrong product
            return name, f"skipped: {ex}", time.time() - t0
        except Exception as ex:                        # network etc: retry
            err = f"{type(ex).__name__}: {ex}"
            time.sleep(3 * attempt)
    return name, f"FAILED: {err}", time.time() - t0


def clip_all(results, bbox, out_dir, workers=4, layer_re=None):
    os.makedirs(out_dir, exist_ok=True)
    user, pw = nr.credentials()
    os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"] = user, pw
    jobs = []
    for r in results:
        p = r.properties
        url = p.get("url")
        jobs.append((url, p.get("fileName") or os.path.basename(str(url)), tuple(bbox), out_dir, layer_re))
    workers = max(1, min(workers, len(jobs)))
    ctx = mp.get_context("spawn")
    print(f"clipping {len(jobs)} file(s) with {workers} process(es) ...", flush=True)
    t0, n_done = time.time(), 0
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as ex:
        futs = [ex.submit(clip_one, j) for j in jobs]
        for fut in as_completed(futs):
            n_done += 1
            name, status, secs = fut.result()
            print(f"[{n_done}/{len(jobs)}] {status:8.60} {name}  ({secs:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ download (parallel)
def dl_one(session, url, dest, retries=6):
    import requests
    part, last = dest + ".part", None
    for attempt in range(1, retries + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            with session.get(url, stream=True, headers={"Range": f"bytes={done}-"} if done else {},
                             timeout=(nr.CONNECT_TIMEOUT, nr.STALL_TIMEOUT)) as r:
                if r.status_code == 416:
                    m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
                    if m and int(m.group(1)) == done:
                        os.replace(part, dest)
                        return
                    os.remove(part)
                    continue
                if r.status_code in (401, 403):
                    raise nr.DownloadError(f"HTTP {r.status_code}: not authorized (check Earthdata login "
                                           f"and the 'Alaska Satellite Facility Data Access' app)")
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    raise nr.DownloadError(f"HTTP {r.status_code}")
                r.raise_for_status()
                if "text/html" in r.headers.get("Content-Type", "").lower():
                    raise nr.DownloadError("got a web page instead of the file (login not accepted)")
                if r.status_code == 206:
                    m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
                    total, mode = (int(m.group(1)) if m else None), "ab"
                else:
                    cl = r.headers.get("Content-Length")
                    total, mode = (int(cl) if cl else None), "wb"
                got = done if mode == "ab" else 0
                with open(part, mode) as f:
                    for block in r.iter_content(4 * 1024 * 1024):
                        f.write(block)
                        got += len(block)
                if total and got < total:
                    raise requests.exceptions.ChunkedEncodingError("closed early")
                os.replace(part, dest)
                return
        except requests.RequestException as ex:
            last = ex
            time.sleep(min(30, 2 ** attempt))
    raise nr.DownloadError(f"gave up after {retries} attempts: {last}")


def download_all(results, out_dir, workers=4):
    import asf_search as asf
    os.makedirs(out_dir, exist_ok=True)
    user, pw = nr.credentials()
    session = asf.ASFSession().auth_with_creds(user, pw)
    todo = []
    for r in results:
        p = r.properties
        name = p.get("fileName") or os.path.basename(str(p.get("url")))
        dest = os.path.join(out_dir, name)
        if not os.path.exists(dest):
            todo.append((p.get("url"), dest, name))
    print(f"downloading {len(todo)} file(s) with {workers} thread(s) ({len(results) - len(todo)} already here)", flush=True)
    failed, t0 = [], time.time()
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(dl_one, session, u, d): (n, d) for u, d, n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            n, d = futs[fut]
            try:
                fut.result()
                print(f"[{i}/{len(todo)}] saved {n}  ({os.path.getsize(d) / 1e6:,.0f} MB)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] FAILED {n}: {e}", file=sys.stderr, flush=True)
                failed.append(n)
    print(f"done in {time.time() - t0:.0f}s" + (f", {len(failed)} failed (rerun to resume)" if failed else ""))


# ------------------------------------------------------------------ entry points
def run(bbox, start=None, end=None, level=None, maturity=None, max_n=250, keep_all=False,
        csv_path="nisar_results.csv", download=None, clip=None, workers=4, layers=None):
    results = nr.search(bbox, start, end, level, maturity, max_n)
    if not keep_all:
        results, dropped = nr.drop_older_versions(results)
        if dropped:
            print(f"Dropped {dropped} older duplicate processing version(s).")
    if not nr.report(results, csv_path):
        return results
    if download:
        download_all(results, download, workers)
    if clip:
        clip_all(results, bbox, clip, workers, layers)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--level"); ap.add_argument("--maturity", choices=["PROVISIONAL", "BETA"])
    ap.add_argument("--max", type=int, default=250)
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--csv", default="nisar_results.csv")
    ap.add_argument("--download", metavar="DIR"); ap.add_argument("--clip", metavar="DIR")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 2) * 2))
    ap.add_argument("--layers", help="regex on HDF5 dataset path, e.g. 'unwrapped|ionosphere|radarGrid'")
    a = ap.parse_args(argv)
    w, s, e, n = a.bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        sys.exit("bbox must be W S E N with W < E and S < N")
    run(a.bbox, a.start, a.end, a.level, a.maturity, a.max, a.keep_all, a.csv,
        a.download, a.clip, a.workers, a.layers)


if __name__ == "__main__":
    main()
