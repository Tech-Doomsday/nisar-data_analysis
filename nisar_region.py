#!/usr/bin/env python3
"""
nisar_region.py - get NISAR data for a lat/lon box, the way ASF Vertex does.

What it does (nothing is processed, nothing is plotted):
  1. SEARCH   all NISAR granules whose footprint touches your box  -> printed table + CSV
  2. DOWNLOAD (optional, --download DIR) the full granule files exactly as ASF serves them
              (live progress, stall detection, automatic retry and resume)
  3. CLIP     (optional, --clip DIR) stream each HDF5 file and save only the raw pixels inside
              your box (every gridded layer in the file, untouched values) to one .npz per granule

Install:
  pip install asf_search requests              (search + download)
  pip install earthaccess h5py fsspec aiohttp pyproj numpy python-dotenv   (only for --clip)

Login (needed only for --download / --clip): NASA Earthdata account, from a .env file
  EARTHDATA_USERNAME=...
  EARTHDATA_PASSWORD=...
or real environment variables. Searching needs no login.
Your Earthdata profile must have the "Alaska Satellite Facility Data Access" application
authorized, otherwise ASF answers with a web page instead of the file.

Examples:
  # what exists over a box? (W S E N in degrees)
  python nisar_region.py --bbox -118.5 33.9 -118.0 34.2 --start 2026-06-17 --end 2026-10-19

  # try ONE file first
  python nisar_region.py --bbox ... --level GUNW --max 1 --download ./nisar_files

  # download everything found (older duplicate processing versions are dropped unless --keep-all)
  python nisar_region.py --bbox ... --level GCOV --download ./nisar_files

  # keep only the box (smaller, but slower per file because it is read over the network)
  python nisar_region.py --bbox ... --level GUNW --clip ./nisar_clips --max 3

Downloads write to NAME.part and are renamed only when complete, so Ctrl+C is safe:
run the same command again and it resumes where it stopped.

Status: the downloader was tested against a local test server (progress, mid-file connection
        drop with resume, HTML-instead-of-file detection). It has not been run against the live
        ASF servers, so check the first real run.
"""
import argparse
import csv
import getpass
import os
import re
import sys
import time

H5PY_PARAMS = {"page_buf_size": 16 * 1024 * 1024, "rdcc_nbytes": 4 * 1024 * 1024}
FSSPEC_PARAMS = {"cache_type": "blockcache", "block_size": 8 * 1024 * 1024}

CONNECT_TIMEOUT = 20   # seconds to establish a connection
STALL_TIMEOUT = 60     # seconds with no data before a download counts as stalled and is retried
RETRIES = 6


# ------------------------------------------------------------------ helpers
def bbox_wkt(w, s, e, n):
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def load_env():
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    for path in (here, find_dotenv(usecwd=True)):
        if path and os.path.exists(path):
            load_dotenv(path, override=False)


def credentials():
    load_env()
    user, pw = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    if not (user and pw):
        print("EARTHDATA_USERNAME / EARTHDATA_PASSWORD not found, asking.", file=sys.stderr)
        user = input("Earthdata username: ")
        pw = getpass.getpass("Earthdata password: ")
    return user, pw


def mb(n):
    try:
        return f"{float(n) / 1e6:,.0f}"
    except (TypeError, ValueError):
        return "?"


def size_bytes(p):
    """Size in bytes from the search metadata, or None if ASF did not report one."""
    for key, scale in (("bytes", 1), ("sizeMB", 1e6)):
        v = p.get(key)
        if isinstance(v, dict):            # some products give {filename: {"bytes": n}}
            try:
                v = sum(float(x.get("bytes", 0)) for x in v.values())
            except (AttributeError, TypeError, ValueError):
                v = None
        try:
            if v is not None and float(v) > 0:
                return float(v) * scale
        except (TypeError, ValueError):
            pass
    return None


def atomic_savez(path, **arrays):
    """Write .npz so a half-written file never appears under the final name."""
    import numpy as np
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ 1. search
def search(bbox, start, end, level, maturity, max_n):
    import asf_search as asf
    kw = {"dataset": "NISAR", "intersectsWith": bbox_wkt(*bbox), "maxResults": max_n}
    if start:
        kw["start"] = start
    if end:
        kw["end"] = end
    if level:
        kw["processingLevel"] = [v.strip() for v in level.split(",")]
    if maturity:
        kw["dataMaturity"] = maturity
    return asf.search(**kw)


_VERSION_TAIL = re.compile(r"_([A-Z])(\d{5})(_[A-Z]_[A-Z]_[A-Z]_\d{3})$")


def drop_older_versions(results):
    """Same scene processed more than once (e.g. ..._X05010_... and ..._P05023_...):
    keep the one with the highest version number. Returns (kept, n_dropped)."""
    best = {}
    for i, r in enumerate(results):
        fid = str(r.properties.get("fileID") or r.properties.get("fileName") or i)
        fid = re.sub(r"\.h5$", "", fid)
        m = _VERSION_TAIL.search(fid)
        key, ver = (fid[:m.start()] + m.group(3), int(m.group(2))) if m else (fid, 0)
        if key not in best or ver > best[key][0]:
            best[key] = (ver, i)
    keep = sorted(i for _, i in best.values())
    return [results[i] for i in keep], len(results) - len(keep)


def row(r):
    p = r.properties
    return {
        "fileID": p.get("fileID"),
        "level": p.get("processingLevel"),
        "start": p.get("startTime"),
        "stop": p.get("stopTime"),
        "path": p.get("pathNumber"),
        "frame": p.get("frameNumber"),
        "direction": p.get("flightDirection"),
        "polarization": p.get("polarization"),
        "size_mb": mb(size_bytes(p)),
        "url": p.get("url"),
    }


def report(results, csv_path):
    rows = [row(r) for r in results]
    if not rows:
        print("No granules touch this box with these filters.")
        return rows
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total, unknown = 0.0, 0
    for x in rows:
        try:
            total += float(x["size_mb"].replace(",", ""))
        except ValueError:
            unknown += 1
        print(f"{str(x['start'])[:19]}  {str(x['level']):6}  {str(x['size_mb']):>7} MB  {x['fileID']}")
    if unknown == len(rows):
        print(f"\n{len(rows)} granule(s). ASF did not report file sizes for these; "
              f"each size is shown when its download starts.")
    elif unknown:
        print(f"\n{len(rows)} granule(s), at least {total / 1000:,.1f} GB ({unknown} with unknown size).")
    else:
        print(f"\n{len(rows)} granule(s), about {total / 1000:,.1f} GB if you download them all.")
    print(f"List saved to {csv_path}")
    return rows


# ------------------------------------------------------------------ 2. download
class DownloadError(Exception):
    pass


def fetch(session, url, dest, retries=RETRIES):
    """Stream url to dest with progress, stall timeout, retry and resume (dest + '.part')."""
    import requests
    part = dest + ".part"
    last_err = None
    for attempt in range(1, retries + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        note = f" (resuming at {mb(done)} MB)" if done else ""
        print(f"  connecting{note} ...", file=sys.stderr, flush=True)
        try:
            with session.get(url, stream=True, headers=headers,
                             timeout=(CONNECT_TIMEOUT, STALL_TIMEOUT)) as resp:
                code = resp.status_code
                if code == 416:                       # asked for bytes past the end
                    m = re.search(r"/(\d+)$", resp.headers.get("Content-Range", ""))
                    if m and int(m.group(1)) == done:
                        os.replace(part, dest)
                        return
                    os.remove(part)
                    continue
                if code in (401, 403):
                    raise DownloadError(
                        f"HTTP {code}: not authorized. Check your Earthdata username/password and that "
                        f"'Alaska Satellite Facility Data Access' is authorized in your Earthdata profile.")
                if 400 <= code < 500 and code not in (408, 429):
                    raise DownloadError(f"HTTP {code} from server for {url}")
                resp.raise_for_status()               # 5xx / 408 / 429 -> retried below
                if "text/html" in resp.headers.get("Content-Type", "").lower():
                    raise DownloadError(
                        "the server sent a web page instead of the file. This usually means the Earthdata "
                        "login was not accepted for this download: check the credentials and that "
                        "'Alaska Satellite Facility Data Access' is authorized in your Earthdata profile.")
                if code == 206:
                    m = re.search(r"/(\d+)$", resp.headers.get("Content-Range", ""))
                    total, mode = (int(m.group(1)) if m else None), "ab"
                else:                                 # 200: server ignored Range, start over
                    cl = resp.headers.get("Content-Length")
                    total, mode, done = (int(cl) if cl else None), "wb", 0
                print(f"  size: {mb(total)} MB", file=sys.stderr, flush=True)
                got, t0, last = done, time.time(), 0.0
                start_got = got
                with open(part, mode) as f:
                    for block in resp.iter_content(1024 * 1024):
                        if not block:
                            continue
                        f.write(block)
                        got += len(block)
                        now = time.time()
                        if now - last >= 1.0:
                            rate = (got - start_got) / max(now - t0, 1e-6) / 1e6
                            pct = f" ({100 * got / total:.0f}%)" if total else ""
                            print(f"\r  {mb(got)} / {mb(total)} MB{pct}  {rate:.1f} MB/s   ",
                                  end="", file=sys.stderr, flush=True)
                            last = now
                print(file=sys.stderr)
                if total and got < total:
                    raise requests.exceptions.ChunkedEncodingError("connection closed before the file was complete")
                os.replace(part, dest)
                return
        except requests.RequestException as ex:
            last_err = ex
            print(f"\n  interrupted ({type(ex).__name__}); retry {attempt}/{retries}", file=sys.stderr, flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise DownloadError(f"gave up after {retries} attempts: {last_err}")


def download_all(results, out_dir):
    import asf_search as asf
    os.makedirs(out_dir, exist_ok=True)
    user, pw = credentials()
    print("logging in to Earthdata ...", file=sys.stderr, flush=True)
    session = asf.ASFSession().auth_with_creds(user, pw)
    print("logged in", file=sys.stderr, flush=True)
    failed = []
    for i, r in enumerate(results, 1):
        p = r.properties
        url = p.get("url")
        name = p.get("fileName") or os.path.basename(str(url))
        dest = os.path.join(out_dir, name)
        if os.path.exists(dest):                      # only complete files ever get the final name
            print(f"[{i}/{len(results)}] have {name}, skipping")
            continue
        print(f"[{i}/{len(results)}] downloading {name}", flush=True)
        try:
            fetch(session, url, dest)
            print(f"  saved {dest}")
        except KeyboardInterrupt:
            print(f"\nstopped. Partial file kept as {dest}.part; run again to resume.", file=sys.stderr)
            raise SystemExit(130)
        except DownloadError as ex:
            print(f"  FAILED: {ex}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} download(s) failed:", file=sys.stderr)
        for n in failed:
            print(f"  {n}", file=sys.stderr)


# ------------------------------------------------------------------ 3. clip
def find_epsg(h5, grp):
    import numpy as np
    node = grp
    while True:
        if "projection" in node:
            p = node["projection"]
            for key in ("epsg_code", "EPSG"):
                if key in p.attrs:
                    return int(np.asarray(p.attrs[key]).item())
            try:
                return int(np.asarray(p[()]).item())
            except Exception:
                pass
        if node.name == "/":
            break
        node = node.parent
    raise KeyError("no projection / EPSG code found above " + grp.name)


def clip_h5(fileobj, bbox):
    """Return {name: array} with every gridded layer cut to the box (raw values)."""
    import h5py
    import numpy as np
    from pyproj import Transformer

    w, s, e, n = bbox
    out, seen, grids = {}, set(), []
    with h5py.File(fileobj, "r", **H5PY_PARAMS) as h5:
        def find_grids(name, obj):
            if isinstance(obj, h5py.Group) and "xCoordinates" in obj and "yCoordinates" in obj:
                grids.append(obj)
        h5.visititems(find_grids)
        if not grids:
            raise KeyError("no xCoordinates/yCoordinates found; is this a gridded (L2/L3) product?")
        for g in grids:
            x, y = g["xCoordinates"][:], g["yCoordinates"][:]
            epsg = find_epsg(h5, g)
            tr = Transformer.from_crs(4326, epsg, always_xy=True)
            mx, my = (w + e) / 2, (s + n) / 2           # corners + edge midpoints of the box
            px, py = tr.transform([w, e, w, e, mx, mx, w, e], [s, s, n, n, s, n, my, my])
            dx = abs(float(x[1] - x[0])) if len(x) > 1 else 0.0    # one-pixel margin so the box
            dy = abs(float(y[1] - y[0])) if len(y) > 1 else 0.0    # edges are fully covered
            xi = np.nonzero((x >= min(px) - dx) & (x <= max(px) + dx))[0]
            yi = np.nonzero((y >= min(py) - dy) & (y <= max(py) + dy))[0]
            if xi.size == 0 or yi.size == 0:
                continue
            xs, ys = slice(int(xi[0]), int(xi[-1]) + 1), slice(int(yi[0]), int(yi[-1]) + 1)
            base = g.name.strip("/").replace("/", "__")
            out[base + "__xCoordinates"], out[base + "__yCoordinates"] = x[xs], y[ys]
            out[base + "__epsg"] = np.array(epsg)
            todo = []

            def find_layers(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.shape == (len(y), len(x)):
                    todo.append(obj)
            g.visititems(find_layers)
            for d in todo:
                if d.name in seen:
                    continue
                seen.add(d.name)
                print(f"    {d.name.strip('/')}  {d.dtype}", file=sys.stderr, flush=True)
                out[d.name.strip("/").replace("/", "__")] = d[ys, xs]
    if not seen:
        raise ValueError("the box does not overlap this file's grid")
    return out


def clip_all(results, bbox, out_dir):
    import earthaccess
    os.makedirs(out_dir, exist_ok=True)
    load_env()
    user, pw = credentials()
    os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"] = user, pw
    earthaccess.login(strategy="environment")
    fs = earthaccess.get_fsspec_https_session()
    for r in results:
        p = r.properties
        url = p.get("url")
        name = p.get("fileName") or os.path.basename(str(url))
        dest = os.path.join(out_dir, os.path.splitext(name)[0] + "_clip.npz")
        if os.path.exists(dest):
            print(f"have {os.path.basename(dest)}, skipping")
            continue
        print(f"- {name} ({mb(size_bytes(p))} MB file, reading only the box)", file=sys.stderr, flush=True)
        try:
            with fs.open(url, mode="rb", **FSSPEC_PARAMS) as f:
                layers = clip_h5(f, bbox)
            atomic_savez(dest, bbox=list(bbox), source=name, **layers)
            print(f"  saved {dest}")
        except Exception as ex:
            print(f"  skipped: {ex}", file=sys.stderr)


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"),
                    help="west south east north, degrees")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--level", help="processing level(s), comma separated: RSLC,GUNW,GCOV,GSLC ...")
    ap.add_argument("--maturity", choices=["PROVISIONAL", "BETA"], help="calibrated or pre-calibration")
    ap.add_argument("--max", type=int, default=250, help="max granules to list (default 250)")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep every processing version of a scene (default: newest version only)")
    ap.add_argument("--csv", help="where to save the list (default nisar_results.csv)")
    ap.add_argument("--download", metavar="DIR", help="download the full granule files to DIR")
    ap.add_argument("--clip", metavar="DIR", help="save only the box, raw values, to DIR (one .npz per granule)")
    a = ap.parse_args(argv)

    w, s, e, n = a.bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        sys.exit("bbox must be W S E N with W < E and S < N (boxes crossing +-180 are not supported)")

    results = search(a.bbox, a.start, a.end, a.level, a.maturity, a.max)
    found = len(results)
    if not a.keep_all:
        results, dropped = drop_older_versions(results)
        if dropped:
            print(f"Dropped {dropped} older duplicate processing version(s); use --keep-all to keep them.")
    rows = report(results, a.csv or "nisar_results.csv")
    if not rows:
        return
    if found == a.max:
        print(f"NOTE: hit --max {a.max}; there may be more. Raise --max or narrow the dates.")
    if a.download:
        download_all(results, a.download)
    if a.clip:
        clip_all(results, a.bbox, a.clip)


if __name__ == "__main__":
    main()