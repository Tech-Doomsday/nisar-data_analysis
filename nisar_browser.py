#!/usr/bin/env python3
"""
nisar_browser.py - look before you download.

Search NISAR granules over a box, see them on a zoomable timeline (every interferogram is a bar
from its reference date to its secondary date), filter by day of month / UTC time / track, click one
to read details straight out of the remote HDF5 file (only a few MB are fetched, never the 2 GB),
then clip only the box for exactly the granules you kept.

Select a granule, press "Read file metadata", tick layers under "View data" and load them: only the box
is read from the remote file, for the ticked layers of that one granule, and shown in a viewer.

Keep next to: nisar_region.py, nisar_fast.py, nisar_browser.html

  python nisar_browser.py --bbox -118.5 33.9 -118.0 34.2 --level GUNW --out ./clips

Opens http://127.0.0.1:8765 . Login comes from .env / environment (same as nisar_region.py);
use --no-login to browse without it (search works, file details and clipping need it).
Needs: asf_search earthaccess h5py fsspec aiohttp pyproj numpy python-dotenv
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import threading
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import nisar_fast as nf
import nisar_region as nr

HERE = os.path.dirname(os.path.abspath(__file__))
C_LIGHT = 299_792_458.0
TS = re.compile(r"\d{8}T\d{6}")
FN = re.compile(r"_(\d{3})_(\d{3})_([AD])_(\d{3})_(\d{3})_")      # refCycle track dir frame secCycle
# datasets worth reading the value of (everything else is only listed by name/shape)
KEY = re.compile(r"frequency|wavelength|bandwidth|ionosph|tropo|tide|baseline|lookDirection|"
                 r"orbitPassDirection|polariz|looks|unwrap|filter|algorithm|version|spacing|"
                 r"referenceTerrain|processingType|isGeocoded|mask|dem|projection|epsg", re.I)
KEY_PATH = re.compile(r"/identification/|/processingInformation/parameters/", re.I)

S = {"granules": {}, "bbox": None, "defaults": {}, "out": "./nisar_clips", "fs": None,
     "jobs": {}, "lj": {}, "cache_mb": 768, "login": True}
META_LOCK = threading.Lock()


# ------------------------------------------------------------------ granule description
def _iso(t):
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:15]}Z"


def _pol(p):
    v = p.get("mainBandPolarization")          # ASF leaves "polarization" empty for NISAR; this is the real field
    if isinstance(v, (list, tuple)):
        v = "/".join(str(x) for x in v)
    return v or None


def describe(i, r):
    p = r.properties
    fid = re.sub(r"\.h5$", "", str(p.get("fileID") or p.get("fileName") or i))
    ts = [_iso(t) for t in TS.findall(fid)]
    m = FN.search(fid)
    if len(ts) >= 4:                                   # interferogram: reference pair + secondary pair
        ref, sec = ts[0], ts[2]
        t0, t1 = min(ref, sec), max(ref, sec)
        base = abs((time_ms(sec) - time_ms(ref)) / 86400e3)
    else:
        ref = sec = None
        t0 = str(p.get("startTime") or (ts[0] if ts else ""))
        t1 = str(p.get("stopTime") or t0)
        ref = t0
        base = None
    sb = size_b = nr.size_bytes(p)
    props = json.loads(json.dumps(dict(p), default=str))
    return {
        "id": fid, "name": p.get("fileName") or os.path.basename(str(p.get("url"))),
        "level": p.get("processingLevel"), "t0": t0, "t1": t1, "ref": ref, "sec": sec,
        "baseline_days": base, "size_mb": round(sb / 1e6) if sb else None,
        "dir": (m.group(3) if m else str(p.get("flightDirection") or "?")[:1]).upper(),
        "track": int(m.group(2)) if m else p.get("pathNumber"),
        "frame": int(m.group(4)) if m else p.get("frameNumber"),
        "ref_cycle": int(m.group(1)) if m else None, "sec_cycle": int(m.group(5)) if m else None,
        "pol": p.get("polarization") or _pol(p), "maturity": p.get("dataMaturity"), "url": p.get("url"),
        "geometry": getattr(r, "geometry", None), "props": props,
    }


def time_ms(iso):
    import calendar
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")) * 1000.0


def do_search(q):
    bbox = tuple(float(v) for v in q["bbox"])
    results = nr.search(bbox, q.get("start") or None, q.get("end") or None, q.get("level") or None,
                        q.get("maturity") or None, int(q.get("max") or 1000))
    found = len(results)
    dropped = 0
    if not q.get("keep_all"):
        results, dropped = nr.drop_older_versions(results)
    gs = [describe(i, r) for i, r in enumerate(results)]
    S["granules"] = {g["id"]: g for g in gs}
    S["bbox"] = bbox
    return {"granules": gs, "found": found, "dropped": dropped, "capped": found >= int(q.get("max") or 1000)}


# ------------------------------------------------------------------ remote file details
def get_fs():
    if S["fs"] is None:
        import earthaccess
        earthaccess.login(strategy="environment")
        S["fs"] = earthaccess.get_fsspec_https_session()
    return S["fs"]


def _val(v):
    import numpy as np
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")[:200]
    if isinstance(v, np.ndarray):
        return [_val(x) for x in v.ravel().tolist()]
    if isinstance(v, np.generic):
        return _val(v.item())
    if isinstance(v, str):
        return v[:200]
    return v


def read_meta(gid):
    import h5py
    g = S["granules"][gid]
    cache = os.path.join(S["out"], ".meta", gid + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    t0 = time.time()
    params, layers = {}, []
    with META_LOCK:
        with get_fs().open(g["url"], mode="rb", cache_type="blockcache", block_size=2 * 1024 * 1024) as f, \
                h5py.File(f, "r", page_buf_size=4 * 1024 * 1024) as h5:
            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                path, n = "/" + name, 1
                for d in obj.shape:
                    n *= d
                if n <= 16 and len(params) < 500 and (KEY.search(path) or KEY_PATH.search(path)):
                    try:
                        params[path] = _val(obj[()])
                    except Exception as ex:
                        params[path] = f"<unreadable: {ex}>"
                else:
                    layers.append({"path": path, "shape": list(obj.shape), "dtype": str(obj.dtype)})
            h5.visititems(visit)
    freq = next((v for k, v in params.items() if re.search(r"(processed)?centerFrequency$", k, re.I)
                 and isinstance(v, (int, float)) and v > 1e8), None)
    wl = next((v for k, v in params.items() if k.lower().endswith("wavelength")
               and isinstance(v, (int, float))), None)
    wl_src = "metadata" if wl else ("c / centerFrequency" if freq else None)
    if wl is None and freq:
        wl = C_LIGHT / freq

    def has(rx):
        return [x["path"] for x in layers if re.search(rx, x["path"], re.I)]
    meta = {"seconds": round(time.time() - t0, 1), "params": params, "layers": layers,
            "facts": {"frequency_hz": freq, "wavelength_m": wl, "wavelength_source": wl_src,
                      "ionosphere": has("ionosph"), "troposphere": has("tropo"), "tide": has("tide"),
                      "radar_grid": has("radarGrid")}}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(meta, open(cache, "w"))
    return meta


# ------------------------------------------------------------------ load jobs: inspect one granule, then read chosen layers of the box
import collections
import logging
import math
import traceback

LOG = logging.getLogger("nisar_browser")
LVL = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
MAXPX = 1_000_000                                   # per layer output; a bigger crop is decimated (every Nth pixel)
PRUNE = {"identification", "processingInformation", "qualityAssurance", "orbit", "attitude",
         "calibrationInformation"}                  # subtrees that hold no gridded rasters; skipped and logged
CATEGORIES = [                                      # first match wins; add a row here to add a kind of data
    ("components", "Connected components", r"connectedComponents$", "which pixels were unwrapped together"),
    ("unwrapped", "Unwrapped phase", r"unwrappedPhase$", "main deformation layer, radians"),
    ("wrapped", "Wrapped interferogram", r"(?<!un)wrappedInterferogram$", "complex fringes (shown as magnitude and phase)"),
    ("coherence", "Coherence", r"coherence", "phase quality, 0 to 1"),
    ("iono", "Ionosphere", r"ionosph", "phase screen estimate and its uncertainty"),
    ("offsets", "Pixel offsets", r"offset|correlationSurfacePeak", "sub-pixel shifts from cross-correlation"),
    ("mask", "Masks", r"mask", "valid-data and similar masks"),
    ("tropo", "Troposphere", r"tropo|hydrostatic|wetDelay", "lookup cubes, not removed from the phase"),
    ("tide", "Solid-earth tide", r"tide", "lookup cube, not removed from the phase"),
    ("baseline", "Baselines", r"baseline", "orbit separation"),
    ("geometry", "Geometry", r"incidenceAngle|losUnitVector|elevationAngle|alongTrackUnitVector|groundTrackVelocity|slantRange|heightAboveEllipsoid",
     "needed to turn line-of-sight motion into vertical or horizontal"),
    ("other", "Everything else", r"", "other gridded layers in the file"),
]
H5 = {"gid": None, "f": None, "h5": None, "job": None}   # the one remote file kept open, so the next layer is fast
CACHE = collections.OrderedDict()                   # (gid, bbox, path) -> {"items": [meta], "raws": [bytes], "bytes": n}
CACHE_LOCK = threading.Lock()


class Job:
    def __init__(self, gid, url, bbox):
        self.id, self.gid, self.url, self.bbox = str(int(time.time() * 1000)), gid, url, bbox
        self.lock, self.cancel = threading.Lock(), threading.Event()
        self.state, self.error, self.catalog = "inspecting", None, None
        self.steps, self.log, self.layers, self.grids = [], [], {}, {}
        self.total = self.done = 0

    def say(self, lvl, msg):
        with self.lock:
            self.log.append({"t": round(time.time(), 3), "lvl": lvl, "msg": msg})
        LOG.log(LVL[lvl], "[%s] %s", self.id, msg)

    def step(self, name):
        with self.lock:
            self.steps.append({"name": name, "status": "running", "t0": time.time(), "t1": None, "detail": ""})
            return len(self.steps) - 1

    def detail(self, i, text):
        with self.lock:
            self.steps[i]["detail"] = text

    def end(self, i, ok=True, detail=""):
        with self.lock:
            self.steps[i].update(status="ok" if ok else "error", t1=time.time(), detail=detail)


class Beat:
    """logs 'still working' every few seconds so a slow remote read is never silent"""
    def __init__(self, job, what, every=8):
        self.job, self.what, self.every, self.ev = job, what, every, threading.Event()

    def __enter__(self):
        self.t0 = time.time()
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while not self.ev.wait(self.every):
            rb = _remote()
            self.job.say("INFO", f"still working: {self.what} ({time.time() - self.t0:.0f}s"
                         + (f", {rb / 1e6:.0f} MB fetched from the server so far)" if rb is not None else ")"))

    def __exit__(self, *a):
        self.ev.set()


def _py(v):
    import numpy as np
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else None
    if isinstance(v, np.generic):
        v = v.item()
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def _close():
    for k in ("h5", "f"):
        try:
            if H5[k] is not None:
                H5[k].close()
        except Exception:
            pass
    H5.update(gid=None, f=None, h5=None)


def _trace(f):
    """log every request to the server (start and end with duration), into the terminal log and the job log"""
    c = getattr(f, "cache", None)
    fetch = getattr(c, "fetcher", None)
    if fetch is None:
        return

    def traced(a, b):
        say = H5["job"].say if H5["job"] else (lambda lvl, m: LOG.log(LVL[lvl], m))
        t = time.time()
        say("DEBUG", f"server request started: bytes {a:,}..{b:,} ({(b - a) / 1e6:.1f} MB)")
        d = fetch(a, b)
        dt = time.time() - t
        say("INFO", f"server request done: {(b - a) / 1e6:.1f} MB at offset {a / 1e6:.0f} MB in {dt:.1f}s ({(b - a) / 1e6 / max(dt, 1e-9):.2f} MB/s)")
        return d
    c.fetcher = traced


def _open(gid, url):
    import h5py
    if H5["gid"] != gid:
        _close()
        f = get_fs().open(url, mode="rb", **{**nr.FSSPEC_PARAMS, "block_size": 4 << 20})   # = the file's HDF5 page size (4 MiB)
        _trace(f)
        H5.update(gid=None, f=f, h5=None)           # file object visible early, so the heartbeat can report bytes
        H5.update(gid=gid, h5=h5py.File(f, "r", **nr.H5PY_PARAMS))
    return H5["h5"]


def _up(obj, name):
    """nearest group at or above obj that holds `name` (xCoordinates, projection)"""
    g = obj.parent
    while True:
        if name in g:
            return g
        if g.name == "/":
            raise ValueError(f"no {name} found above {obj.name}")
        g = g.parent


EPSG = {}


def _epsg(h5, grp, gid):
    """same as nisar_fast.clip_h5: the group's own projection, else the first grid's (e.g. radarGrid)"""
    import h5py
    try:
        return nr.find_epsg(h5, grp)
    except KeyError:
        if gid not in EPSG:
            found = []

            def visit(name, o):
                if not found and isinstance(o, h5py.Group) and "xCoordinates" in o:
                    try:
                        found.append(nr.find_epsg(h5, o))
                    except KeyError:
                        pass
            h5.visititems(visit)
            EPSG[gid] = found[0] if found else None
        if EPSG[gid] is None:
            raise ValueError("no projection found in this file")
        return EPSG[gid]


def _window(grp, bbox, epsg):
    """pixel window of the lon/lat box on this group's x/y grid, plus lon/lat of 3 corners of that window"""
    import numpy as np
    from pyproj import Transformer
    xs = grp["xCoordinates"][()].astype("float64")
    ys = grp["yCoordinates"][()].astype("float64")
    w, s, e, n = bbox
    t = np.linspace(0, 1, 21)
    lon = np.r_[w + (e - w) * t, w + (e - w) * t, np.full(21, w), np.full(21, e)]     # all four edges
    lat = np.r_[np.full(21, s), np.full(21, n), s + (n - s) * t, s + (n - s) * t]
    px, py = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    ix = np.where((xs >= px.min() - abs(dx)) & (xs <= px.max() + abs(dx)))[0]
    iy = np.where((ys >= py.min() - abs(dy)) & (ys <= py.max() + abs(dy)))[0]
    if not len(ix) or not len(iy):
        raise ValueError("the search box does not overlap this layer's grid")
    x0, x1, y0, y1 = int(ix.min()), int(ix.max()) + 1, int(iy.min()), int(iy.max()) + 1
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    L, R, T, B = xs[x0] - dx / 2, xs[x1 - 1] + dx / 2, ys[y0] - dy / 2, ys[y1 - 1] + dy / 2
    return x0, x1, y0, y1, [[float(c) for c in inv.transform(a, b)] for a, b in ((L, T), (R, T), (L, B))]


def categorize(path):
    return next(k for k, _, rx, _ in CATEGORIES if re.search(rx, path, re.I))


def _remote():
    """bytes requested from the server so far by the open file (None if this fsspec version does not count)"""
    c = getattr(H5["f"], "cache", None)
    v = getattr(c, "total_requested_bytes", None)
    if v is None and hasattr(c, "miss_count") and hasattr(c, "blocksize"):
        v = c.miss_count * c.blocksize              # older fsspec: blocks fetched x block size (approximate)
    return v


def inspect(job):
    import h5py
    try:
        bbox = job.bbox
        job.say("INFO", f"inspecting {job.url}")
        job.say("INFO", f"search box W S E N = {list(bbox)}; decimation limit {MAXPX:,} px per layer")
        i, t0 = job.step("Open remote file"), time.time()
        with Beat(job, "opening the remote file (login + header)"), META_LOCK:
            H5["job"] = job
            h5 = _open(job.gid, job.url)
        job.end(i, True, f"{time.time() - t0:.1f}s")
        rb = _remote()
        job.say("INFO", f"file open in {time.time() - t0:.1f}s, fetched " + (f"{rb / 1e6:.1f} MB" if rb is not None else "n/a"))
        job.say("DEBUG", f"fsspec params {({**nr.FSSPEC_PARAMS, 'block_size': 4 << 20})}, h5py params {nr.H5PY_PARAMS}")

        i, t0 = job.step("List gridded layers"), time.time()
        cache = os.path.join(S["out"], ".catalog", job.gid + ".json")
        rasters, st = [], {"groups": 0, "datasets": 0, "pruned": []}      # rasters: (path, shape, dtype, grid group)
        old = json.load(open(cache)) if os.path.exists(cache) else None
        if old and all(len(x) == 4 for x in old):               # (a cache written by an older version has no grid column: ignored)
            rasters = [(n, tuple(sh), dt, gr) for n, sh, dt, gr in old]
            d = f"{len(rasters)} 2-D/3-D layers, read from the disk cache {cache}"
        else:
            def walk(grp, grid=None):
                grid = grp.name if "xCoordinates" in grp else grid          # nearest group above that holds x/y coordinates
                st["groups"] += 1
                job.say("DEBUG", f"entering {grp.name} ({time.time() - t0:.1f}s)")
                for k in grp.keys():
                    if k in PRUNE:
                        st["pruned"].append(grp.name.rstrip("/") + "/" + k)
                        continue
                    o = grp.get(k)
                    if isinstance(o, h5py.Group):
                        walk(o, grid)
                    elif isinstance(o, h5py.Dataset):
                        st["datasets"] += 1
                        if o.ndim in (2, 3):
                            rasters.append((o.name, o.shape, str(o.dtype), grid))
                job.detail(i, f"{st['datasets']} datasets, {len(rasters)} rasters, {st['groups']} groups so far")
            with Beat(job, "listing the layers"), META_LOCK:
                H5["job"] = job
                roots = []
                sci = h5.get("science")             # only grids/ and metadata/radarGrid hold gridded rasters; nothing else is opened
                for band in (sci.keys() if sci is not None else []):
                    for prod in sci[band].keys():
                        for sub in ("grids", "metadata/radarGrid"):
                            if sub in sci[band][prod]:
                                roots.append(sci[band][prod][sub])
                if not roots:
                    job.say("WARN", "no /science/<band>/<product>/grids found; scanning the whole file (slow)")
                    roots = [h5]
                job.say("INFO", "looking in: " + ", ".join(r.name for r in roots))
                for r in roots:
                    walk(r)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump([[n, list(sh), dt, gr] for n, sh, dt, gr in rasters], open(cache, "w"))
            d = f"{st['datasets']} datasets in {st['groups']} groups, {len(rasters)} are 2-D/3-D"
        rb = _remote()
        job.say("INFO", d + f" ({time.time() - t0:.1f}s)" + (f", {rb / 1e6:.0f} MB fetched in total" if rb is not None else ""))
        job.end(i, True, d)

        import numpy as np
        layers = []
        for path, shape, dtype, grid in rasters:
            if grid is None:
                job.say("DEBUG", f"not on a map grid, ignored: {path} {list(shape)}")
                continue
            try:
                isz = np.dtype(dtype).itemsize
            except TypeError:
                isz = 0
            layers.append({"path": path, "shape": list(shape), "dtype": dtype, "grid": grid, "cat": categorize(path),
                           "full": int(np.prod(shape)) * isz})
        job.say("INFO", f"{len(layers)} loadable layers on {len({e['grid'] for e in layers})} grid(s); "
                        f"{len(rasters) - len(layers)} ignored (not on a map grid)")
        job.catalog = {"layers": layers, "bbox": list(bbox), "pruned": st["pruned"], "grids": sorted({e["grid"] for e in layers}),
                       "categories": [{"key": k, "label": l, "note": n} for k, l, _, n in CATEGORIES]}
        job.state = "ready"
        job.say("INFO", "ready: choose layers and press Load")
    except Exception as ex:
        _close()
        job.error = f"{type(ex).__name__}: {ex}"
        with job.lock:
            for s in job.steps:
                if s["status"] == "running":
                    s.update(status="error", t1=time.time())
        job.say("ERROR", "inspection failed: " + job.error)
        job.say("DEBUG", traceback.format_exc())
        job.state = "error"


def _range_get(url, off, n):
    """exactly bytes off..off+n of the file, nothing else"""
    import earthaccess
    if S.get("req") is None:
        S["req"] = earthaccess.get_requests_https_session()
    for attempt in (1, 2, 3):
        try:
            r = S["req"].get(url, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=(30, 300))
            if r.status_code != 206:
                raise IOError(f"HTTP {r.status_code} (expected 206)")
            if len(r.content) != n:
                raise IOError(f"got {len(r.content)} of {n} bytes")
            return r.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * attempt)


def _fast_chunks(job, obj, y0, y1, x0, x1):
    """rows y0:y1, cols x0:x1 of a chunked 2-D layer, fetching only the compressed bytes of the chunks the box touches
    (a plain h5py read drags in whole 4 MiB file pages around every chunk, ~4x more data on a slow link)"""
    import zlib
    import numpy as np
    pl = obj.id.get_create_plist()
    pipe = [pl.get_filter(i)[0] for i in range(pl.get_nfilters())]       # 1 deflate, 2 shuffle, 3 fletcher32
    if any(f not in (1, 2, 3) for f in pipe):
        raise ValueError(f"filter pipeline {pipe} is not just deflate/shuffle/fletcher32")
    cy, cx = obj.chunks
    isz = obj.dtype.itemsize
    out = np.full((y1 - y0, x1 - x0), obj.fillvalue, dtype=obj.dtype)
    jobs = [(r, c) for r in range(y0 // cy * cy, y1, cy) for c in range(x0 // cx * cx, x1, cx)]
    job.say("INFO", f"{obj.name}: {len(jobs)} chunk(s) of {cy}x{cx} touch the box; fetching only their bytes (filters {pipe})")
    t_all = tot = 0
    t_all = time.time()
    for k, (r, c) in enumerate(jobs, 1):
        info = obj.id.get_chunk_info_by_coord((r, c))
        if not info.size:
            job.say("DEBUG", f"chunk {r},{c} was never written: fill value")
            continue
        if info.filter_mask:
            raise ValueError(f"chunk {r},{c} was stored with some filters skipped (mask {info.filter_mask})")
        t = time.time()
        data = _range_get(job.url, info.byte_offset, info.size)
        dt = time.time() - t
        tot += info.size
        for f in reversed(pipe):
            if f == 3:
                data = data[:-4]
            elif f == 1:
                data = zlib.decompress(data)
            else:
                data = np.frombuffer(data, np.uint8).reshape(isz, -1).T.tobytes()
        ch = np.frombuffer(data, obj.dtype)
        if ch.size != cy * cx:
            raise ValueError(f"chunk {r},{c} decoded to {ch.size} values, expected {cy * cx}")
        ch = ch.reshape(cy, cx)
        ya, yb, xa, xb = max(y0, r), min(y1, r + cy), max(x0, c), min(x1, c + cx)
        out[ya - y0:yb - y0, xa - x0:xb - x0] = ch[ya - r:yb - r, xa - c:xb - c]
        job.say("INFO", f"chunk {k}/{len(jobs)} at rows {r}.. cols {c}..: {info.size / 1e3:.0f} KB in {dt:.1f}s ({info.size / 1e6 / max(dt, 1e-9):.2f} MB/s)")
    job.say("INFO", f"{obj.name}: {tot / 1e6:.2f} MB of chunk data in {time.time() - t_all:.1f}s")
    return out


def _read_window(job, obj, win, st):
    import numpy as np
    x0, x1, y0, y1 = win
    if S.get("fast", True) and obj.ndim == 2 and obj.chunks:
        try:
            a = _fast_chunks(job, obj, y0, y1, x0, x1)
            if not S.get("verified"):                # once per run: the same pixels through plain h5py must be identical
                t = time.time()
                cy0, cx0 = y0 + (y1 - y0) // 2, x0 + (x1 - x0) // 2
                ref = np.asarray(obj[cy0:cy0 + 8, cx0:cx0 + 8])
                mine = a[cy0 - y0:cy0 - y0 + 8, cx0 - x0:cx0 - x0 + 8]
                if not np.array_equal(ref, mine, equal_nan=ref.dtype.kind in "fc"):
                    S["fast"] = False
                    raise ValueError("exact-chunk result differs from the h5py read; exact-chunk reads switched off for this run")
                S["verified"] = True
                job.say("INFO", f"exact-chunk read verified against a plain h5py read of 8x8 pixels ({time.time() - t:.1f}s)")
            return a[::st, ::st]
        except Exception as ex:
            job.say("ERROR" if "differs" in str(ex) else "WARN", f"{obj.name}: exact-chunk read not used: {type(ex).__name__}: {ex}; using a plain h5py read")
            job.say("DEBUG", traceback.format_exc())
    return np.asarray(obj[(slice(None),) * (obj.ndim - 2) + (slice(y0, y1, st), slice(x0, x1, st))])


def _grid(job, h5, ent):
    """the box on this layer's grid: reads that grid's coordinate arrays once per job, only when a layer on it is loaded"""
    gn = ent["grid"]
    if gn not in job.grids:
        t = time.time()
        grp = h5[gn]
        ny, nx = grp["yCoordinates"].shape[0], grp["xCoordinates"].shape[0]
        epsg = _epsg(h5, grp, job.gid)
        x0, x1, y0, y1, corners = _window(grp, job.bbox, epsg)
        job.grids[gn] = {"ny": ny, "nx": nx, "epsg": epsg, "win": [x0, x1, y0, y1], "corners": corners}
        job.say("INFO", f"grid {gn}: {nx}x{ny} px, EPSG {epsg}; box = columns {x0}:{x1}, rows {y0}:{y1} ({x1 - x0}x{y1 - y0} px) [{time.time() - t:.1f}s]")
    G = job.grids[gn]
    if tuple(ent["shape"][-2:]) != (G["ny"], G["nx"]):
        raise ValueError(f"layer size {ent['shape'][-2]}x{ent['shape'][-1]} differs from its grid {G['ny']}x{G['nx']}")
    return G


def read_layer(job, ent):
    import numpy as np
    path, key = ent["path"], (job.gid, job.bbox, ent["path"])
    with CACHE_LOCK:
        if key in CACHE:
            CACHE.move_to_end(key)
            job.say("INFO", f"{path}: already in the server memory cache")
            return {"cached": True, "mem": CACHE[key]["bytes"]}
    for attempt in (1, 2, 3):
        try:
            with Beat(job, f"reading {path}"), META_LOCK:
                H5["job"] = job
                h5 = _open(job.gid, job.url)
                G = _grid(job, h5, ent)
                x0, x1, y0, y1 = G["win"]
                nz = ent["shape"][0] if len(ent["shape"]) == 3 else 1
                st = max(1, math.ceil(math.sqrt((x1 - x0) * (y1 - y0) * nz / MAXPX)))
                job.say("INFO", f"{path}: dtype {ent['dtype']}, file shape {ent['shape']}, reading rows {y0}:{y1} cols {x0}:{x1} step {st}")
                obj, r0, t0 = h5[path], _remote(), time.time()
                a = _read_window(job, obj, G["win"], st)
                secs, r1 = time.time() - t0, _remote()
                units = _py(obj.attrs["units"]) if "units" in obj.attrs else None
                fill = _py(obj.attrs["_FillValue"]) if "_FillValue" in obj.attrs else None
            break
        except ValueError:
            raise
        except Exception as ex:
            job.say("WARN", f"{path}: attempt {attempt}/3 failed: {type(ex).__name__}: {ex}")
            job.say("DEBUG", traceback.format_exc())
            _close()
            if attempt == 3:
                raise
            time.sleep(3 * attempt)
    corners = G["corners"]
    remote = None if r0 is None or r1 is None else r1 - r0
    a = a if a.ndim == 3 else a[None]
    cx = np.iscomplexobj(a)
    bad = ~np.isfinite(a)
    if fill is not None and not cx and isinstance(fill, (int, float)) and np.isfinite(fill):
        bad |= a == fill
    outs = [("magnitude", "amp", np.abs(a)), ("phase", "phase", np.angle(a))] if cx else [("", "data", a)]
    items, raws = [], []
    for suffix, kind, arr in outs:
        f = arr.astype("float32")
        f[bad] = np.nan
        v = f[~bad]
        s = dict.fromkeys(("min", "max", "mean", "p2", "p98"))
        if v.size:
            p2, p98 = np.percentile(v[::max(1, v.size // 200000)], [2, 98])
            s = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "p2": float(p2), "p98": float(p98)}
        items.append({"path": path, "label": (path.split("/")[-1] + " " + suffix).strip(), "kind": kind, "nz": f.shape[0],
                      "h": f.shape[1], "w": f.shape[2], "step": st, "units": units, "fill": fill, "dtype": ent["dtype"],
                      "valid": float(v.size / f.size), "corners": corners, **s})
        raws.append(np.ascontiguousarray(f, dtype="<f4").tobytes())
    n = sum(len(r) for r in raws)
    job.say("INFO", f"{path}: read {a.shape} {a.dtype} in {secs:.1f}s, remote bytes requested "
            + (f"{remote / 1e6:.1f} MB" if remote is not None else "n/a (this fsspec does not count)")
            + f", valid {items[0]['valid'] * 100:.1f}%, range {items[0]['min']} .. {items[0]['max']}, fill {fill}, units {units}, memory {n / 1e6:.1f} MB")
    with CACHE_LOCK:
        CACHE[key] = {"items": items, "raws": raws, "bytes": n}
        cap = S.get("cache_mb", 768) * 1e6
        while len(CACHE) > 1 and sum(e["bytes"] for e in CACHE.values()) > cap:
            k, e = CACHE.popitem(last=False)
            job.say("WARN", f"memory cache over {cap / 1e6:.0f} MB: dropped {k[2]} ({e['bytes'] / 1e6:.1f} MB); load it again to view it")
    return {"cached": False, "mem": n, "remote": remote}


def run_load(job, paths):
    lay = {e["path"]: e for e in job.catalog["layers"]}
    with job.lock:
        job.state, job.total, job.done = "loading", len(paths), 0
        for p in paths:
            job.layers[p] = {"status": "queued"}
    job.cancel.clear()
    job.say("INFO", f"loading {len(paths)} layer(s)")
    t_all, fails = time.time(), 0
    for p in paths:
        L = job.layers[p]
        if job.cancel.is_set():
            L["status"] = "cancelled"
            continue
        L.update(status="reading", t0=time.time(), t1=None, error=None)
        try:
            if p not in lay or "unavailable" in lay[p]:
                raise ValueError("layer is not loadable: " + str(lay.get(p, {}).get("unavailable", "unknown layer")))
            L.update(read_layer(job, lay[p]), status="done")
        except Exception as ex:
            fails += 1
            L.update(status="failed", error=f"{type(ex).__name__}: {ex}")
            job.say("ERROR", f"{p}: {L['error']}")
            job.say("DEBUG", traceback.format_exc())
        L["t1"] = time.time()
        job.done += 1
    left = sum(1 for p in paths if job.layers[p]["status"] == "cancelled")
    job.say("INFO" if not fails else "WARN", f"finished in {time.time() - t_all:.1f}s: {job.done - left - fails} ok, {fails} failed, {left} cancelled")
    job.state = "ready"


def get_job(jid):
    if jid not in S["lj"]:
        raise ValueError("unknown job (was the server restarted? close and reopen the overlay)")
    return S["lj"][jid]


def start_inspect(gid):
    g = S["granules"].get(gid)
    if not g or not S["bbox"]:
        raise ValueError("search again, this granule is not in the current results")
    for j in S["lj"].values():                      # reopening the overlay resumes the same job, loaded layers included
        if j.gid == gid and j.bbox == S["bbox"] and j.state != "error":
            return j
    job = Job(gid, g["url"], S["bbox"])
    S["lj"][job.id] = job
    threading.Thread(target=inspect, args=(job,), daemon=True).start()
    return job


def job_view(job, since):
    with job.lock:
        return {"now": time.time(), "state": job.state, "error": job.error, "steps": [dict(x) for x in job.steps],
                "log": job.log[since:], "next": len(job.log), "layers": {k: dict(v) for k, v in job.layers.items()},
                "done": job.done, "total": job.total, "ready": job.catalog is not None}


def layer_payload(jid, path):
    import base64
    job = get_job(jid)
    with CACHE_LOCK:
        e = CACHE.get((job.gid, job.bbox, path))
        if e is None:
            raise ValueError(f"not in the server memory cache any more (limit {S.get('cache_mb', 768)} MB); load it again")
        CACHE.move_to_end((job.gid, job.bbox, path))
        return {"items": [{**m, "data": base64.b64encode(r).decode()} for m, r in zip(e["items"], e["raws"])]}


# ------------------------------------------------------------------ clip jobs
def run_clip(job_id, ids, layers, workers):
    job = S["jobs"][job_id]
    todo = [(S["granules"][i]["url"], S["granules"][i]["name"], S["bbox"], S["out"], layers or None)
            for i in ids if i in S["granules"]]
    job["total"] = len(todo)
    try:
        with ProcessPoolExecutor(max(1, min(workers, len(todo) or 1)), mp_context=mp.get_context("spawn"),
                                 initializer=nf._init_worker) as ex:
            futs = {ex.submit(nf.clip_one, j): j[1] for j in todo}
            for fut in as_completed(futs):
                name, status, secs = fut.result()
                job["items"][name] = {"status": status, "secs": round(secs)}
                job["done"] += 1
    except Exception as ex:
        job["error"] = f"{type(ex).__name__}: {ex}"
    job["running"] = False


# ------------------------------------------------------------------ http
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                body = open(os.path.join(HERE, "nisar_browser.html"), "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/config":
                self._send({**S["defaults"], "out": os.path.abspath(S["out"]), "login": S["login"]})
            elif u.path == "/api/meta":
                self._send(read_meta(q["id"][0]))
            elif u.path == "/api/lj":
                self._send(job_view(get_job(q["id"][0]), int(q.get("since", ["0"])[0])))
            elif u.path == "/api/catalog":
                self._send(get_job(q["id"][0]).catalog or {})
            elif u.path == "/api/layer":
                self._send(layer_payload(q["job"][0], q["path"][0]))
            elif u.path == "/api/job":
                self._send(S["jobs"].get(q["id"][0], {}))
            else:
                self._send({"error": "not found"}, 404)
        except Exception as ex:
            self._send({"error": f"{type(ex).__name__}: {ex}"}, 500)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/search":
                self._send(do_search(body))
            elif self.path == "/api/inspect":
                j = start_inspect(body["id"])
                self._send({"job": j.id, "bbox": list(j.bbox)})
            elif self.path == "/api/load":
                j = get_job(body["job"])
                if j.state != "ready":
                    raise ValueError(f"cannot start a load while the job is {j.state}")
                if not body.get("paths"):
                    raise ValueError("no layers selected")
                j.state = "loading"
                threading.Thread(target=run_load, args=(j, list(body["paths"])), daemon=True).start()
                self._send({"ok": True})
            elif self.path == "/api/cancel":
                get_job(body["job"]).cancel.set()
                self._send({"ok": True})
            elif self.path == "/api/clip":
                if not S["bbox"]:
                    raise ValueError("search first")
                jid = str(int(time.time() * 1000))
                S["jobs"][jid] = {"running": True, "total": 0, "done": 0, "items": {}}
                threading.Thread(target=run_clip, daemon=True,
                                 args=(jid, body["ids"], body.get("layers"), int(body.get("workers") or 4))).start()
                self._send({"job": jid})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as ex:
            self._send({"error": f"{type(ex).__name__}: {ex}"}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--level", default="GUNW")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="./nisar_clips")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-login", action="store_true")
    ap.add_argument("--no-fast", action="store_true", help="read layers through plain h5py instead of exact chunk ranges")
    ap.add_argument("--cache-mb", type=int, default=768, help="server memory for loaded layers")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    S["out"] = a.out
    S["cache_mb"] = a.cache_mb
    S["fast"] = not a.no_fast
    os.makedirs(os.path.join(a.out, ".logs"), exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-5s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(a.out, ".logs", "nisar_browser.log"))])
    for noisy in ("fsspec", "asyncio", "aiohttp", "h5py", "urllib3", "asf_search"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    S["defaults"] = {"bbox": a.bbox, "level": a.level, "start": a.start or "", "end": a.end or ""}
    S["login"] = not a.no_login
    if S["login"]:
        user, pw = nr.credentials()
        os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"] = user, pw
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)       # localhost only: it holds your login
    url = f"http://127.0.0.1:{a.port}"
    print(f"NISAR browser on {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
