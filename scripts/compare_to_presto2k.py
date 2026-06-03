#!/usr/bin/env python3
"""
Proxy database comparison: custom reconstruction vs PReSto2k reference.

Produces every artifact the validation HTML needs — `comparison.json`,
stacked-archive / stacked-ptype temporal coverage PNGs, Robinson spatial
maps, and CSVs under `downloads/` (full TSID list, only-in-presto2k,
only-in-custom, dropped-records, and one CSV per compilation per database).

Runs inside the davidedge/lmr2 Docker image (cfr-env).

Usage:
  python compare_to_presto2k.py \
      --custom-pickle /app/lipd_cfr.pkl \
      --presto2k      /app/presto2k_pdb.pkl \
      --recon         /recons/job_r01_recon.nc \
      --query-params  /app/query_params.json \
      --skipped       /app/prepare-data/skipped_records.csv \
      --lipdverse     /app/cache/lipdverseQuery.csv \
      --out-dir       /validation
"""

import argparse
import ast
import csv
import json
import os
import pickle
import re
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

LIPDVERSE_URL = 'https://lipdverse.org/lipdverse/lipdverseQuery.zip'
LIPDVERSE_CSV_CACHE = '/tmp/lipdverseQuery.csv'

ARCHIVE_COLORS = {
    'Tree': '#228B22', 'Coral': '#FF6347', 'Ice': '#4169E1',
    'Lake': '#8B4513', 'Marine': '#006400', 'Speleothem': '#9370DB',
    'Borehole': '#FF8C00', 'Documents': '#808080', 'Bivalve': '#DEB887',
    'Sclerosponge': '#20B2AA', 'Hybrid': '#C0C0C0', 'Other': '#999999',
}

# Ptype-specific palette — reuse archive color as base, perturb for sub-types
def ptype_color(ptype, archive):
    base = ARCHIVE_COLORS.get(archive, '#999999')
    # Light perturbation to separate sub-types within an archive
    if '.' not in str(ptype):
        return base
    suffix_hash = hash(ptype.split('.', 1)[1]) & 0xFFFF
    r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
    shift = (suffix_hash % 60) - 30
    r, g, b = (max(0, min(255, c + shift)) for c in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'


# ═══════════════════════════════════════════════════════════════════════════
# Robust unpickling (cfr may not be importable here)
# ═══════════════════════════════════════════════════════════════════════════

class GenericUnpickler(pickle.Unpickler):
    """Tolerate missing cfr/pylipd classes when unpickling."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            class Stub:
                def __init__(self, *a, **kw): pass
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
            Stub.__name__ = name
            Stub.__module__ = module
            return Stub


# ═══════════════════════════════════════════════════════════════════════════
# Utility: normalize archive names across data sources
# ═══════════════════════════════════════════════════════════════════════════

_ARCHIVE_MAP = {
    'tree': 'Tree', 'wood': 'Tree',
    'coral': 'Coral',
    'ice': 'Ice', 'glacierice': 'Ice', 'groundice': 'Ice', 'ice core': 'Ice',
    'lake': 'Lake', 'lakesediment': 'Lake', 'terrestrialsediment': 'Lake',
    'lake sediment': 'Lake',
    'marine': 'Marine', 'marinesediment': 'Marine',
    'marine sediment': 'Marine',
    'speleothem': 'Speleothem',
    'sclerosponge': 'Sclerosponge',
    'borehole': 'Borehole',
    'bivalve': 'Bivalve', 'molluskshell': 'Bivalve',
    'documents': 'Documents', 'document': 'Documents',
    'hybrid': 'Hybrid',
    'other': 'Other',
}


def normalize_archive(source):
    if not source:
        return 'Other'
    s = str(source).split('.')[0].strip().lower()
    return _ARCHIVE_MAP.get(s, s.title() if s else 'Other')


# ═══════════════════════════════════════════════════════════════════════════
# lipdverseQuery loading and per-TSID lookup
# ═══════════════════════════════════════════════════════════════════════════

def ensure_lipdverse_csv(path):
    """Download and cache lipdverseQuery.csv if not present."""
    if path and os.path.exists(path):
        return path
    target = path or LIPDVERSE_CSV_CACHE
    os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, 'lv.zip')
        print(f'  downloading {LIPDVERSE_URL} ...')
        urllib.request.urlretrieve(LIPDVERSE_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            for m in zf.namelist():
                if m.endswith('.csv'):
                    zf.extract(m, tmp)
                    os.replace(os.path.join(tmp, m), target)
                    break
    print(f'  cached to {target}')
    return target


def load_lipdverse(path):
    """Return tsid -> { dataSetName, archive, variableName, compilations, ... }."""
    print(f'Loading lipdverseQuery from {path} ...')
    meta = {}
    ds_tsids = defaultdict(list)   # dataSetName -> [tsid, ...]  (for presto2k resolution)
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = (row.get('paleoData_TSid') or '').strip()
            if not t:
                continue
            ds = (row.get('dataSetName') or '').strip()
            arc = row.get('archiveType') or ''
            vn = row.get('paleoData_variableName') or ''
            comp_raw = (row.get('paleoData_mostRecentCompilations') or '').strip()
            comps = set()
            if comp_raw and comp_raw != 'NA':
                for tok in comp_raw.replace(';', ',').split(','):
                    tok = tok.strip().strip('"\'')
                    if tok:
                        comps.add(tok)
            meta[t] = {
                'dataSetName': ds,
                'archive': normalize_archive(arc),
                'archive_raw': arc,
                'variableName': vn,
                'compilations': comps,
            }
            if ds:
                ds_tsids[ds].append(t)
    print(f'  {len(meta)} TSIDs indexed across {len(ds_tsids)} datasets')
    return meta, ds_tsids


# ═══════════════════════════════════════════════════════════════════════════
# Presto2k: load records and resolve pids -> TSIDs
# ═══════════════════════════════════════════════════════════════════════════

def _time_stats(t_arr, clip_range):
    """Return (start, end, n_obs) clipped to the DA window."""
    if t_arr is None or t_arr.size == 0:
        return None, None, 0
    lo, hi = clip_range
    t_clip = t_arr[(t_arr >= lo) & (t_arr <= hi)]
    if t_clip.size == 0:
        return None, None, 0
    return int(np.floor(t_clip.min())), int(np.floor(t_clip.max())), int(t_clip.size)


def load_presto2k(path, lv_meta, ds_tsids, clip_range):
    """Return dict tsid -> record-info (ptype, archive, lat/lon/time_*).
    Time stats are clipped to ``clip_range`` (recon_period) so pre-CE data
    on a long record doesn't skew the earliest/latest stats."""
    print(f'Loading presto2k from {path} ...')
    with open(path, 'rb') as f:
        pdb = GenericUnpickler(f).load()
    records = pdb.records if hasattr(pdb, 'records') else pdb
    print(f'  {len(records)} records')

    known = set(lv_meta)
    out = {}
    unresolved = []
    for pid, rec in records.items():
        tsid = None
        for t in known:
            if pid.endswith('_' + t):
                tsid = t
                break
        if tsid is None:
            # pid starts with dataSetName_…; take suffix
            for ds in ds_tsids:
                if pid.startswith(ds + '_'):
                    candidate = pid[len(ds) + 1:]
                    if candidate in known:
                        tsid = candidate
                    else:
                        tsid = candidate  # accept even if not in lipdverse index
                    break
        if tsid is None:
            unresolved.append(pid)
            continue

        # Extract record attrs (tolerant of Stub records)
        def g(o, *names, default=None):
            for n in names:
                v = getattr(o, n, None)
                if v is not None:
                    return v
            return default

        t_arr = g(rec, 'time')
        t_arr = np.asarray(t_arr, dtype=float) if t_arr is not None else None
        if t_arr is not None:
            t_arr = t_arr[np.isfinite(t_arr)]
        start, end, n_obs = _time_stats(t_arr, clip_range)
        ptype = str(g(rec, 'ptype', default='') or '')
        arc_from_ptype = normalize_archive(ptype.split('.')[0] if ptype else '')
        arc = (arc_from_ptype
               if arc_from_ptype != 'Other'
               else lv_meta.get(tsid, {}).get('archive', 'Other'))
        out[tsid] = {
            'pid': pid,
            'ptype': ptype,
            'archive': arc,
            'lat': float(g(rec, 'lat', default=np.nan)),
            'lon': float(g(rec, 'lon', default=np.nan)),
            'time_start': start,
            'time_end': end,
            'n_obs': n_obs,
            'compilations': sorted(lv_meta.get(tsid, {}).get('compilations', set())),
        }
    if unresolved:
        print(f'  unresolved presto2k pids: {len(unresolved)}  '
              f'(examples: {unresolved[:3]})')
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Custom: load pickle + recon attrs
# ═══════════════════════════════════════════════════════════════════════════

def _parse_pid_list(s):
    s = str(s)
    if s.startswith('['):
        try:
            return set(str(p) for p in ast.literal_eval(s))
        except Exception:
            pass
    return set(p.strip().strip('"\'') for p in s.split(',') if p.strip())


def load_custom(pickle_path, recon_path, lv_meta):
    """Return (custom dict, pids_assim, pids_eval, recon_period tuple).
    recon_period is read from the recon's time axis."""
    print(f'Loading custom pickle from {pickle_path} ...')
    df = pd.read_pickle(pickle_path)
    print(f'  {len(df)} rows  ({df["paleoData_pages2kID"].nunique()} unique pids)')

    print(f'Loading recon attrs from {recon_path} ...')
    ds = xr.open_dataset(recon_path)
    pids_assim = _parse_pid_list(ds.attrs.get('pids_assim', ''))
    pids_eval = _parse_pid_list(ds.attrs.get('pids_eval', ''))
    # Derive recon_period from the netCDF time axis.
    time_axis = np.asarray(ds['time'].values).astype(int)
    recon_period = (int(time_axis.min()), int(time_axis.max()))
    ds.close()
    pids_used = pids_assim | pids_eval
    print(f'  pids_assim={len(pids_assim)}  pids_eval={len(pids_eval)}  '
          f'used={len(pids_used)}')
    print(f'  recon_period: {recon_period}')

    out = {}
    for _, r in df.iterrows():
        pid = str(r['paleoData_pages2kID'])
        if pid in out:
            continue  # duplicate (shouldn't happen with the patch, but be safe)
        t = np.asarray(r.get('year'), dtype=float)
        t = t[np.isfinite(t)]
        start, end, n_obs = _time_stats(t, recon_period)
        ptype = str(r.get('ptype', ''))
        arc = normalize_archive(ptype.split('.')[0] if ptype else '')
        # Prefer inCompilation info stored directly in the pickle if present
        # (populated by a future lipd_to_pdb.py update); fall back to
        # lipdverseQuery's mostRecent compilation.
        col_comps = r.get('paleoData_compilations')
        if col_comps and isinstance(col_comps, (list, tuple, set)):
            comps = sorted(str(c) for c in col_comps if c)
        else:
            comps = sorted(lv_meta.get(pid, {}).get('compilations', set()))
        out[pid] = {
            'ptype': ptype,
            'archive': arc,
            'lat': float(r.get('geo_meanLat', np.nan)),
            'lon': float(r.get('geo_meanLon', np.nan)),
            'variableName': str(r.get('paleoData_variableName', '')),
            'time_start': start,
            'time_end': end,
            'n_obs': n_obs,
            'compilations': comps,
            'in_assim': pid in pids_assim,
            'in_eval': pid in pids_eval,
        }
    return out, pids_assim, pids_eval, recon_period


# ═══════════════════════════════════════════════════════════════════════════
# Aggregations / tables
# ═══════════════════════════════════════════════════════════════════════════

def counter_table(tsids, keyfn):
    c = Counter()
    for t in tsids:
        c[keyfn(t)] += 1
    return c


def build_stats(custom, presto2k, used_tsids):
    def _stats(rec_dict):
        lats = [r['lat'] for r in rec_dict.values() if not np.isnan(r['lat'])]
        starts = [r['time_start'] for r in rec_dict.values() if r.get('time_start') is not None]
        ends = [r['time_end'] for r in rec_dict.values() if r.get('time_end') is not None]
        lens = [e - s for s, e in zip(starts, ends)]
        nobs = [r['n_obs'] for r in rec_dict.values() if r.get('n_obs')]
        archs = Counter(r['archive'] for r in rec_dict.values())
        ptypes = Counter(r['ptype'] for r in rec_dict.values())
        return {
            'records': len(rec_dict),
            'distinct_archives': len(archs),
            'distinct_ptypes': len(ptypes),
            'earliest_start': int(min(starts)) if starts else None,
            'latest_end': int(max(ends)) if ends else None,
            'median_record_length': int(np.median(lens)) if lens else None,
            'median_n_obs': int(np.median(nobs)) if nobs else None,
        }
    custom_used = {t: custom[t] for t in used_tsids if t in custom}
    return {
        'custom_used': _stats(custom_used),
        'presto2k': _stats(presto2k),
    }


_VERSION_SUFFIX_RE = re.compile(r'-\d[0-9_]*$')


def _compilation_name(full):
    """Strip a trailing ``-<version>`` suffix so `Pages2kTemperature-2_2_0`
    and `Pages2kTemperature-2_1_4` collapse to ``Pages2kTemperature``.
    Records typically carry every historic version of a compilation in
    ``paleoData_inCompilationBeta``; without this collapse they would
    double-count across versions."""
    return _VERSION_SUFFIX_RE.sub('', str(full))


def build_compilation_table(custom, presto2k, used_tsids):
    """Per-compilation side-by-side, aggregated by compilation name
    (versions collapsed). Each record is counted once per compilation it
    appears in."""
    custom_comps = defaultdict(set)
    custom_versions = defaultdict(set)
    for t in used_tsids:
        info = custom.get(t, {})
        raw = info.get('compilations') or ['(none)']
        for c in raw:
            name = _compilation_name(c)
            custom_comps[name].add(t)
            custom_versions[name].add(c)

    p2k_comps = defaultdict(set)
    p2k_versions = defaultdict(set)
    for t, info in presto2k.items():
        raw = info.get('compilations') or ['(none)']
        for c in raw:
            name = _compilation_name(c)
            p2k_comps[name].add(t)
            p2k_versions[name].add(c)

    all_comps = sorted(set(custom_comps) | set(p2k_comps))
    rows = []
    for c in all_comps:
        cu = custom_comps.get(c, set())
        pr = p2k_comps.get(c, set())
        versions = sorted(custom_versions.get(c, set()) |
                          p2k_versions.get(c, set()))
        rows.append({
            'compilation': c,
            'versions': versions,
            'custom_count': len(cu),
            'presto2k_count': len(pr),
            'shared': len(cu & pr),
            'custom_only': len(cu - pr),
            'p2k_only': len(pr - cu),
            '_custom_tsids': sorted(cu),
            '_p2k_tsids': sorted(pr),
        })
    return rows


def build_archive_table(custom, presto2k, used_tsids):
    shared = used_tsids & set(presto2k)
    only_c = used_tsids - set(presto2k)
    only_p = set(presto2k) - used_tsids

    def arc(t):
        return (custom.get(t, {}).get('archive')
                or presto2k.get(t, {}).get('archive', 'Other'))

    s = counter_table(shared, arc)
    c = counter_table(only_c, arc)
    p = counter_table(only_p, arc)
    arcs = sorted(set(s) | set(c) | set(p))
    rows = []
    for a in arcs:
        rows.append({
            'archive': a, 'shared': s.get(a, 0),
            'p2k_only': p.get(a, 0), 'custom_only': c.get(a, 0),
        })
    return rows


def build_ptype_table(custom, presto2k, used_tsids):
    shared = used_tsids & set(presto2k)
    only_c = used_tsids - set(presto2k)
    only_p = set(presto2k) - used_tsids

    def pt(t):
        return (custom.get(t, {}).get('ptype')
                or presto2k.get(t, {}).get('ptype') or '')

    s = counter_table(shared, pt)
    c = counter_table(only_c, pt)
    p = counter_table(only_p, pt)
    pts = sorted(set(s) | set(c) | set(p))
    return [{
        'ptype': p_, 'shared': s.get(p_, 0),
        'p2k_only': p.get(p_, 0), 'custom_only': c.get(p_, 0),
    } for p_ in pts]


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def _temporal_plot(df_rows, group_key, title_suffix, out_path,
                   color_fn=None):
    """df_rows: iterable of dicts with time_start, time_end, archive, ptype, source."""
    recs = [(r['time_start'], r['time_end'], r[group_key], r['source'])
            for r in df_rows
            if r.get('time_start') is not None and r.get('time_end') is not None]
    if not recs:
        return False

    starts = [r[0] for r in recs]; ends = [r[1] for r in recs]
    lo = max(0, int(np.percentile(starts, 2)))
    hi = min(2100, int(np.percentile(ends, 98)))
    if hi <= lo:
        lo, hi = min(starts), max(ends)

    n_bins = 100
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    groups = sorted({r[2] for r in recs}, key=lambda g: (-sum(1 for r in recs if r[2] == g), g))

    fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, src, label in [(ax_c, 'custom', 'Custom Run'),
                            (ax_p, 'presto2k', 'PReSto2k')]:
        bottom = np.zeros(n_bins)
        for g in groups:
            counts = np.zeros(n_bins)
            for t0, t1, gr, s in recs:
                if gr != g or s != src:
                    continue
                for i in range(n_bins):
                    if t0 <= edges[i + 1] and t1 >= edges[i]:
                        counts[i] += 1
            if counts.sum() == 0:
                continue
            col = color_fn(g) if color_fn else ARCHIVE_COLORS.get(g, '#999')
            ax.bar(centers, counts, width=width * 0.95, bottom=bottom,
                   label=g, color=col, edgecolor='none')
            bottom += counts
        ax.set_ylabel(f'{label}\n# records')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  fontsize=7, ncol=1, borderaxespad=0, frameon=False)

    ax_p.set_xlabel('Year CE')
    fig.suptitle(f'Temporal coverage — {title_suffix}', fontsize=13)
    ax_c.set_xlim(lo, hi)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return True


def plot_spatial_maps(custom, presto2k, used_tsids, out_path):
    """Side-by-side Robinson maps with one shared legend below, colored
    by archive type. Legend order is a union across both panels so the
    symbols line up regardless of which archives are present."""
    # Collect union of archives across both panels so the legend lists
    # every archive once, in a consistent order.
    def _arc_points(recs):
        arc_pts = defaultdict(list)
        for _, r in recs:
            if np.isfinite(r['lat']) and np.isfinite(r['lon']):
                arc_pts[r['archive']].append((r['lon'], r['lat']))
        return arc_pts

    c_recs = [(t, custom[t]) for t in used_tsids if t in custom]
    p_recs = list(presto2k.items())
    arc_c = _arc_points(c_recs)
    arc_p = _arc_points(p_recs)
    all_archives = sorted(
        set(arc_c) | set(arc_p),
        key=lambda a: -max(len(arc_c.get(a, [])), len(arc_p.get(a, []))))

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], hspace=0.05)
    ax_c = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax_p = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')

    handles = []
    for arc in all_archives:
        color = ARCHIVE_COLORS.get(arc, '#999')
        # Dummy handle for the shared legend
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                   markerfacecolor=color,
                                   markeredgecolor='black',
                                   markeredgewidth=0.3,
                                   markersize=7, label=arc))

    for ax, title, arc_pts, recs in [
        (ax_c, f'Custom Run ({len(c_recs)} records used in DA)',
         arc_c, c_recs),
        (ax_p, f'PReSto2k ({len(p_recs)} records)', arc_p, p_recs),
    ]:
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        for arc in all_archives:
            pts = arc_pts.get(arc, [])
            if not pts:
                continue
            lons, lats = zip(*pts)
            ax.scatter(lons, lats, s=22, alpha=0.75,
                       color=ARCHIVE_COLORS.get(arc, '#999'),
                       edgecolor='black', linewidth=0.3,
                       transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=12)

    legend_ax.legend(
        handles=handles,
        loc='center', ncol=min(len(all_archives), 8),
        fontsize=9, frameon=False,
        handletextpad=0.4, columnspacing=1.5)

    fig.suptitle('Spatial distribution by archive type', fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CSV writers
# ═══════════════════════════════════════════════════════════════════════════

def write_records_csv(path, custom, presto2k, used_tsids, source_filter=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p2k_set = set(presto2k)
    rows = []
    all_t = used_tsids | p2k_set
    for t in sorted(all_t):
        in_c = t in used_tsids
        in_p = t in p2k_set
        src = 'both' if (in_c and in_p) else ('custom_run' if in_c else 'presto2k')
        if source_filter and src != source_filter:
            continue
        info = custom.get(t) or presto2k.get(t) or {}
        rows.append({
            'tsid': t,
            'source': src,
            'archive': info.get('archive', ''),
            'ptype': info.get('ptype', ''),
            'lat': info.get('lat', ''),
            'lon': info.get('lon', ''),
            'time_start': info.get('time_start', ''),
            'time_end': info.get('time_end', ''),
            'n_obs': info.get('n_obs', ''),
            'compilations': ';'.join(info.get('compilations') or []),
            'in_assim': bool(custom.get(t, {}).get('in_assim')) if in_c else '',
            'in_eval': bool(custom.get(t, {}).get('in_eval')) if in_c else '',
            'presto2k_pid': presto2k.get(t, {}).get('pid', '') if in_p else '',
        })
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['tsid', 'source'])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_compilation_csv(path, tsids, custom, presto2k):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['tsid', 'archive', 'ptype', 'lat', 'lon',
                    'time_start', 'time_end', 'n_obs'])
        for t in sorted(tsids):
            info = custom.get(t) or presto2k.get(t) or {}
            w.writerow([t, info.get('archive', ''), info.get('ptype', ''),
                        info.get('lat', ''), info.get('lon', ''),
                        info.get('time_start', ''), info.get('time_end', ''),
                        info.get('n_obs', '')])


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--custom-pickle', required=True)
    ap.add_argument('--presto2k', required=True)
    ap.add_argument('--recon', required=True,
                    help='Path to per-seed netCDF with pids_assim/pids_eval attrs')
    ap.add_argument('--query-params', default=None)
    ap.add_argument('--skipped', default=None,
                    help='Path to prepare-data/skipped_records.csv (optional)')
    ap.add_argument('--lipdverse', default=LIPDVERSE_CSV_CACHE)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out = args.out_dir
    downloads = os.path.join(out, 'downloads')
    comp_dir = os.path.join(downloads, 'compilations')
    os.makedirs(comp_dir, exist_ok=True)

    # ── Load data ──
    lv_path = ensure_lipdverse_csv(args.lipdverse)
    lv_meta, ds_tsids = load_lipdverse(lv_path)

    # Load custom first so we can use the recon_period as the clip window.
    custom, pids_assim, pids_eval, recon_period = load_custom(
        args.custom_pickle, args.recon, lv_meta)
    presto2k = load_presto2k(args.presto2k, lv_meta, ds_tsids, recon_period)
    pids_used = pids_assim | pids_eval

    # Funnel
    funnel = {
        'requested': None,
        'in_pickle': len(custom),
        'after_ptype_filter': None,  # infer from skipped_records if present
        'post_psm': len(pids_used),
        'assimilated': len(pids_assim),
        'eval': len(pids_eval),
        'recon_period': list(recon_period),
    }
    requested_compilations = []
    if args.query_params and os.path.exists(args.query_params):
        with open(args.query_params) as f:
            qp = json.load(f)
        req = set(qp.get('tsids') or [])
        removed = set(qp.get('removedTsids') or [])
        funnel['requested'] = len(req - removed)
        funnel['removed_tsids'] = len(removed)
        comp_raw = qp.get('compilation') or ''
        if isinstance(comp_raw, str):
            requested_compilations = [c.strip()
                                       for c in comp_raw.split(',') if c.strip()]
        elif isinstance(comp_raw, list):
            requested_compilations = [str(c).strip()
                                       for c in comp_raw if str(c).strip()]

    # ── Compilation table ──
    comp_rows = build_compilation_table(custom, presto2k, pids_used)

    # Write per-compilation CSVs, stripping private _keys
    for row in comp_rows:
        slug = (row['compilation']
                .replace('/', '_').replace(' ', '_').replace('(', '')
                .replace(')', ''))
        if row['_custom_tsids']:
            write_compilation_csv(
                os.path.join(comp_dir, f'custom_run_{slug}.csv'),
                row['_custom_tsids'], custom, presto2k)
            row['custom_csv'] = f'downloads/compilations/custom_run_{slug}.csv'
        if row['_p2k_tsids']:
            write_compilation_csv(
                os.path.join(comp_dir, f'presto2k_{slug}.csv'),
                row['_p2k_tsids'], custom, presto2k)
            row['presto2k_csv'] = f'downloads/compilations/presto2k_{slug}.csv'
        del row['_custom_tsids']
        del row['_p2k_tsids']

    # ── Archive / ptype tables ──
    archive_rows = build_archive_table(custom, presto2k, pids_used)
    ptype_rows = build_ptype_table(custom, presto2k, pids_used)
    stats = build_stats(custom, presto2k, pids_used)

    # ── Only-lists (top 25 for HTML + full CSV) ──
    p2k_set = set(presto2k)
    only_p2k = sorted(p2k_set - pids_used)
    only_custom = sorted(pids_used - p2k_set)
    shared = sorted(pids_used & p2k_set)

    def list_slice(tsids, n=25):
        out = []
        for t in tsids[:n]:
            info = custom.get(t) or presto2k.get(t) or {}
            out.append({
                'tsid': t,
                'archive': info.get('archive', ''),
                'ptype': info.get('ptype', ''),
                'dataSetName': lv_meta.get(t, {}).get('dataSetName', ''),
                'n_obs': info.get('n_obs', 0),
                'time_start': info.get('time_start'),
                'time_end': info.get('time_end'),
            })
        return out

    # ── Dropped records (skipped_records.csv + PSM-fail inference) ──
    dropped = []
    if args.skipped and os.path.exists(args.skipped):
        with open(args.skipped, encoding='utf-8') as f:
            dropped.extend(list(csv.DictReader(f)))
    # Post-pickle drops: records in pickle but not in pids_used
    pickle_not_used = set(custom) - pids_used
    for t in sorted(pickle_not_used):
        info = custom[t]
        # Hard to distinguish ptype-filter vs PSM-failure without the log;
        # label as 'post-pickle filter (ptype or PSM calibration)'.
        dropped.append({
            'dataSetName': lv_meta.get(t, {}).get('dataSetName', ''),
            'TSID': t,
            'variableName': info.get('variableName', ''),
            'archiveType': info.get('archive', ''),
            'reason': 'filtered after pickle (ptype filter or PSM calibration)',
        })
    dropped_csv = os.path.join(downloads, 'dropped_records.csv')
    os.makedirs(os.path.dirname(dropped_csv), exist_ok=True)
    if dropped:
        with open(dropped_csv, 'w', newline='', encoding='utf-8') as f:
            fields = sorted({k for r in dropped for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(dropped)
    dropped_counter = Counter(r.get('reason', 'unknown') for r in dropped)

    # ── Temporal-coverage plots ──
    plot_rows = []
    for t in pids_used:
        info = custom.get(t)
        if not info:
            continue
        plot_rows.append({
            'time_start': info.get('time_start'), 'time_end': info.get('time_end'),
            'archive': info.get('archive', 'Other'), 'ptype': info.get('ptype', ''),
            'source': 'custom',
        })
    for t, info in presto2k.items():
        plot_rows.append({
            'time_start': info.get('time_start'), 'time_end': info.get('time_end'),
            'archive': info.get('archive', 'Other'), 'ptype': info.get('ptype', ''),
            'source': 'presto2k',
        })

    print('Rendering temporal-coverage plots ...')
    tc_arc_ok = _temporal_plot(
        plot_rows, 'archive', 'by archive type',
        os.path.join(out, 'temporal_coverage_archive.png'))
    def _pcol(p):
        return ptype_color(p, normalize_archive(p.split('.')[0] if p else ''))
    tc_pt_ok = _temporal_plot(
        plot_rows, 'ptype', 'by proxy type (ptype)',
        os.path.join(out, 'temporal_coverage_ptype.png'),
        color_fn=_pcol)

    print('Rendering spatial map ...')
    plot_spatial_maps(custom, presto2k, pids_used,
                      os.path.join(out, 'spatial_map_comparison.png'))

    # ── Full CSV artifacts ──
    print('Writing CSV artifacts ...')
    n_all = write_records_csv(os.path.join(downloads, 'used_vs_presto2k.csv'),
                               custom, presto2k, pids_used)
    n_p2k = write_records_csv(os.path.join(downloads, 'only_presto2k.csv'),
                               custom, presto2k, pids_used, source_filter='presto2k')
    n_cust = write_records_csv(os.path.join(downloads, 'only_custom.csv'),
                                custom, presto2k, pids_used, source_filter='custom_run')

    # ── Comparison JSON ──
    comparison = {
        'funnel': funnel,
        'requested_compilations': requested_compilations,
        'stats': stats,
        'compilation_rows': comp_rows,
        'archive_rows': archive_rows,
        'ptype_rows': ptype_rows,
        'counts': {
            'shared': len(shared),
            'only_presto2k': len(only_p2k),
            'only_custom': len(only_custom),
            'total_unique': len(shared) + len(only_p2k) + len(only_custom),
            'assimilated': len(pids_assim),
            'eval': len(pids_eval),
        },
        'only_presto2k_preview': list_slice(only_p2k),
        'only_custom_preview': list_slice(only_custom),
        'dropped_reasons': dict(dropped_counter),
        'artifacts': {
            'temporal_coverage_archive': 'temporal_coverage_archive.png' if tc_arc_ok else None,
            'temporal_coverage_ptype': 'temporal_coverage_ptype.png' if tc_pt_ok else None,
            'spatial_map': 'spatial_map_comparison.png',
            'downloads': {
                'all': 'downloads/used_vs_presto2k.csv',
                'only_presto2k': 'downloads/only_presto2k.csv',
                'only_custom': 'downloads/only_custom.csv',
                'dropped_records': 'downloads/dropped_records.csv' if dropped else None,
            },
        },
    }
    out_json = os.path.join(out, 'comparison.json')
    with open(out_json, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f'\nWrote comparison.json  '
          f'(shared={len(shared)} / p2k-only={len(only_p2k)} / '
          f'custom-only={len(only_custom)})')
    print(f'All records CSV: {n_all} rows')
    print(f'Only-p2k CSV: {n_p2k} rows   Only-custom CSV: {n_cust} rows')


if __name__ == '__main__':
    main()
