#!/usr/bin/env python3
"""
Convert LiPD zip archive to cfr.ProxyDatabase pickle.

Uses pylipd's built-in get_timeseries() to extract proxy records from the
RDF graph (avoids writing custom SPARQL with fragile property paths), then
maps the flat time-series dicts to cfr.ProxyRecord objects.

Usage:
    python lipd_to_pdb.py <lipd_files.zip> <output_lipd_cfr.pkl> [query_params.json]

When query_params.json is supplied, only rows whose `paleoData_TSid` is in
`tsids - removedTsids` are retained. pylipd stores the presto-catalog TSID
under `paleoData_TSid` (capital T, lowercase id) — not `TSID` or
`paleoData_TSID`. Without this positive filter, every sibling paleoData
column in each .lpd leaks in (sampleCount, EPS, RBAR, segmentLength,
correlationCoefficient, ARSTAN / residualChronology flavors of the same
chronology), each getting its own PSM and Kalman update downstream.
"""

import sys
import os
import re
import math
import json
import zipfile
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from pylipd.lipd import LiPD


# ── Proxy type mapping ────────────────────────────────────────────────────────
PTYPE_MAP = {
    ('tree',            'trw'):                      'tree.TRW',
    ('tree',            'tree ring width'):           'tree.TRW',
    ('tree',            'ringwidth'):                 'tree.TRW',
    ('tree',            'ring width'):                'tree.TRW',
    ('tree',            'mxd'):                       'tree.MXD',
    ('tree',            'maximum latewood density'):  'tree.MXD',
    ('wood',            'trw'):                       'tree.TRW',
    ('wood',            'ringwidth'):                 'tree.TRW',
    ('wood',            'ring width'):                'tree.TRW',
    ('wood',            'mxd'):                       'tree.MXD',
    ('coral',           'd18o'):                      'coral.d18O',
    ('coral',           'srca'):                      'coral.SrCa',
    ('coral',           'calcification'):             'coral.calc',
    ('sclerosponge',    'd18o'):                      'sclerosponge.d18O',
    ('sclerosponge',    'srca'):                      'sclerosponge.SrCa',
    ('ice core',        'd18o'):                      'ice.d18O',
    ('ice core',        'dd'):                        'ice.dD',
    ('ice core',        'd2h'):                       'ice.dD',
    ('ice core',        'melt'):                      'ice.melt',
    ('ice core',        'accumulation'):              'ice.accumulation',
    ('glacierice',      'd18o'):                      'ice.d18O',
    ('glacierice',      'dd'):                        'ice.dD',
    ('lake sediment',   'varve_thickness'):           'lake.varve_thickness',
    ('lake sediment',   'varve thickness'):           'lake.varve_thickness',
    ('lake sediment',   'varve_property'):            'lake.varve_property',
    ('lake sediment',   'chironomid'):                'lake.chironomid',
    ('lake sediment',   'midge'):                     'lake.midge',
    ('lake sediment',   'reflectance'):               'lake.reflectance',
    ('lake sediment',   'bsi'):                       'lake.BSi',
    ('lake sediment',   'accumulation'):              'lake.accumulation',
    ('lakesediment',    'chironomid'):                'lake.chironomid',
    ('lakesediment',    'reflectance'):               'lake.reflectance',
    ('lakesediment',    'bsi'):                       'lake.BSi',
    ('marine sediment', 'alkenone'):                  'marine.alkenone',
    ('marine sediment', 'uk37'):                      'marine.alkenone',
    ('marine sediment', 'mgca'):                      'marine.MgCa',
    ('marine sediment', 'mg/ca'):                     'marine.MgCa',
    ('marine sediment', 'tex86'):                     'marine.other',
    ('marine sediment', 'temperature'):               'marine.other',
    ('marinesediment',  'alkenone'):                  'marine.alkenone',
    ('marinesediment',  'uk37'):                      'marine.alkenone',
    ('marinesediment',  'mgca'):                      'marine.MgCa',
    ('borehole',        'temperature'):               'borehole',
    ('speleothem',      'd18o'):                      'speleothem.d18O',
    ('documents',       'temperature'):               'documents',
    ('bivalve',         'd18o'):                      'bivalve.d18O',
    ('molluskshell',    'd18o'):                      'bivalve.d18O',
}

ARCHIVE_DEFAULTS = {
    'tree':                 'tree.TRW',
    'wood':                 'tree.TRW',
    'coral':                'coral.d18O',
    'ice core':             'ice.d18O',
    'glacierice':           'ice.d18O',
    'lake sediment':        'lake.other',
    'lakesediment':         'lake.other',
    'marine sediment':      'marine.other',
    'marinesediment':       'marine.other',
    'speleothem':           'speleothem.d18O',
    'borehole':             'borehole',
    'documents':            'documents',
    'sclerosponge':         'sclerosponge.d18O',
    'bivalve':              'bivalve.d18O',
    'molluskshell':         'bivalve.d18O',
    'hybrid':               'hybrid',
    'peat':                 'lake.other',
    'terrestrialsediment':  'lake.other',
}


def create_ptype(archive_type, standard_name):
    arch = str(archive_type or '').lower().strip()
    std  = str(standard_name  or '').lower().strip()
    arch_nsp = arch.replace(' ', '')
    key = (arch, std)
    if key in PTYPE_MAP:
        return PTYPE_MAP[key]
    # Also try with spaces removed from archive
    for (a, s), ptype in PTYPE_MAP.items():
        if a.replace(' ', '') == arch_nsp and s == std:
            return ptype
    # Partial match on std name
    for (a, s), ptype in PTYPE_MAP.items():
        if (a == arch or a.replace(' ', '') == arch_nsp) and s and s in std:
            return ptype
    return ARCHIVE_DEFAULTS.get(arch, ARCHIVE_DEFAULTS.get(arch_nsp, f'{arch}.unknown'))


# ── Seasonality conversion ────────────────────────────────────────────────────
MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mar': 3,  'apr': 4,  'may': 5,  'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9,  'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'june': 6, 'july': 7, 'august': 8, 'september': 9,
    'october': 10, 'november': 11, 'december': 12,
}
ANNUAL = list(range(1, 13))


def convert_seasonality(seasonality_str, latitude=None):
    if not seasonality_str or (isinstance(seasonality_str, float) and math.isnan(seasonality_str)):
        return ANNUAL
    s = str(seasonality_str).strip().lower()
    if not s or s in ('nan', 'none', 'annual', 'annual (all)', 'year-round'):
        return ANNUAL

    nh = (latitude is None) or (float(latitude) >= 0)
    named = {
        'summer':         [6, 7, 8]           if nh else [12, 1, 2],
        'winter':         [12, 1, 2]          if nh else [6, 7, 8],
        'spring':         [3, 4, 5]           if nh else [9, 10, 11],
        'fall':           [9, 10, 11]         if nh else [3, 4, 5],
        'autumn':         [9, 10, 11]         if nh else [3, 4, 5],
        'warm season':    [6, 7, 8]           if nh else [12, 1, 2],
        'cold season':    [12, 1, 2]          if nh else [6, 7, 8],
        'growing season': [4, 5, 6, 7, 8, 9] if nh else [10, 11, 12, 1, 2, 3],
        'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11],
    }
    if s in named:
        return named[s]

    m = re.match(r'([a-z]+)[^a-z]+([a-z]+)', s)
    if m:
        m1, m2 = MONTH_ABBR.get(m.group(1)), MONTH_ABBR.get(m.group(2))
        if m1 and m2:
            return list(range(m1, m2 + 1)) if m1 <= m2 else list(range(m1, 13)) + list(range(1, m2 + 1))

    nums = re.findall(r'-?\d+', s)
    if nums:
        months = [abs(int(n)) for n in nums if 1 <= abs(int(n)) <= 12]
        if months:
            return months

    if s in MONTH_ABBR:
        return [MONTH_ABBR[s]]
    return ANNUAL


def time_to_year_ce(arr, var_name, std_name=''):
    """Convert time axis to year CE. Age BP → 1950 − age; age ka → 1950 − age*1000."""
    v = str(var_name or '').lower()
    s = str(std_name  or '').lower()
    if 'ka' in v or 'ka' in s:
        return 1950.0 - arr * 1000.0
    if 'age' in v or 'age' in s or 'bp' in v or 'bp' in s:
        return 1950.0 - arr
    return arr  # already year CE


# ── Time-series dict helpers ──────────────────────────────────────────────────
# Variable names that are time axes (not proxy data)
_TIME_VARS = {
    'year', 'age', 'yearce', 'agebp', 'ageka', 'yearad', 'year_ad', 'year_bp',
    'age_bp', 'age_ka', 'years', 'ages', 'time', 'yearrounded', 'yearb2k',
    'ybp', 'ka', 'yearensemble', 'agemedian', 'agebchron', 'agecopra',
    'agebacon', 'agelinreg', 'agelininterp', 'ageoxcal', 'ageoriginal',
}

# Variables that are metadata / depth / uncertainty — skip as proxy
_SKIP_VARS = {
    'depth', 'depthtop', 'depthbottom', 'depthcomposite', 'section',
    'core', 'sampleid', 'notes', 'material', 'deletethis',
    'needstobechanged', 'latitude', 'longitude', 'elevation',
    'uncertainty', 'uncertaintylow', 'uncertaintyhigh',
}


def _is_time_var(vname):
    v = str(vname or '').strip().lower().replace(' ', '').replace('-', '').replace('_', '')
    return v in _TIME_VARS or v.startswith('age') or v.startswith('year')


def _is_skip_var(vname):
    v = str(vname or '').strip().lower()
    return v in _SKIP_VARS or v.startswith('depth') or v.startswith('uncertainty')


def _to_float_array(val):
    if val is None:
        return None
    try:
        arr = np.array(list(val) if not isinstance(val, (list, np.ndarray)) else val, dtype=float)
        if arr.ndim == 0 or arr.size == 0 or not np.any(np.isfinite(arr)):
            return None
        return arr
    except (TypeError, ValueError):
        return None


def _get_time_from_row(row):
    """
    Extract the time array and its variable name from a pylipd ts row dict.

    pylipd's get_timeseries() stores the co-located time axis as top-level
    keys (e.g. 'age', 'year') alongside paleoData_* proxy keys.
    """
    # Priority: year-like keys first (already in year CE), then age-like
    for key in ('year', 'yearCE', 'yearAD', 'Year', 'yearRounded',
                'age', 'ageBP', 'ageKa', 'Age'):
        val = row.get(key)
        if val is not None:
            arr = _to_float_array(val)
            if arr is not None:
                return arr, key

    # Fall back to time_values + time_variableName
    tv  = row.get('time_values') or row.get('paleoData_time_values')
    tvn = row.get('time_variableName', '')
    if tv is not None:
        arr = _to_float_array(tv)
        if arr is not None:
            return arr, str(tvn)

    return None, ''


def _get_scalar(row, *keys, default=0.0):
    for k in keys:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    zip_path    = sys.argv[1]
    output_path = sys.argv[2]
    qp_path     = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"Input:  {zip_path}")
    print(f"Output: {output_path}")

    # Load TSID filter from query_params.json if provided. When present,
    # `requested_tsids` is the positive whitelist (tsids - removedTsids); rows
    # whose `paleoData_TSid` is not in this set are rejected.
    requested_tsids = None
    removed_tsids = set()
    if qp_path and os.path.isfile(qp_path):
        print(f"Query params: {qp_path}")
        with open(qp_path) as f:
            qp = json.load(f)
        removed_tsids = set(qp.get('removedTsids') or [])
        requested = set(qp.get('tsids') or [])
        if requested:
            requested_tsids = requested - removed_tsids
            print(f"  TSID whitelist: {len(requested_tsids)} requested "
                  f"({len(removed_tsids)} explicitly removed)")
        elif removed_tsids:
            print(f"  Will remove {len(removed_tsids)} TSIDs from removedTsids list "
                  f"(no positive whitelist)")
    else:
        print("Query params: not provided (TSID filter disabled)")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUnzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        n_files = sum(1 for f in os.listdir(tmpdir) if f.endswith('.lpd'))
        print(f"Extracted {n_files} .lpd files")

        print("\nLoading with pylipd (muting verbose output) ...")
        with open(os.devnull, 'w') as devnull:
            _real_stdout, _real_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                L = LiPD()
                L.load_from_dir(tmpdir)
                all_ds = L.get_all_dataset_names()
            finally:
                sys.stdout, sys.stderr = _real_stdout, _real_stderr
        print(f"Loaded {len(all_ds)} datasets")

        print("Extracting time series via pylipd get_timeseries() ...")
        with open(os.devnull, 'w') as devnull:
            _real_stdout, _real_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = L.get_timeseries(all_ds)
            finally:
                sys.stdout, sys.stderr = _real_stdout, _real_stderr

        # Normalise whatever get_timeseries() returns into a flat list of dicts.
        # pylipd API has varied across versions:
        #   - dict  {dataset_name: [ts, ...]}   (observed in pylipd 1.5.x)
        #   - tuple (ts_list, df)
        #   - list  [ts, ...]
        #   - DataFrame
        rows = []
        if isinstance(result, dict):
            for val in result.values():
                if isinstance(val, list):
                    rows.extend(val)
                elif val is not None:
                    rows.append(val)
        elif isinstance(result, tuple):
            ts_list, df = result
            if ts_list and isinstance(ts_list, list):
                rows = ts_list
            elif df is not None and hasattr(df, 'iterrows'):
                rows = [r.to_dict() for _, r in df.iterrows()]
        elif isinstance(result, list):
            rows = result
        elif hasattr(result, 'iterrows'):
            rows = [r.to_dict() for _, r in result.iterrows()]

        # Each row may be a TimeSeries object — coerce to plain dict
        rows = [r.to_dict() if hasattr(r, 'to_dict') else
                (dict(r) if not isinstance(r, dict) else r)
                for r in rows]

        if not rows:
            raise RuntimeError(
                "get_timeseries() returned no rows — check pylipd version "
                "and that .lpd files were loaded correctly"
            )
        print(f"Got {len(rows)} time series")

    # ── Pre-filter: separate proxy candidates from time/depth/metadata rows ──
    proxy_rows = []
    n_meta = 0
    for row in rows:
        vname = str(row.get('paleoData_variableName') or '').strip()
        if not vname or _is_time_var(vname) or _is_skip_var(vname):
            n_meta += 1
            continue
        proxy_rows.append(row)
    print(f"Filtered {n_meta} time/depth/metadata rows, {len(proxy_rows)} proxy candidates remain")

    # ── Build DataFrame (cfr.ProxyDatabase.fetch expects pd.read_pickle → from_df) ──
    df_rows = []
    n_ok    = 0
    n_skip  = 0

    SKIP_MISSING_VALUES  = 'missing proxy values'
    SKIP_MISSING_TIME    = 'missing time axis'
    SKIP_ALL_NAN         = 'no valid time-value pairs (all NaN)'
    SKIP_CONSTANT        = 'constant value (zero variance → EnKF blow-up)'
    SKIP_BAD_COORDS      = 'missing/invalid coordinates'
    SKIP_REMOVED_TSID    = 'user-removed TSID (removedTsids)'
    SKIP_NOT_REQUESTED   = 'TSID not in query_params.tsids whitelist'
    SKIP_NO_TSID         = 'row has no paleoData_TSid'

    skip_records = []              # per-record details for CSV

    def _skip(reason, row, archive=None):
        nonlocal n_skip
        n_skip += 1
        arch = archive or str(row.get('archiveType') or row.get('archive') or 'unknown').lower().strip()
        skip_records.append({
            'dataSetName': str(row.get('dataSetName') or ''),
            'TSID': str(row.get('paleoData_TSid') or ''),
            'variableName': str(row.get('paleoData_variableName') or ''),
            'archiveType': arch,
            'reason': reason,
        })

    for row in proxy_rows:
        vname = str(row.get('paleoData_variableName') or '').strip()
        archive = str(row.get('archiveType') or row.get('archive') or '').lower().strip()

        # Apply TSID filter first (before expensive array work). pylipd exposes
        # presto's TSID under `paleoData_TSid` — not `TSID` or `paleoData_TSID`.
        tsid = row.get('paleoData_TSid')
        if not tsid:
            _skip(SKIP_NO_TSID, row, archive)
            continue
        if requested_tsids is not None and tsid not in requested_tsids:
            _skip(SKIP_NOT_REQUESTED, row, archive)
            continue
        if removed_tsids and tsid in removed_tsids:
            _skip(SKIP_REMOVED_TSID, row, archive)
            continue

        # Proxy values
        val_arr = _to_float_array(row.get('paleoData_values'))
        if val_arr is None:
            _skip(SKIP_MISSING_VALUES, row, archive)
            continue

        # Time axis
        time_raw, time_vname = _get_time_from_row(row)
        if time_raw is None:
            _skip(SKIP_MISSING_TIME, row, archive)
            continue
        time_arr = time_to_year_ce(time_raw, time_vname,
                                    row.get('time_standardName', ''))

        # Align, remove NaNs, sort ascending
        n = min(len(time_arr), len(val_arr))
        time_arr, val_arr = time_arr[:n], val_arr[:n]
        mask = np.isfinite(time_arr) & np.isfinite(val_arr)
        if not mask.any():
            _skip(SKIP_ALL_NAN, row, archive)
            continue
        time_arr, val_arr = time_arr[mask], val_arr[mask]
        idx = np.argsort(time_arr)
        time_arr, val_arr = time_arr[idx], val_arr[idx]

        # Skip constant-value records: OLS slope=0 → varye=0, MSE=0 → ob_err=0
        # → kdenom=0 → EnKF Kalman gain blows up
        if np.std(val_arr) < 1e-6:
            _skip(SKIP_CONSTANT, row, archive)
            continue

        # Coordinates
        try:
            lat  = _get_scalar(row, 'geo_meanLat',  'geo_meanLatitude',  'latitude')
            lon  = _get_scalar(row, 'geo_meanLon',  'geo_meanLongitude', 'longitude')
            elev = _get_scalar(row, 'geo_meanElev', 'geo_meanElevation', 'elevation')
        except Exception:
            _skip(SKIP_BAD_COORDS, row, archive)
            continue

        # Proxy type
        std_name = str(row.get('paleoData_standardName') or
                       row.get('paleoData_proxy')         or
                       row.get('paleoData_proxyGeneral')  or
                       vname)
        ptype    = create_ptype(archive, std_name)

        # Record ID — already validated as non-empty above; use it directly
        # so the pid matches the presto TSID catalog exactly.
        pid = str(tsid)

        # Multi-compilation membership from pylipd's
        # paleoData_inCompilationBeta: a list of
        #   [{'compilationName': 'iso2k', 'compilationVersion': ['1_1_2', ...]}]
        # dicts. Expand to "<name>-<version>" strings so a record that's in
        # both iso2k 1_1_2 and CoralHydro2k 1_0_0 gets tagged with both.
        comp_beta = row.get('paleoData_inCompilationBeta')
        compilations = []
        if comp_beta:
            try:
                entries = (comp_beta if isinstance(comp_beta, list)
                           else [comp_beta])
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get('compilationName') or '').strip()
                    vers = entry.get('compilationVersion') or []
                    if not isinstance(vers, (list, tuple)):
                        vers = [vers]
                    if name:
                        if vers:
                            for v in vers:
                                compilations.append(f'{name}-{str(v).strip()}')
                        else:
                            compilations.append(name)
            except Exception:
                pass

        df_rows.append({
            'paleoData_pages2kID':    pid,
            'geo_meanLat':            lat,
            'geo_meanLon':            lon,
            'geo_meanElev':           elev,
            'year':                   time_arr,
            'paleoData_values':       val_arr,
            'ptype':                  ptype,
            'paleoData_variableName': vname,
            'paleoData_units':        str(row.get('paleoData_units') or 'unknown'),
            'paleoData_compilations': compilations,
        })
        n_ok += 1

    print(f"\nProxy records: {n_ok} added, {n_skip} skipped (of {len(proxy_rows)} candidates)")

    # ── Skip reason breakdown ────────────────────────────────────────────────
    if skip_records:
        skip_df = pd.DataFrame(skip_records)
        reason_counts = skip_df['reason'].value_counts()
        print("\nSkipped records breakdown by reason:")
        for reason, count in reason_counts.items():
            print(f"  {reason:<50} {count:>4} records")
            arch_counts = skip_df[skip_df['reason'] == reason]['archiveType'].value_counts()
            for arch, cnt in arch_counts.items():
                print(f"    {arch:<48} {cnt:>4}")

    if n_ok == 0:
        raise RuntimeError(
            "No proxy records were added — check paleoData structure and time key names"
        )

    df = pd.DataFrame(df_rows)

    # Ptype breakdown
    ptypes = df['ptype'].value_counts()
    print("\nProxy type breakdown:")
    for pt, cnt in ptypes.items():
        print(f"  {pt:<40} {cnt:>4} records")

    # ── Save CSVs to prepare-data/ directory ─────────────────────────────────
    output_dir = os.path.dirname(output_path)
    prep_dir = os.path.join(output_dir, 'prepare-data')
    os.makedirs(prep_dir, exist_ok=True)

    skip_csv = os.path.join(prep_dir, 'skipped_records.csv')
    if skip_records:
        skip_df.to_csv(skip_csv, index=False)
        print(f"\nSaved {len(skip_records)} skipped records to {skip_csv}")
    else:
        pd.DataFrame(columns=['dataSetName', 'TSID', 'variableName', 'archiveType', 'reason']).to_csv(skip_csv, index=False)
        print(f"\nNo skipped records — saved empty {skip_csv}")

    ptype_csv = os.path.join(prep_dir, 'proxy_type_breakdown.csv')
    ptype_df = pd.DataFrame({'ptype': ptypes.index, 'count': ptypes.values})
    ptype_df.to_csv(ptype_csv, index=False)
    print(f"Saved proxy type breakdown to {ptype_csv}")

    print(f"\nSaving proxy DataFrame to {output_path} ...")
    df.to_pickle(output_path)
    print("Done.")


if __name__ == '__main__':
    main()
