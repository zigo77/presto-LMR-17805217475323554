#!/usr/bin/env python3
"""
Convert LiPD zip archive to cfr.ProxyDatabase pickle.

Uses pylipd's built-in get_timeseries() to extract proxy records from the
RDF graph (avoids writing custom SPARQL with fragile property paths), then
maps the flat time-series dicts to cfr.ProxyRecord objects.

Usage:
    python lipd_to_pdb.py <lipd_files.zip> <output_lipd_cfr.pkl>
"""

import sys
import os
import re
import math
import zipfile
import tempfile

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
    print(f"Input:  {zip_path}")
    print(f"Output: {output_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUnzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        n_files = sum(1 for f in os.listdir(tmpdir) if f.endswith('.lpd'))
        print(f"Extracted {n_files} .lpd files")

        print("\nLoading with pylipd ...")
        L = LiPD()
        L.load_from_dir(tmpdir)
        all_ds = L.get_all_dataset_names()
        print(f"Loaded {len(all_ds)} datasets")

        print("\nExtracting time series via pylipd get_timeseries() ...")
        result = L.get_timeseries(all_ds)

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

        # Print available keys from first row for diagnostics
        if rows:
            sample_keys = [k for k in rows[0].keys()
                           if not str(rows[0].get(k, '')).startswith('[')]
            print(f"Sample row keys: {sample_keys[:30]}")

    # ── Build DataFrame (cfr.ProxyDatabase.fetch expects pd.read_pickle → from_df) ──
    df_rows = []
    n_ok    = 0
    n_skip  = 0

    for row in rows:
        vname = str(row.get('paleoData_variableName') or '').strip()

        # Skip time/depth/metadata variables
        if not vname or _is_time_var(vname) or _is_skip_var(vname):
            n_skip += 1
            continue

        # Proxy values
        val_arr = _to_float_array(row.get('paleoData_values'))
        if val_arr is None:
            n_skip += 1
            continue

        # Time axis
        time_raw, time_vname = _get_time_from_row(row)
        if time_raw is None:
            n_skip += 1
            continue
        time_arr = time_to_year_ce(time_raw, time_vname,
                                    row.get('time_standardName', ''))

        # Align, remove NaNs, sort ascending
        n = min(len(time_arr), len(val_arr))
        time_arr, val_arr = time_arr[:n], val_arr[:n]
        mask = np.isfinite(time_arr) & np.isfinite(val_arr)
        if not mask.any():
            n_skip += 1
            continue
        time_arr, val_arr = time_arr[mask], val_arr[mask]
        idx = np.argsort(time_arr)
        time_arr, val_arr = time_arr[idx], val_arr[idx]

        # Skip constant-value records: OLS slope=0 → varye=0, MSE=0 → ob_err=0
        # → kdenom=0 → EnKF Kalman gain blows up
        if np.std(val_arr) < 1e-6:
            n_skip += 1
            continue

        # Coordinates
        try:
            lat  = _get_scalar(row, 'geo_meanLat',  'geo_meanLatitude',  'latitude')
            lon  = _get_scalar(row, 'geo_meanLon',  'geo_meanLongitude', 'longitude')
            elev = _get_scalar(row, 'geo_meanElev', 'geo_meanElevation', 'elevation')
        except Exception:
            n_skip += 1
            continue

        # Proxy type
        std_name = str(row.get('paleoData_standardName') or
                       row.get('paleoData_proxy')         or
                       row.get('paleoData_proxyGeneral')  or
                       vname)
        archive  = str(row.get('archiveType') or row.get('archive') or '')
        ptype    = create_ptype(archive, std_name)

        # Record ID
        pid = str(row.get('TSID') or row.get('paleoData_TSID') or
                  row.get('dataSetName') or f'record_{n_ok}')

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
        })
        n_ok += 1

    print(f"\nProxy records: {n_ok} added, {n_skip} skipped")

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

    print(f"\nSaving proxy DataFrame to {output_path} ...")
    df.to_pickle(output_path)
    print("Done.")


if __name__ == '__main__':
    main()
