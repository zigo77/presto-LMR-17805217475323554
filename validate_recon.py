"""
Instrumental validation for LMR reconstruction results.

Compares reconstruction GMST and spatial fields against multiple instrumental
datasets (GISTEMP, HadCRUT5) and the published LMRv2.1, computing both
correlation (R) and coefficient of efficiency (CE).

Modeled after the PReSto2k validation notebook:
  LinkedEarth/presto2k_cfr_pb  notebooks/validation/C03_a_validating_PReSto2k.ipynb

Run inside davidedge/lmr2:latest Docker container.
"""

import os
import csv
import json
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import cfr

# ── Configuration ────────────────────────────────────────────────────────────
RECON_DIR    = os.environ.get('RECON_DIR', '/recons')
OUT_DIR      = os.environ.get('VALIDATION_DIR', '/validation')
LMR_V21_PATH = os.environ.get(
    'LMR_V21_PATH', '/reference_data/gmt_MCruns_ensemble_full_LMRv2.1.nc')
VALID_START  = 1880
VALID_END    = 2000
ANOM_PERIOD  = [1951, 1980]
COMPARISON_CSV  = os.environ.get('COMPARISON_CSV', '/app/proxy_db_comparison.csv')
COMPARISON_JSON = os.environ.get('COMPARISON_JSON',
                                  os.path.join(OUT_DIR, 'comparison.json'))

os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def area_weighted_mean(da):
    """Area-weighted spatial mean of a DataArray with lat/lon dims."""
    wgts = np.cos(np.deg2rad(da['lat']))
    return float(da.weighted(wgts).mean(('lat', 'lon')).values)


def ensts_to_1d(ensts):
    """Extract a 1D time series from an EnsTS (uses median across ensemble)."""
    time = np.asarray(ensts.time)
    val = np.asarray(ensts.value)
    if val.ndim == 2:
        val_1d = np.nanmedian(val, axis=1)
    else:
        val_1d = val
    return time, val_1d


def coefficient_of_efficiency(obs, pred):
    """Nash-Sutcliffe coefficient of efficiency (CE).
    CE = 1 - sum((obs-pred)^2) / sum((obs-mean(obs))^2)
    Perfect reconstruction = 1, climatology = 0, worse than climatology < 0.
    """
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 5:
        return float('nan')
    o, p = obs[mask], pred[mask]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def pearson_r(a, b):
    """Pearson correlation between two arrays over their valid (finite) entries."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float('nan')
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def align_series(time_a, val_a, time_b, val_b, ymin, ymax):
    """Align two time series to common integer years within [ymin, ymax].
    Returns (common_years, vals_a_aligned, vals_b_aligned)."""
    years_a = np.asarray(time_a, dtype=int)
    years_b = np.asarray(time_b, dtype=int)
    common = np.intersect1d(years_a, years_b)
    common = common[(common >= ymin) & (common <= ymax)]
    if len(common) == 0:
        return common, np.array([]), np.array([])
    idx_a = np.searchsorted(years_a, common)
    idx_b = np.searchsorted(years_b, common)
    return common, val_a[idx_a], val_b[idx_b]


def fetch_hadcrut5_gmst():
    """Download HadCRUT5 global annual mean temperature anomaly.
    Returns (years, values) as numpy arrays."""
    import urllib.request
    url = ('https://www.metoffice.gov.uk/hadobs/hadcrut5/data/'
           'HadCRUT.5.0.2.0/analysis/diagnostics/'
           'HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv')
    print(f'  Downloading HadCRUT5 from {url} ...')
    try:
        response = urllib.request.urlopen(url, timeout=60)
        lines = response.read().decode('utf-8').strip().split('\n')
    except Exception as e:
        print(f'  WARNING: Failed to download HadCRUT5: {e}')
        return None, None

    # Parse CSV: columns are Time, Anomaly (deg C), ...
    years, vals = [], []
    for line in lines[1:]:  # skip header
        parts = line.split(',')
        try:
            years.append(int(float(parts[0])))
            vals.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    return np.array(years), np.array(vals)


# ═══════════════════════════════════════════════════════════════════════════
# Proxy comparison functions
# ═══════════════════════════════════════════════════════════════════════════

# Standard paleoclimate archive-type colors (keyed by normalized display name)
ARCHIVE_COLORS = {
    'Tree': '#228B22', 'Coral': '#FF6347', 'Ice': '#4169E1',
    'Lake': '#8B4513', 'Marine': '#006400', 'Speleothem': '#9370DB',
    'Borehole': '#FF8C00', 'Documents': '#808080', 'Bivalve': '#DEB887',
    'Sclerosponge': '#20B2AA', 'Hybrid': '#C0C0C0', 'Other': '#999999',
}


def plot_temporal_coverage(rows, out_path):
    """Stacked bar chart of proxy record temporal coverage by archive type."""
    records = []
    for r in rows:
        try:
            t0 = float(r['time_start'])
            t1 = float(r['time_end'])
            archive = r.get('archiveType', 'Other')
            source = r.get('source', '')
            records.append((t0, t1, archive, source))
        except (ValueError, TypeError):
            continue
    if not records:
        return False

    # Determine range, clip outliers
    starts = [r[0] for r in records]
    ends = [r[1] for r in records]
    lo = max(0, np.percentile(starts, 5))
    hi = min(2100, np.percentile(ends, 95))
    if hi <= lo:
        lo, hi = min(starts), max(ends)

    n_bins = 80
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    archive_types = sorted(set(r[2] for r in records))

    counts = {at: np.zeros(n_bins) for at in archive_types}
    for t0, t1, archive, _ in records:
        for i in range(n_bins):
            if t0 <= bin_edges[i + 1] and t1 >= bin_edges[i]:
                counts[archive][i] += 1

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(n_bins)
    for at in archive_types:
        color = ARCHIVE_COLORS.get(at, '#999999')
        ax.bar(bin_centers, counts[at], width=bin_width * 0.95, bottom=bottom,
               label=at, color=color, edgecolor='none')
        bottom += counts[at]

    ax.set_xlabel('Year CE')
    ax.set_ylabel('Number of Records')
    ax.set_title('Temporal Coverage of Proxy Records by Archive Type')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=8, ncol=1, borderaxespad=0, frameon=False)
    ax.set_xlim(lo, hi)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def build_comparison_table(rows):
    """Build HTML table summarizing records by archive type and source."""
    from collections import Counter
    type_source = Counter()
    for r in rows:
        archive = r.get('archiveType', 'Other')
        source = r.get('source', 'unknown')
        type_source[(archive, source)] += 1

    archives = sorted(set(a for a, _ in type_source.keys()))

    html = '<table>\n'
    html += '    <tr><th>Archive Type</th><th>Shared</th>'
    html += '<th>PReSto2k Only</th><th>Custom Only</th><th>Total</th></tr>\n'

    totals = {'both': 0, 'presto2k': 0, 'custom_run': 0}
    for arch in archives:
        b = type_source.get((arch, 'both'), 0)
        p = type_source.get((arch, 'presto2k'), 0)
        c = type_source.get((arch, 'custom_run'), 0)
        totals['both'] += b
        totals['presto2k'] += p
        totals['custom_run'] += c
        html += f'    <tr><td>{arch}</td><td>{b}</td><td>{p}</td>'
        html += f'<td>{c}</td><td>{b + p + c}</td></tr>\n'

    grand = sum(totals.values())
    html += f'    <tr style="font-weight:bold"><td>Total</td>'
    html += f'<td>{totals["both"]}</td><td>{totals["presto2k"]}</td>'
    html += f'<td>{totals["custom_run"]}</td><td>{grand}</td></tr>\n'
    html += '  </table>\n'
    return html, totals


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load reconstruction
# ═══════════════════════════════════════════════════════════════════════════
print(f'Loading reconstruction from {RECON_DIR} ...')
res = cfr.ReconRes(RECON_DIR)
res.load(['tas', 'tas_gm'], verbose=True)

recon_tas = res.recons['tas']      # ClimateField (ensemble-mean spatial field)
recon_gm  = res.recons['tas_gm']   # EnsTS (global mean, full ensemble)

recon_time = np.asarray(recon_gm.time)
recon_val  = np.asarray(recon_gm.value)
if recon_val.ndim == 1:
    recon_val = recon_val.reshape(-1, 1)
recon_median = np.nanmedian(recon_val, axis=1)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Fetch instrumental observations
# ═══════════════════════════════════════════════════════════════════════════
print('Fetching GISTEMP observations ...')
obs = cfr.ClimateField().fetch('gistemp1200_ERSSTv4', vn='tempanomaly')
obs = obs.get_anom(ref_period=ANOM_PERIOD)
obs = obs.annualize(months=list(range(1, 13)))
obs_gm = obs.geo_mean()
obs_time, obs_1d = ensts_to_1d(obs_gm)

# Build GISTEMP EnsTS for cfr.compare()
gis_values = obs_1d[:, np.newaxis] if obs_1d.ndim == 1 else obs_1d
gis_ensts = cfr.EnsTS(time=obs_time, value=gis_values,
                       value_name='Temperature Anomaly')

print('Fetching HadCRUT5 observations ...')
had_time, had_vals = fetch_hadcrut5_gmst()
has_hadcrut = had_time is not None and len(had_time) > 0
if has_hadcrut:
    had_values = had_vals[:, np.newaxis]
    had_ensts = cfr.EnsTS(time=had_time, value=had_values,
                          value_name='Temperature Anomaly')
    print(f'  HadCRUT5: {len(had_time)} years ({had_time.min()}-{had_time.max()})')

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load LMRv2.1 reference
# ═══════════════════════════════════════════════════════════════════════════
lmr_v21_time = None
lmr_v21_median = None
lmr_v21_ensts = None
if os.path.exists(LMR_V21_PATH):
    print(f'Loading published LMRv2.1 GMST from {LMR_V21_PATH} ...')
    lmr_v21 = xr.open_dataset(LMR_V21_PATH)
    gmt = lmr_v21['gmt']
    ens_dims = [d for d in ('MCrun', 'members') if d in gmt.dims]
    if ens_dims:
        gmt_ens = gmt.stack(ensemble=ens_dims)
    else:
        gmt_ens = gmt
    ens_arr = np.asarray(gmt_ens.values)

    raw_time = lmr_v21['time'].values
    try:
        lmr_v21_time = np.array([int(t.year) for t in raw_time])
    except AttributeError:
        lmr_v21_time = np.asarray(raw_time, dtype=float).astype(int)

    lmr_v21_median = np.nanmedian(ens_arr, axis=1)
    lmr_v21_q05    = np.nanquantile(ens_arr, 0.05, axis=1)
    lmr_v21_q95    = np.nanquantile(ens_arr, 0.95, axis=1)
    lmr_v21_ensts  = cfr.EnsTS(time=lmr_v21_time, value=ens_arr,
                                value_name='GMSTa')
    print(f'  LMRv2.1 GMST: {len(lmr_v21_time)} years, '
          f'{ens_arr.shape[1]} ensemble members')
else:
    print(f'WARNING: LMRv2.1 reference not found at {LMR_V21_PATH}')


# ═══════════════════════════════════════════════════════════════════════════
# 4. GMST Instrumental Validation (R and CE)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\nComputing GMST validation metrics ({VALID_START}-{VALID_END}) ...')

# Collect all results: {dataset_name: {recon_name: {R, CE}}}
gmst_results = {}


def compute_gmst_stats(recon_name, recon_t, recon_v, ref_name, ref_t, ref_v):
    """Compute R and CE between two GMST time series."""
    _, ra, rb = align_series(recon_t, recon_v, ref_t, ref_v,
                             VALID_START, VALID_END)
    r_val = pearson_r(ra, rb)
    ce_val = coefficient_of_efficiency(rb, ra)  # obs=ref, pred=recon
    print(f'  {recon_name} vs {ref_name}: R={r_val:.4f}, CE={ce_val:.4f}')
    if ref_name not in gmst_results:
        gmst_results[ref_name] = {}
    gmst_results[ref_name][recon_name] = {'R': r_val, 'CE': ce_val}
    return r_val, ce_val


recon_years = recon_time.astype(int)

# vs GISTEMP
compute_gmst_stats('Custom Recon', recon_years, recon_median,
                   'GISTEMP', obs_time.astype(int), obs_1d)
if lmr_v21_time is not None:
    compute_gmst_stats('LMRv2.1', lmr_v21_time, lmr_v21_median,
                       'GISTEMP', obs_time.astype(int), obs_1d)

# vs HadCRUT5
if has_hadcrut:
    compute_gmst_stats('Custom Recon', recon_years, recon_median,
                       'HadCRUT5', had_time, had_vals)
    if lmr_v21_time is not None:
        compute_gmst_stats('LMRv2.1', lmr_v21_time, lmr_v21_median,
                           'HadCRUT5', had_time, had_vals)

# Consensus (mean of available instrumental datasets)
consensus_refs = [('GISTEMP', obs_time.astype(int), obs_1d)]
if has_hadcrut:
    consensus_refs.append(('HadCRUT5', had_time, had_vals))

if len(consensus_refs) > 1:
    # Align all instrumental datasets to common years
    all_years = consensus_refs[0][1]
    for _, t, _ in consensus_refs[1:]:
        all_years = np.intersect1d(all_years, t)
    all_years = all_years[(all_years >= VALID_START) & (all_years <= VALID_END)]

    if len(all_years) > 10:
        consensus_vals = []
        for _, t, v in consensus_refs:
            idx = np.searchsorted(t.astype(int), all_years)
            consensus_vals.append(v[idx])
        consensus_mean = np.mean(consensus_vals, axis=0)

        compute_gmst_stats('Custom Recon', recon_years, recon_median,
                           'Consensus', all_years, consensus_mean)
        if lmr_v21_time is not None:
            compute_gmst_stats('LMRv2.1', lmr_v21_time, lmr_v21_median,
                               'Consensus', all_years, consensus_mean)

# vs LMRv2.1 (direct recon-to-recon comparison over full overlap)
if lmr_v21_time is not None:
    overlap_start = int(max(recon_years.min(), lmr_v21_time.min()))
    overlap_end   = int(min(recon_years.max(), lmr_v21_time.max()))
    if overlap_end > overlap_start:
        _, ra, rb = align_series(recon_years, recon_median,
                                 lmr_v21_time, lmr_v21_median,
                                 overlap_start, overlap_end)
        lmr_r = pearson_r(ra, rb)
        lmr_ce = coefficient_of_efficiency(rb, ra)
        gmst_results['LMRv2.1 (direct)'] = {
            'Custom Recon': {'R': lmr_r, 'CE': lmr_ce,
                             'period': f'{overlap_start}-{overlap_end}'}
        }
        print(f'  Custom Recon vs LMRv2.1 (full overlap {overlap_start}-{overlap_end}): '
              f'R={lmr_r:.4f}, CE={lmr_ce:.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# 5. Spatial Validation Maps (Correlation + CE)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\nComputing spatial validation maps ({VALID_START}-{VALID_END}) ...')

# Spatial correlation
corr_field = recon_tas.compare(obs, stat='corr', timespan=[VALID_START, VALID_END])
corr_da = corr_field.da
geo_mean_corr = area_weighted_mean(corr_da)
print(f'  Geographic mean correlation: {geo_mean_corr:.4f}')

# Spatial CE
ce_field = recon_tas.compare(obs, stat='CE', timespan=[VALID_START, VALID_END])
ce_da = ce_field.da
geo_mean_ce = area_weighted_mean(ce_da)
print(f'  Geographic mean CE: {geo_mean_ce:.4f}')

# Plot spatial correlation
fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.Robinson()})
corr_da.plot(ax=ax, transform=ccrs.PlateCarree(),
             cmap='RdYlBu_r', vmin=-1, vmax=1,
             cbar_kwargs={'label': 'Correlation (r)',
                          'orientation': 'horizontal',
                          'shrink': 0.7, 'pad': 0.08})
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
ax.set_global()
ax.set_title(f'Reconstruction vs GISTEMP Correlation ({VALID_START}-{VALID_END})\n'
             f'Geographic Mean r = {geo_mean_corr:.3f}', fontsize=13)
fig.savefig(os.path.join(OUT_DIR, 'spatial_corr_map.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Plot spatial CE
fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.Robinson()})
ce_da.plot(ax=ax, transform=ccrs.PlateCarree(),
           cmap='RdYlBu_r', vmin=-1, vmax=1,
           cbar_kwargs={'label': 'Coefficient of Efficiency (CE)',
                        'orientation': 'horizontal',
                        'shrink': 0.7, 'pad': 0.08})
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
ax.set_global()
ax.set_title(f'Reconstruction vs GISTEMP CE ({VALID_START}-{VALID_END})\n'
             f'Geographic Mean CE = {geo_mean_ce:.3f}', fontsize=13)
fig.savefig(os.path.join(OUT_DIR, 'spatial_ce_map.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Combined side-by-side spatial maps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6),
                               subplot_kw={'projection': ccrs.Robinson()})
corr_da.plot(ax=ax1, transform=ccrs.PlateCarree(),
             cmap='RdYlBu_r', vmin=-1, vmax=1,
             cbar_kwargs={'label': 'r', 'orientation': 'horizontal',
                          'shrink': 0.8, 'pad': 0.08})
ax1.coastlines(linewidth=0.5)
ax1.set_global()
ax1.set_title(f'Correlation (mean r = {geo_mean_corr:.3f})')

ce_da.plot(ax=ax2, transform=ccrs.PlateCarree(),
           cmap='RdYlBu_r', vmin=-1, vmax=1,
           cbar_kwargs={'label': 'CE', 'orientation': 'horizontal',
                        'shrink': 0.8, 'pad': 0.08})
ax2.coastlines(linewidth=0.5)
ax2.set_global()
ax2.set_title(f'Coefficient of Efficiency (mean CE = {geo_mean_ce:.3f})')

fig.suptitle(f'Spatial Validation vs GISTEMP ({VALID_START}-{VALID_END})', fontsize=14)
fig.savefig(os.path.join(OUT_DIR, 'spatial_corr_ce_combined.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. GMST Time Series Plot (with ensemble spread)
# ═══════════════════════════════════════════════════════════════════════════
print('Generating GMST time series plot ...')

recon_q05 = np.nanquantile(recon_val, 0.05, axis=1)
recon_q95 = np.nanquantile(recon_val, 0.95, axis=1)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(recon_time, recon_q05, recon_q95,
                alpha=0.3, color='steelblue',
                label='Custom recon (5-95% range)')
ax.plot(recon_time, recon_median, color='steelblue', lw=1.5,
        label='Custom recon (median)')

if lmr_v21_time is not None:
    ax.fill_between(lmr_v21_time, lmr_v21_q05, lmr_v21_q95,
                    alpha=0.25, color='darkorange',
                    label='LMRv2.1 (5-95% range)')
    ax.plot(lmr_v21_time, lmr_v21_median, color='darkorange', lw=1.5,
            label='LMRv2.1 (median)')

ax.plot(obs_time, obs_1d, color='red', lw=1.5, label='GISTEMP', alpha=0.85)

if has_hadcrut:
    ax.plot(had_time, had_vals, color='green', lw=1.5, ls='--',
            label='HadCRUT5', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title('Global Mean Surface Temperature')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=6, frameon=False)
t_min = recon_time.min()
if lmr_v21_time is not None:
    t_min = min(t_min, lmr_v21_time.min())
ax.set_xlim(t_min, 2000)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
fig.savefig(os.path.join(OUT_DIR, 'gmst_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6b. GMST Ensemble Members Plot (all iterations)
# ═══════════════════════════════════════════════════════════════════════════
print('Generating GMST ensemble members plot ...')
n_ens = recon_val.shape[1]

fig, ax = plt.subplots(figsize=(14, 6))

# Plot every ensemble member as a thin translucent line
# Cap at 200 members for readability; if more, subsample evenly
max_lines = 200
if n_ens <= max_lines:
    plot_indices = range(n_ens)
else:
    plot_indices = np.linspace(0, n_ens - 1, max_lines, dtype=int)

for i in plot_indices:
    ax.plot(recon_time, recon_val[:, i], color='steelblue',
            alpha=max(0.03, 3.0 / n_ens), lw=0.4)

# Overlay median and quantiles
ax.fill_between(recon_time, recon_q05, recon_q95,
                alpha=0.15, color='navy', label='5-95% range')
ax.plot(recon_time, recon_median, color='navy', lw=2,
        label='Ensemble median')

# Overlay instrumental
ax.plot(obs_time, obs_1d, color='red', lw=1.5, label='GISTEMP', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title(f'GMST: All {n_ens} Ensemble Members')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=6, frameon=False)
t_min = recon_time.min()
ax.set_xlim(t_min, 2000)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
fig.savefig(os.path.join(OUT_DIR, 'gmst_ensemble_members.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Zoomed instrumental-period version
fig, ax = plt.subplots(figsize=(14, 6))
mask_t = (recon_time >= VALID_START) & (recon_time <= VALID_END)
for i in plot_indices:
    ax.plot(recon_time[mask_t], recon_val[mask_t, i], color='steelblue',
            alpha=max(0.05, 5.0 / n_ens), lw=0.5)
ax.fill_between(recon_time[mask_t], recon_q05[mask_t], recon_q95[mask_t],
                alpha=0.15, color='navy', label='5-95% range')
ax.plot(recon_time[mask_t], recon_median[mask_t], color='navy', lw=2,
        label='Ensemble median')
omask = (obs_time >= VALID_START) & (obs_time <= VALID_END)
ax.plot(obs_time[omask], obs_1d[omask], color='red', lw=2,
        label='GISTEMP', alpha=0.85)
if has_hadcrut:
    hmask = (had_time >= VALID_START) & (had_time <= VALID_END)
    ax.plot(had_time[hmask], had_vals[hmask], color='green', lw=2,
            ls='--', label='HadCRUT5', alpha=0.85)
ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title(f'GMST Ensemble Members: Instrumental Period ({VALID_START}-{VALID_END})')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=6, frameon=False)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, 'gmst_ensemble_members_instrumental.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 7. GMST Difference Plot (recon - LMRv2.1)
# ═══════════════════════════════════════════════════════════════════════════
if lmr_v21_time is not None:
    print('Generating GMST difference plot ...')
    _, recon_aligned, lmr_aligned = align_series(
        recon_years, recon_median, lmr_v21_time, lmr_v21_median,
        int(max(recon_years.min(), lmr_v21_time.min())),
        int(min(recon_years.max(), lmr_v21_time.max())))
    diff_years = np.arange(
        int(max(recon_years.min(), lmr_v21_time.min())),
        int(min(recon_years.max(), lmr_v21_time.max())) + 1)
    diff_years = diff_years[:len(recon_aligned)]
    difference = recon_aligned - lmr_aligned

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(diff_years, 0, difference,
                    where=difference >= 0, color='firebrick', alpha=0.5)
    ax.fill_between(diff_years, 0, difference,
                    where=difference < 0, color='steelblue', alpha=0.5)
    ax.plot(diff_years, difference, color='black', lw=0.5, alpha=0.7)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Year CE')
    ax.set_ylabel('Difference (\u00b0C)')
    ax.set_title('GMST Difference: Custom Reconstruction - LMRv2.1\n'
                 '(Red = warmer than LMRv2.1, Blue = cooler)')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'gmst_difference.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Instrumental Period Detail (1880-2000)
# ═══════════════════════════════════════════════════════════════════════════
print('Generating instrumental period detail plot ...')

fig, ax = plt.subplots(figsize=(14, 5))
mask = (recon_time >= VALID_START) & (recon_time <= VALID_END)
ax.fill_between(recon_time[mask], recon_q05[mask], recon_q95[mask],
                alpha=0.3, color='steelblue',
                label='Custom recon (5-95%)')
ax.plot(recon_time[mask], recon_median[mask], color='steelblue', lw=2,
        label='Custom recon (median)')

if lmr_v21_time is not None:
    lmask = (lmr_v21_time >= VALID_START) & (lmr_v21_time <= VALID_END)
    ax.fill_between(lmr_v21_time[lmask], lmr_v21_q05[lmask],
                    lmr_v21_q95[lmask], alpha=0.2, color='darkorange',
                    label='LMRv2.1 (5-95%)')
    ax.plot(lmr_v21_time[lmask], lmr_v21_median[lmask],
            color='darkorange', lw=2, label='LMRv2.1 (median)')

omask = (obs_time >= VALID_START) & (obs_time <= VALID_END)
ax.plot(obs_time[omask], obs_1d[omask], color='red', lw=2,
        label='GISTEMP', alpha=0.85)

if has_hadcrut:
    hmask = (had_time >= VALID_START) & (had_time <= VALID_END)
    ax.plot(had_time[hmask], had_vals[hmask], color='green', lw=2,
            ls='--', label='HadCRUT5', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title(f'Instrumental Validation Period ({VALID_START}-{VALID_END})')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=6, frameon=False)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, 'gmst_instrumental_detail.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Save metrics CSV
# ═══════════════════════════════════════════════════════════════════════════
metrics_path = os.path.join(OUT_DIR, 'validation_metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['geo_mean_spatial_corr', f'{geo_mean_corr:.4f}'])
    writer.writerow(['geo_mean_spatial_CE', f'{geo_mean_ce:.4f}'])
    for ref_name, recons in gmst_results.items():
        for recon_name, stats in recons.items():
            prefix = f'{recon_name}_vs_{ref_name}'
            writer.writerow([f'{prefix}_R', f'{stats["R"]:.4f}'])
            writer.writerow([f'{prefix}_CE', f'{stats["CE"]:.4f}'])
    writer.writerow(['validation_period', f'{VALID_START}-{VALID_END}'])
    writer.writerow(['anom_ref_period', f'{ANOM_PERIOD[0]}-{ANOM_PERIOD[1]}'])
    writer.writerow(['n_ensemble_members', int(recon_val.shape[1])])

# Also save as JSON for programmatic access
json_metrics = {
    'spatial': {'corr_geo_mean': geo_mean_corr, 'CE_geo_mean': geo_mean_ce},
    'gmst': gmst_results,
    'config': {
        'validation_period': [VALID_START, VALID_END],
        'anom_ref_period': ANOM_PERIOD,
        'n_ensemble_members': int(recon_val.shape[1]),
    }
}
with open(os.path.join(OUT_DIR, 'validation_metrics.json'), 'w') as f:
    json.dump(json_metrics, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Load proxy-database comparison (comparison.json from compare_to_presto2k.py)
# ═══════════════════════════════════════════════════════════════════════════
comparison_html = ''
comparison_data = None

if os.path.exists(COMPARISON_JSON):
    print(f'Loading proxy comparison from {COMPARISON_JSON} ...')
    with open(COMPARISON_JSON) as f:
        comparison_data = json.load(f)
    print(f'  shared={comparison_data["counts"]["shared"]}  '
          f'p2k-only={comparison_data["counts"]["only_presto2k"]}  '
          f'custom-only={comparison_data["counts"]["only_custom"]}')


def _fmt(v, dash='—'):
    return dash if v is None or v == '' else str(v)


def _render_funnel(funnel):
    """Render the 5-stage funnel as metric chips."""
    stages = [
        ('Requested', funnel.get('requested'),
         'TSIDs in query_params.json after removedTsids are excluded'),
        ('In pickle',  funnel.get('in_pickle'),
         'Unique paleoData_TSid rows written by lipd_to_pdb.py'),
        ('After proxy-type filter', funnel.get('after_ptype_filter'),
         'Records retained by cfr.filter_proxydb (archive / proxy-type whitelist)'),
        ('PSM-calibrated', funnel.get('post_psm'),
         'Records for which a proxy system model was fit against the '
         'instrumental period; eligible for data assimilation.'),
        ('Assimilated', funnel.get('assimilated'),
         'Records used to update the Kalman state during reconstruction'),
    ]
    cards = []
    for label, val, tip in stages:
        if val is None:
            continue
        cards.append(f'''
      <div class="metric-card" title="{tip}">
        <div class="value">{val}</div>
        <div class="label">{label}</div>
      </div>''')
    return '<div class="metric-grid">' + ''.join(cards) + '</div>'


def _render_compilation_table(rows):
    body = []
    for r in rows:
        c_csv = r.get('custom_csv')
        p_csv = r.get('presto2k_csv')
        c_link = (f'<a href="{c_csv}" download>⬇ CSV</a>' if c_csv else '—')
        p_link = (f'<a href="{p_csv}" download>⬇ CSV</a>' if p_csv else '—')
        versions = r.get('versions') or []
        if versions:
            tip = 'Versions represented: ' + ', '.join(versions)
            name_cell = (f'<span title="{tip}" style="border-bottom: 1px dotted #999; '
                         f'cursor: help;">{r["compilation"]}</span>')
        else:
            name_cell = r['compilation']
        body.append(
            f'<tr><td>{name_cell}</td>'
            f'<td>{r["custom_count"]}</td><td>{c_link}</td>'
            f'<td>{r["presto2k_count"]}</td><td>{p_link}</td>'
            f'<td>{r["shared"]}</td><td>{r["custom_only"]}</td>'
            f'<td>{r["p2k_only"]}</td></tr>')
    return (
        '<table><tr>'
        '<th>Compilation</th>'
        '<th>Custom</th><th>CSV</th>'
        '<th>PReSto2k</th><th>CSV</th>'
        '<th>Shared</th><th>Custom-only</th><th>PReSto2k-only</th>'
        '</tr>' + ''.join(body) + '</table>'
    )


def _render_archive_table(rows):
    body = []
    ts = {'shared': 0, 'p2k_only': 0, 'custom_only': 0}
    for r in rows:
        ts['shared'] += r['shared']; ts['p2k_only'] += r['p2k_only']
        ts['custom_only'] += r['custom_only']
        body.append(
            f'<tr><td>{r["archive"]}</td><td>{r["shared"]}</td>'
            f'<td>{r["p2k_only"]}</td><td>{r["custom_only"]}</td>'
            f'<td>{r["shared"] + r["p2k_only"] + r["custom_only"]}</td></tr>')
    body.append(
        f'<tr style="font-weight:bold"><td>Total</td><td>{ts["shared"]}</td>'
        f'<td>{ts["p2k_only"]}</td><td>{ts["custom_only"]}</td>'
        f'<td>{sum(ts.values())}</td></tr>')
    return ('<table><tr><th>Archive</th><th>Shared</th>'
            '<th>PReSto2k only</th><th>Custom only</th><th>Total</th></tr>'
            + ''.join(body) + '</table>')


def _render_ptype_table(rows):
    body = []
    for r in rows:
        body.append(
            f'<tr><td>{r["ptype"]}</td><td>{r["shared"]}</td>'
            f'<td>{r["p2k_only"]}</td><td>{r["custom_only"]}</td></tr>')
    return ('<table><tr><th>ptype</th><th>Shared</th>'
            '<th>PReSto2k only</th><th>Custom only</th></tr>'
            + ''.join(body) + '</table>')


def _render_preview_list(items, csv_link=None, total=None):
    if not items:
        return '<p><em>None.</em></p>'
    body = []
    for r in items:
        rng = f'{_fmt(r.get("time_start"))}–{_fmt(r.get("time_end"))}'
        body.append(
            f'<tr><td><code>{r["tsid"]}</code></td>'
            f'<td>{r.get("archive", "")}</td>'
            f'<td>{r.get("ptype", "")}</td>'
            f'<td>{r.get("dataSetName", "")}</td>'
            f'<td>{rng}</td><td>{r.get("n_obs", 0)}</td></tr>')
    table = ('<table><tr><th>TSID</th><th>Archive</th><th>ptype</th>'
             '<th>Dataset</th><th>Years</th><th>n_obs</th></tr>'
             + ''.join(body) + '</table>')
    footer = ''
    if csv_link and total and total > len(items):
        footer = (f'<p><a href="{csv_link}" download>'
                  f'⬇ Download full CSV ({total} rows)</a></p>')
    return table + footer


def _render_stats_table(stats, recon_period):
    c = stats['custom_used']; p = stats['presto2k']
    rp = (f'{recon_period[0]}&ndash;{recon_period[1]} CE'
          if recon_period else 'reconstruction window')
    rows_def = [
        ('Number of records',                    'records', None),
        ('Distinct archive types',               'distinct_archives',
         'Custom counts archives retained after the proxy-type filter '
         '(filter_proxydb_kwargs in lmr_configs.yml); '
         'PReSto2k reports the full published record set.'),
        ('Distinct proxy types',                 'distinct_ptypes',
         'Same caveat as archive types.'),
        ('Earliest record start (Year CE)',      'earliest_start',
         f'Restricted to the reconstruction period {rp}. '
         'Negative values indicate BCE.'),
        ('Latest record end (Year CE)',          'latest_end',
         f'Restricted to the reconstruction period {rp}.'),
        ('Median record length (years)',         'median_record_length',
         f'Measured within the reconstruction period {rp}.'),
        ('Median observations per record',       'median_n_obs',
         f'Observations within the reconstruction period {rp}.'),
    ]
    body = []
    for label, key, tip in rows_def:
        label_html = (f'<span title="{tip}" style="border-bottom: 1px dotted #999; '
                      f'cursor: help;">{label}</span>' if tip else label)
        body.append(f'<tr><td>{label_html}</td>'
                    f'<td>{_fmt(c.get(key))}</td>'
                    f'<td>{_fmt(p.get(key))}</td></tr>')
    return ('<table><tr><th>Statistic</th>'
            '<th>Custom (assimilated + evaluation)</th>'
            '<th>PReSto2k</th></tr>' + ''.join(body) + '</table>')


def _render_dropped_reasons(reasons, csv_link=None):
    if not reasons:
        return ''
    body = []
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        body.append(f'<tr><td>{reason}</td><td>{n}</td></tr>')
    out = ('<table><tr><th>Reason</th><th>Records</th></tr>'
           + ''.join(body) + '</table>')
    if csv_link:
        out += (f'<p><a href="{csv_link}" download>'
                '⬇ Download full list</a></p>')
    return out


if comparison_data:
    c = comparison_data
    arts = c.get('artifacts', {})
    dls = arts.get('downloads', {})

    # Build sub-sections
    funnel_html = _render_funnel(c['funnel'])
    recon_period = c['funnel'].get('recon_period')
    stats_html = _render_stats_table(c['stats'], recon_period)
    comp_html = _render_compilation_table(c['compilation_rows'])
    arch_html = _render_archive_table(c['archive_rows'])
    ptype_html = _render_ptype_table(c['ptype_rows'])
    requested_comps = c.get('requested_compilations') or []

    tc_arc = arts.get('temporal_coverage_archive')
    tc_pt  = arts.get('temporal_coverage_ptype')
    spat   = arts.get('spatial_map')

    toggle_html = ''
    if tc_arc and tc_pt:
        toggle_html = f'''
  <div style="margin: 12px 0;">
    <label style="margin-right: 16px;">
      <input type="radio" name="tc" value="archive" checked>
      Colour by archive type
    </label>
    <label>
      <input type="radio" name="tc" value="ptype">
      Colour by proxy type
    </label>
  </div>
  <img id="tc-img" src="{tc_arc}"
       alt="Temporal coverage by archive type">
  <script>
    document.querySelectorAll('input[name=tc]').forEach(r =>
      r.addEventListener('change', e =>
        document.getElementById('tc-img').src =
          'temporal_coverage_' + e.target.value + '.png'));
  </script>'''
    elif tc_arc:
        toggle_html = f'<img src="{tc_arc}" alt="Temporal coverage">'

    preview_p2k = _render_preview_list(
        c.get('only_presto2k_preview', []),
        csv_link=dls.get('only_presto2k'),
        total=c['counts']['only_presto2k'])
    preview_custom = _render_preview_list(
        c.get('only_custom_preview', []),
        csv_link=dls.get('only_custom'),
        total=c['counts']['only_custom'])

    dropped_html = _render_dropped_reasons(
        c.get('dropped_reasons', {}),
        csv_link=dls.get('dropped_records'))

    comparison_html = f'''
  <details class="section">
    <summary style="font-size: 1.3rem; font-weight: 600; cursor: pointer;
                    color: #374151; padding: 8px 0;">
      Proxy Database Comparison vs PReSto2k
      <span style="font-weight: 400; color: #6b7280; font-size: 0.95rem;">
        (shared {c["counts"]["shared"]}, custom-only {c["counts"]["only_custom"]},
         PReSto2k-only {c["counts"]["only_presto2k"]})
      </span>
    </summary>

    <div style="padding-top: 16px;">
      <p>Comparison of the proxy records used in this reconstruction
         (those assimilated into the Kalman filter together with those
         withheld for evaluation) against the published
         <strong>PReSto2k</strong> reference database, matched on
         <code>paleoData_TSid</code>.</p>

      <h3>Record-selection funnel</h3>
      <p>Attrition of records from presto's initial TSID request through
         pipeline filtering, PSM calibration, and final assimilation.</p>
      {funnel_html}

      <h3>Side-by-side statistics</h3>
      {stats_html}

      <h3>Compilation membership</h3>
      <p>Records counted by compilation, aggregated across all versions
         (hover the compilation name to see the versions present).
         A record is counted under every compilation its LiPD metadata
         claims membership of, so a record listed in both
         <em>iso2k</em> and <em>CoralHydro2k</em> is counted in both
         rows.</p>
      <p><strong>Source of membership data.</strong> Custom-run
         memberships come from each record's
         <code>paleoData_inCompilationBeta</code> field, extracted in
         <code>lipd_to_pdb.py</code>. PReSto2k records do not carry that
         field in <code>presto2k_pdb.pkl</code>; their memberships fall
         back to lipdverse's single
         <code>paleoData_mostRecentCompilations</code> tag (one
         compilation per record), so the PReSto2k counts are a lower
         bound. Records with no membership land in the <em>(none)</em>
         bucket.</p>
      <p><strong>A zero count is not a bug.</strong> Each compilation
         curates its own set of paleoclimate records, often with
         different <code>paleoData_TSid</code> namespaces. If presto's
         <code>tsids</code> request happens not to intersect a given
         compilation's records (e.g. the user requests iso2k corals,
         which are physically distinct samples from CoralHydro2k's
         corals), that compilation will show 0 even though the request
         listed it under <code>compilation</code> in
         <code>query_params.json</code>.</p>
      {f"<p><strong>Compilations requested in query_params.json:</strong> <code>{', '.join(requested_comps)}</code></p>" if requested_comps else ''}
      {comp_html}

      <h3>Records by archive type</h3>
      {arch_html}

      <details>
        <summary>Detailed breakdown by proxy type</summary>
        {ptype_html}
      </details>

      <h3>Temporal coverage</h3>
      <p>Record counts by year, partitioned by source (custom run on top,
         PReSto2k below). Switch between archive-type and proxy-type
         (<code>ptype</code>) colouring.</p>
      {toggle_html}

      {f'<h3>Spatial distribution</h3><img src="{spat}" alt="Spatial maps">'
        if spat else ''}

      <h3>Records exclusive to PReSto2k <small>({c["counts"]["only_presto2k"]})</small></h3>
      <p>Records present in the PReSto2k reference but absent from the
         custom reconstruction (for example, archives excluded by
         <code>filter_proxydb_kwargs</code>, or TSIDs not included in the
         custom query).</p>
      {preview_p2k}

      <h3>Records exclusive to the custom run <small>({c["counts"]["only_custom"]})</small></h3>
      <p>Records requested in this reconstruction that do not appear in
         the PReSto2k reference.</p>
      {preview_custom}

      {'<h3>Records discarded during processing</h3>' + dropped_html if dropped_html else ''}
    </div>
  </details>
'''
    print('  Comparison HTML section built (comparison.json)')

elif os.path.exists(COMPARISON_CSV):
    print(f'Legacy comparison CSV found at {COMPARISON_CSV} — '
          f'comparison.json preferred; rendering minimal block.')
    with open(COMPARISON_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    archive_table_html, tot = build_comparison_table(rows)
    comparison_html = (
        f'<h2>Proxy Database Comparison (legacy)</h2>{archive_table_html}')
else:
    print('No proxy comparison data found — skipping comparison section')


# ═══════════════════════════════════════════════════════════════════════════
# 11. Generate HTML report
# ═══════════════════════════════════════════════════════════════════════════
print('Generating HTML report ...')

# Build GMST results table rows
table_rows = ''
for ref_name in ['GISTEMP', 'HadCRUT5', 'Consensus']:
    if ref_name not in gmst_results:
        continue
    for recon_name in ['Custom Recon', 'LMRv2.1']:
        if recon_name not in gmst_results[ref_name]:
            continue
        stats = gmst_results[ref_name][recon_name]
        r_val = stats['R']
        ce_val = stats['CE']
        # Color CE: green if > 0.5, orange if > 0, red if negative
        if ce_val > 0.5:
            ce_color = '#16a34a'
        elif ce_val > 0:
            ce_color = '#d97706'
        else:
            ce_color = '#dc2626'
        chip_class = 'chip-custom' if recon_name == 'Custom Recon' else 'chip-lmrv21'
        label_class = 'label-custom' if recon_name == 'Custom Recon' else 'label-lmrv21'
        table_rows += f'''    <tr>
      <td><span class="chip {chip_class}"></span><span class="{label_class}">{recon_name}</span></td>
      <td>{ref_name}</td>
      <td>{r_val:.4f}</td>
      <td style="color: {ce_color}; font-weight: 600;">{ce_val:.4f}</td>
    </tr>\n'''

# Direct vs LMRv2.1 row
lmr_direct_row = ''
if 'LMRv2.1 (direct)' in gmst_results:
    d = gmst_results['LMRv2.1 (direct)']['Custom Recon']
    period = d.get('period', '')
    lmr_direct_row = f'''    <tr>
      <td><span class="chip chip-custom"></span><span class="label-custom">Custom Recon</span></td>
      <td><span class="chip chip-lmrv21"></span><span class="label-lmrv21">LMRv2.1 ({period})</span></td>
      <td>{d["R"]:.4f}</td>
      <td>{d["CE"]:.4f}</td>
    </tr>'''

# Determine if we have the difference plot
has_diff_plot = lmr_v21_time is not None

html = f"""<!DOCTYPE html>
<html>
<head>
  <title>LMR Instrumental Validation</title>
  <style>
    :root {{
      --custom: #4682b4;
      --lmrv21: #ff8c00;
      --gistemp: #dc2626;
      --hadcrut: #16a34a;
      --bg: #f7f8fa;
    }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 1100px; margin: 0 auto; padding: 24px; color: #1a1a1a;
           background: var(--bg); }}
    h1 {{ border-bottom: 3px solid var(--custom); padding-bottom: 12px; font-size: 1.8rem; }}
    h2 {{ color: #374151; margin-top: 36px; font-size: 1.3rem;
          border-left: 4px solid var(--custom); padding-left: 12px; }}
    p {{ line-height: 1.6; color: #4b5563; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%;
             background: white; border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    th, td {{ border: 1px solid #e5e7eb; padding: 10px 16px; text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 600; font-size: 0.9rem;
          text-transform: uppercase; letter-spacing: 0.03em; color: #6b7280; }}
    img {{ max-width: 100%; margin: 12px 0; border: 1px solid #e5e7eb;
           border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    .back {{ margin-top: 32px; }}
    .label-custom  {{ color: var(--custom);  font-weight: 600; }}
    .label-lmrv21  {{ color: var(--lmrv21);  font-weight: 600; }}
    .label-gistemp {{ color: var(--gistemp); font-weight: 600; }}
    .label-hadcrut {{ color: var(--hadcrut); font-weight: 600; }}
    .chip {{
      display: inline-block; width: 0.75em; height: 0.75em;
      border-radius: 2px; margin-right: 6px; vertical-align: baseline;
    }}
    .chip-custom  {{ background: var(--custom); }}
    .chip-lmrv21  {{ background: var(--lmrv21); }}
    .chip-gistemp {{ background: var(--gistemp); }}
    .chip-hadcrut {{ background: var(--hadcrut); }}
    .section {{ background: white; padding: 24px; border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 20px 0; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px; margin: 16px 0; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
    .metric-card .value {{ font-size: 2rem; font-weight: 700; color: var(--custom); }}
    .metric-card .label {{ font-size: 0.85rem; color: #6b7280; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Instrumental Validation Report</h1>
  <p>Validation of the custom LMR reconstruction against instrumental observations
     and the published LMRv2.1, following the methodology of
     <a href="https://github.com/LinkedEarth/presto2k_cfr_pb/blob/main/notebooks/validation/C03_a_validating_PReSto2k.ipynb">PReSto2k validation</a>.</p>

  <div class="metric-grid">
    <div class="metric-card">
      <div class="value">{geo_mean_corr:.3f}</div>
      <div class="label">Spatial Correlation (geo. mean)</div>
    </div>
    <div class="metric-card">
      <div class="value">{geo_mean_ce:.3f}</div>
      <div class="label">Spatial CE (geo. mean)</div>
    </div>
    <div class="metric-card">
      <div class="value">{gmst_results.get('GISTEMP', {}).get('Custom Recon', {}).get('R', float('nan')):.3f}</div>
      <div class="label">GMST R vs GISTEMP</div>
    </div>
    <div class="metric-card">
      <div class="value">{gmst_results.get('GISTEMP', {}).get('Custom Recon', {}).get('CE', float('nan')):.3f}</div>
      <div class="label">GMST CE vs GISTEMP</div>
    </div>
  </div>

  <h2>GMST Validation Metrics ({VALID_START}&ndash;{VALID_END})</h2>
  <p>Performance of the ensemble-median global mean surface temperature
     (GMST) against instrumental datasets over the validation window.
     Two metrics are reported for each reconstruction/reference pair.</p>
  <p><strong>Pearson correlation</strong> (<em>R</em>): linear association
     between the two time series. <em>R</em> measures pattern agreement
     but is insensitive to systematic offsets or amplitude errors.</p>
  <p><strong>Nash&ndash;Sutcliffe Coefficient of Efficiency</strong>
     (<em>CE</em>; Nash &amp; Sutcliffe, 1970):
     <br>&nbsp;&nbsp;<em>CE</em> = 1 &minus; &Sigma;(<em>y</em> &minus;
     <em>&#374;</em>)<sup>2</sup> &nbsp;/&nbsp;
     &Sigma;(<em>y</em> &minus; <em>&#562;</em>)<sup>2</sup>,
     <br>where <em>y</em> is the observation, <em>&#374;</em> is the
     reconstruction, and <em>&#562;</em> is the mean of the observations
     over the validation window.
     <em>CE</em> captures pattern, amplitude, and bias simultaneously:
     <ul>
       <li><em>CE</em> = 1 &mdash; perfect reconstruction; residual
           variance is zero.</li>
       <li><em>CE</em> = 0 &mdash; the reconstruction has the same
           predictive skill as simply using the observed mean; its
           residual variance equals the variance of the observations.</li>
       <li><em>CE</em> &lt; 0 &mdash; the observed mean would be a better
           predictor than the reconstruction.</li>
     </ul>
     <em>CE</em> is sensitive to outliers, so a single large miss can
     dominate it.</p>
  <p><strong>Reference datasets</strong>:
     <span class="label-gistemp">GISTEMP</span> (NASA GISS surface
     temperature analysis, ERSSTv4 ocean;
     <a href="https://data.giss.nasa.gov/gistemp/">data.giss.nasa.gov</a>);
     <span class="label-hadcrut">HadCRUT5</span> (Met Office Hadley
     Centre / CRU analysis;
     <a href="https://www.metoffice.gov.uk/hadobs/hadcrut5/">metoffice.gov.uk</a>);
     <em>Consensus</em> &mdash; the arithmetic mean of the instrumental
     datasets listed above, taken over the years where all inputs have
     data. This is not an authoritative product; it is a local summary
     used here to smooth between-dataset differences when evaluating
     the reconstruction.</p>
  <table>
    <tr><th>Reconstruction</th><th>Reference</th><th>R</th><th>CE</th></tr>
{table_rows}{lmr_direct_row}
  </table>

  <h2>Spatial Validation vs GISTEMP</h2>
  <p>Grid-point <em>R</em> and <em>CE</em> between the
     <span class="label-custom">custom reconstruction</span> and
     <span class="label-gistemp">GISTEMP</span> over {VALID_START}&ndash;{VALID_END}.
     Each cell's score is computed against its own 1951&ndash;1980
     climatology (for anomalies) and its own time-series mean over the
     validation window (for the <em>CE</em> denominator); the geographic
     mean reported with each map is area-weighted.</p>
  <p><strong>Why the geographic-mean <em>CE</em> is so much lower than
     the GMST <em>CE</em></strong>. The two numbers measure different
     things. At any single grid cell, most of the observed variance is
     local &mdash; weather noise, regional modes (ENSO, NAO, AMO, PDO)
     &mdash; which a paleoclimate reconstruction does not resolve. The
     reconstruction primarily captures the large-scale, slowly varying
     forced signal (volcanic, solar, anthropogenic), which is a small
     fraction of local variance. So per-cell <em>CE</em> is typically
     low or negative across much of the globe. When those cells are
     averaged into the global mean, the local noise averages towards
     zero and the forced signal dominates the residual variance, so the
     global-mean <em>CE</em> ends up much higher. In other words, the
     geographic mean of per-cell <em>CE</em> is <strong>not</strong> the
     <em>CE</em> of the geographic-mean time series &mdash; these are
     two different statistics and should not be expected to match.</p>
  <img src="spatial_corr_ce_combined.png" alt="Spatial correlation and CE maps">

  <details>
    <summary>Individual maps</summary>
    <img src="spatial_corr_map.png" alt="Spatial correlation map">
    <img src="spatial_ce_map.png" alt="Spatial CE map">
  </details>

  <h2>GMST Time Series</h2>
  <p><span class="label-custom">Custom reconstruction</span> ensemble spread
     compared against <span class="label-lmrv21">LMRv2.1</span>,
     <span class="label-gistemp">GISTEMP</span>{', and <span class="label-hadcrut">HadCRUT5</span>' if has_hadcrut else ''}.</p>
  <img src="gmst_timeseries.png" alt="GMST time series">

  <h2>GMST Ensemble Members ({n_ens} total)</h2>
  <p>Every ensemble member plotted individually, showing the full spread
     of the reconstruction across all iterations and seeds.</p>
  <img src="gmst_ensemble_members.png" alt="GMST all ensemble members">

  <h3>Instrumental Period ({VALID_START}-{VALID_END})</h3>
  <p>Zoomed view of ensemble members during the instrumental overlap period,
     with <span class="label-gistemp">GISTEMP</span>{' and <span class="label-hadcrut">HadCRUT5</span>' if has_hadcrut else ''}
     overlaid.</p>
  <img src="gmst_ensemble_members_instrumental.png" alt="Ensemble members instrumental period">

  <h2>Instrumental Period Detail</h2>
  <p>Zoomed view of the validation period ({VALID_START}-{VALID_END}) with
     ensemble spread and all reference datasets.</p>
  <img src="gmst_instrumental_detail.png" alt="Instrumental detail">

  {'<h2>GMST Difference (Custom - LMRv2.1)</h2>' if has_diff_plot else ''}
  {'<p>Year-by-year difference between the custom reconstruction and LMRv2.1 ensemble medians. Red = warmer, Blue = cooler.</p>' if has_diff_plot else ''}
  {'<img src="gmst_difference.png" alt="GMST difference plot">' if has_diff_plot else ''}

  {comparison_html}

  <p class="back"><a href="../index.html">&larr; Back to results</a></p>
</body>
</html>"""

with open(os.path.join(OUT_DIR, 'index.html'), 'w') as f:
    f.write(html)

print(f'\nValidation complete. Outputs in {OUT_DIR}/')
print(f'  Plots: spatial_corr_map.png, spatial_ce_map.png, '
      f'spatial_corr_ce_combined.png')
print(f'  Plots: gmst_timeseries.png, gmst_instrumental_detail.png'
      f'{", gmst_difference.png" if has_diff_plot else ""}')
if comparison_data:
    print(f'  Comparison: temporal_coverage_archive.png, '
          f'temporal_coverage_ptype.png, spatial_map_comparison.png')
    print(f'              downloads/used_vs_presto2k.csv + '
          f'compilation CSVs')
print(f'  Data:  validation_metrics.csv, validation_metrics.json')
print(f'  HTML:  index.html')
