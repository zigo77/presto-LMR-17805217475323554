"""Regenerate README.md for a custom PReSto LMR reconstruction.

Reads `query_params.json`, `lmr_configs.yml`, and (if present)
`cleaning_report.json` and `README_NOTES.md` from the repo root, and writes
a `README.md` that surfaces the run-specific data selection, data-cleaning
summary, reconstruction parameters, and any user-authored notes.

`README_NOTES.md` is the author's escape hatch — its contents are inserted
verbatim near the top of the README and survive regenerations. Everything
else in the README is a pure function of the input files, so re-running
with unchanged inputs is byte-stable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


TEMPLATE_REPO = "https://github.com/DaveEdge1/LMR2"
PRESTO_URL = "https://paleopresto.com"

MODE_DESCRIPTIONS = {
    "filtered": (
        "filtered (records hand-selected via PReSto's interactive map "
        "interface and queried from LiPDverse)"
    ),
    "archived": (
        "archived (a pre-built compilation pickle was downloaded directly "
        "from LiPDverse)"
    ),
    "traditional": (
        "traditional (a pre-generated proxy database pickle was downloaded "
        "from a provided URL)"
    ),
}


def _format_compilations(raw):
    """'CoralHydro2k-1_0_0,Pages2k' → 'CoralHydro2k 1.0.0, Pages2k'."""
    if not raw:
        return None
    out = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            name, _, version = token.partition("-")
            out.append(f"{name.strip()} {version.replace('_', '.').strip()}")
        else:
            out.append(token)
    return ", ".join(out)


def _year_value(year):
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def _format_year(year):
    y = _year_value(year)
    if y is None:
        return str(year)
    if y < 0:
        return f"{abs(y)} BCE"
    return f"{y} CE"


def _format_period(period):
    """Render a [start, end] range, suppressing the redundant era suffix
    on the start year when both endpoints share an era."""
    if not period or len(period) < 2:
        return None
    a, b = _year_value(period[0]), _year_value(period[1])
    if a is not None and b is not None and ((a >= 0) == (b >= 0)):
        start = f"{abs(a)}" if a < 0 else f"{a}"
        return f"{start}–{_format_year(b)}"
    return f"{_format_year(period[0])}–{_format_year(period[1])}"


def _format_locrad(km):
    if km is None:
        return "—"
    try:
        return f"{int(km):,} km"
    except (TypeError, ValueError):
        return str(km)


def _format_archives(value):
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def summarize_cleaning(report):
    """Aggregate a PReSto cleaning_report.json (list of duplicate groups).

    Returns a dict with `groups`, `considered`, `kept`, `removed`, and
    `top_reason` (string or None) — or None if the report is unrecognized.
    """
    if not isinstance(report, list) or not report:
        return None
    groups = len(report)
    considered = kept = removed = 0
    removals_by_reason = {}
    for group in report:
        if not isinstance(group, dict):
            continue
        records = group.get("records") or []
        considered += len(records)
        n_removed_here = 0
        for rec in records:
            decision = (rec.get("decision") or "").strip().lower()
            if decision == "keep":
                kept += 1
            elif decision == "remove":
                removed += 1
                n_removed_here += 1
        if n_removed_here:
            note = (group.get("notes") or "uncategorized").strip() or "uncategorized"
            removals_by_reason[note] = removals_by_reason.get(note, 0) + n_removed_here

    top_reason = None
    if removals_by_reason and removed:
        note, count = max(removals_by_reason.items(), key=lambda kv: kv[1])
        if count / removed >= 0.5:
            top_reason = (note, count)

    return {
        "groups": groups,
        "considered": considered,
        "kept": kept,
        "removed": removed,
        "top_reason": top_reason,
    }


def _cleaning_bullet(summary):
    if not summary or not summary["considered"]:
        return None
    parts = [
        f"{summary['considered']} records reviewed across "
        f"{summary['groups']} duplicate-detection groups; "
        f"{summary['removed']} removed"
    ]
    if summary["top_reason"]:
        note, count = summary["top_reason"]
        for prefix in ("removed by ", "removed "):
            if note.lower().startswith(prefix):
                note = note[len(prefix):]
                break
        parts.append(f" (predominantly *{note}* — {count} of {summary['removed']})")
    parts.append(
        ". See [`cleaning_report.json`](cleaning_report.json) for per-record decisions."
    )
    return (
        "**Data cleaning ([PReSto data-cleaning app]"
        "(https://paleopresto.com)):** " + "".join(parts)
    )


def build_readme(query, configs, *, cleaning_report=None,
                 user_notes=None, pages_url=None, releases_url=None):
    mode = (query.get("mode") or "").strip().lower()
    mode_desc = MODE_DESCRIPTIONS.get(mode, mode or "—")

    compilations = _format_compilations(query.get("compilation"))
    archive_types = _format_archives(query.get("archiveTypes"))
    interp_var = query.get("interpVars") or query.get("variableName") or "temperature"

    tsids = query.get("tsids") or []
    removed_tsids = query.get("removedTsids") or []

    recon_period = _format_period(configs.get("recon_period")) or "—"
    anom_period = _format_period(configs.get("prior_anom_period")) or "—"
    nens = configs.get("nens", "—")
    seeds = configs.get("recon_seeds") or []
    assim_frac = configs.get("assim_frac", "—")
    loc_rad = _format_locrad(configs.get("recon_loc_rad"))
    months = configs.get("prior_annualize_months") or []
    annualize_label = (
        "annual mean (Jan–Dec)"
        if sorted(int(m) for m in months) == list(range(1, 13))
        else f"months {', '.join(str(m) for m in months)}"
    )

    ptype_block = configs.get("filter_proxydb_kwargs") or {}
    ptype_keys = ptype_block.get("keys") or []

    lines = []
    lines.append("# Custom LMR Reconstruction")
    lines.append("")
    lines.append(
        f"This repository was generated by [PReSto]({PRESTO_URL}) "
        "(Paleoclimate Reconstruction Storehouse) from the "
        f"[LMR Template]({TEMPLATE_REPO}). It runs a Last Millennium "
        "Reanalysis ([Tardif et al., 2019]"
        "(https://doi.org/10.5194/cp-15-1251-2019)) reconstruction over "
        "the parameters and proxy selection captured below."
    )
    lines.append("")

    notes_text = (user_notes or "").strip()
    if notes_text:
        lines.append(notes_text)
        lines.append("")

    lines.append("## Reconstruction parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Reconstruction window | {recon_period} |")
    lines.append(f"| Anomaly reference period | {anom_period} |")
    lines.append(f"| Calibration target | {interp_var} |")
    lines.append(f"| Prior averaging | {annualize_label} |")
    lines.append(f"| Ensemble size | {nens} members |")
    if not seeds:
        seeds_label = "—"
    elif len(seeds) == 1:
        seeds_label = str(seeds[0])
    else:
        seeds_label = f"{', '.join(str(s) for s in seeds)} ({len(seeds)} seeds)"
    lines.append(f"| Seeds | {seeds_label} |")
    lines.append(f"| Assimilation fraction | {assim_frac} |")
    lines.append(f"| Localization radius | {loc_rad} |")
    lines.append("")
    lines.append("(See `lmr_configs.yml` for the authoritative settings.)")
    lines.append("")

    lines.append("## Proxy data selection")
    lines.append("")
    lines.append(f"- **Mode:** {mode_desc}")
    if compilations:
        lines.append(f"- **Source compilations:** {compilations}")
    if archive_types:
        lines.append(f"- **Archive types requested:** {archive_types}")
    if ptype_keys:
        lines.append(
            "- **Proxy-type whitelist (post-load):** "
            f"{', '.join(ptype_keys)}"
        )
    if tsids:
        lines.append(f"- **Records selected:** {len(tsids)}")
    if removed_tsids:
        lines.append(f"- **Records explicitly excluded:** {len(removed_tsids)}")
    cleaning_summary = summarize_cleaning(cleaning_report) if cleaning_report else None
    cleaning_bullet = _cleaning_bullet(cleaning_summary)
    if cleaning_bullet:
        lines.append(f"- {cleaning_bullet}")
    if pages_url:
        validation_url = pages_url.rstrip("/") + "/validation/index.html"
        lines.append(
            "- **Comparison vs PReSto2k:** see the *Proxy Database vs "
            f"PReSto2k* section of the [validation page]({validation_url}) "
            "for the per-compilation overlap, archive/ptype breakdown, and "
            "spatial/temporal coverage relative to the PReSto2k reference."
        )
    lines.append("")
    lines.append("(See `query_params.json` for the full TSID list.)")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "- Reconstruction NetCDFs are committed to `recons/` after each "
        "successful run."
    )
    if pages_url:
        lines.append(
            "- Validation page (vs HadCRUT5 and GISTEMP, with a proxy "
            f"comparison vs PReSto2k) and the interactive visualization: "
            f"<{pages_url}>"
        )
    else:
        lines.append(
            "- A validation page (against HadCRUT5 and GISTEMP, with a "
            "proxy comparison vs PReSto2k) and the interactive "
            "visualization are deployed to GitHub Pages — see this "
            "repository's **Settings → Pages** for the deployed URL."
        )
    lines.append("")

    if releases_url:
        lines.append("## Citation & archive")
        lines.append("")
        lines.append(
            "Each successful reconstruction is bundled into a tagged "
            f"GitHub Release named `recon-<run_id>` at <{releases_url}>. "
            "The release preserves the recon NetCDFs, the proxy database "
            "that was assimilated (`lipd_cfr.pkl`), the validation page, "
            "and the input configs — these survive beyond the 90-day "
            "Actions artifact retention so the run remains fully "
            "auditable and reproducible long after the workflow logs "
            "expire."
        )
        lines.append("")
        lines.append(
            "If [GitHub–Zenodo integration]"
            "(https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) "
            "is enabled on this repository, each release also receives a "
            "citable DOI, with a stable concept DOI for the "
            "reconstruction series as a whole."
        )
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "This reconstruction reproduces the offline data-assimilation "
        "method of [Hakim et al. (2016)]"
        "(https://doi.org/10.1002/2016JD024751) using the "
        "[cfr](https://fzhu2e.github.io/cfr/) Python package "
        "([Zhu et al., 2024]"
        "(https://doi.org/10.5194/gmd-17-3409-2024)). Proxies are sourced "
        "from [LiPDverse](https://lipdverse.org)."
    )
    lines.append("")

    lines.append("## Acknowledgements")
    lines.append("")
    lines.append(
        f"Built from the [PReSto LMR Template]({TEMPLATE_REPO}) by "
        "[David Edge](https://orcid.org/0000-0001-6938-2850), "
        "[Deborah Khider](https://orcid.org/0000-0001-7501-8430), "
        "[Nick McKay](https://orcid.org/0000-0003-3598-5113), "
        "[Tanaya Gondhalekar](https://orcid.org/0009-0004-2440-3266), & "
        "[Julien Emile-Geay](https://orcid.org/0000-0001-5920-4751). "
        f"Hosted by [PReSto]({PRESTO_URL})."
    )
    lines.append("")

    lines.append("---")
    lines.append(
        "*This README is regenerated automatically by "
        "`scripts/generate_readme.py` from `query_params.json`, "
        "`lmr_configs.yml`, and (if present) `cleaning_report.json`. "
        "Hand edits to this file will be overwritten on the next run — "
        "to add commentary that survives regenerations, write it in "
        "`README_NOTES.md` (created at repo root), where it will appear "
        "verbatim near the top of this page.*"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="query_params.json",
                        help="Path to query_params.json (default: %(default)s)")
    parser.add_argument("--configs", default="lmr_configs.yml",
                        help="Path to lmr_configs.yml (default: %(default)s)")
    parser.add_argument("--cleaning-report", default="cleaning_report.json",
                        help="Path to cleaning_report.json; silently "
                             "skipped if missing (default: %(default)s)")
    parser.add_argument("--notes", default="README_NOTES.md",
                        help="Path to user-authored notes file inserted "
                             "verbatim near the top of the README; "
                             "silently skipped if missing "
                             "(default: %(default)s)")
    parser.add_argument("--pages-url",
                        help="Public GitHub Pages URL for this repo, "
                             "linked from the Results section.")
    parser.add_argument("--releases-url",
                        help="GitHub Releases URL for this repo, linked "
                             "from the Citation & archive section. When "
                             "omitted the section is suppressed.")
    parser.add_argument("--out", default="README.md",
                        help="Output README path (default: %(default)s)")
    args = parser.parse_args()

    query_path = Path(args.query)
    configs_path = Path(args.configs)

    if not query_path.exists():
        print(f"ERROR: {query_path} not found", file=sys.stderr)
        return 1
    if not configs_path.exists():
        print(f"ERROR: {configs_path} not found", file=sys.stderr)
        return 1

    with query_path.open("r", encoding="utf-8") as f:
        query = json.load(f)
    with configs_path.open("r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    cleaning_report = None
    cleaning_path = Path(args.cleaning_report)
    if cleaning_path.exists():
        try:
            with cleaning_path.open("r", encoding="utf-8") as f:
                cleaning_report = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARN: could not parse {cleaning_path}: {exc}",
                  file=sys.stderr)

    user_notes = None
    notes_path = Path(args.notes)
    if notes_path.exists():
        try:
            user_notes = notes_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"WARN: could not read {notes_path}: {exc}", file=sys.stderr)

    text = build_readme(
        query, configs,
        cleaning_report=cleaning_report,
        user_notes=user_notes,
        pages_url=args.pages_url,
        releases_url=args.releases_url,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
