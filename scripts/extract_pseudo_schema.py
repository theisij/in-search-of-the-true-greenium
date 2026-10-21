#!/usr/bin/env python3
"""
extract_pseudo_schema.py — record the *shape* of the proprietary raw inputs.

Reads every raw file that cannot be redistributed (see
data/DATA_CLASSIFICATION.md), using the same reader settings as the pipeline,
and writes data/pseudo_schema.json. The JSON holds, per file and per column:
name, dtype, missing share, date format, numeric quantiles, string masks, and
the distinct values of low-cardinality columns. It never stores identifiers,
names, or any row of the real data. scripts/make_pseudo_data.py consumes it.

This script is a one-off tool for the authors; replicators never need to run
it. It reads the real data from data/raw/ and nothing in the pipeline depends
on it.

Run with:  uv run python scripts/extract_pseudo_schema.py [--sample-rows N]
"""

import argparse
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "pseudo_schema.json"

# ----------------------------------------------------------------------------
# Which files to profile, and how the pipeline reads them.
# `reader` holds the pandas.read_csv kwargs the pipeline uses (minus usecols/
# nrows), so the profile reflects what the code will actually see.
# ----------------------------------------------------------------------------
FILES = {
    # --- JKP (CRSP/Compustat-derived) ---------------------------------------
    "jkp/Characteristics.zip": {
        "type": "zip_multi", "reader": {}, "force_categories": ["excntry", "size_grp"],
        "used_by": "greenium/expected_returns.py:market_data",
        "usecols": ['id', 'eom', 'gvkey', 'permno', 'excntry', 'crsp_shrcd',
                    'me', 'me_lag1', 'me_company', 'book_equity', 'sales', 'assets',
                    'bev_mev', 'ni_me', 'ebitda_mev', 'size_grp', 'ff49', 'sic', 'gics',
                    'ret_exc', 'ret_exc_lead1m', 'be_me', 'debt_at', 'ebit_at', 'ni_be',
                    'ret_12_1', 'beta_252d', 'cash_at'],
    },
    "jkp/return_cutoffs.csv": {"type": "csv", "reader": {},
                               "used_by": "greenium/expected_returns.py:market_data"},
    "jkp/rf.csv": {"type": "csv", "reader": {},
                   "used_by": "greenium/expected_returns.py:expected_returns"},
    # --- MSCI ----------------------------------------------------------------
    "msci/ESG Ratings Timeseries Expanded 2007 to 2012.xlsx": {"type": "xlsx", "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2013.xlsx": {"type": "xlsx", "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2014.xlsx": {"type": "xlsx", "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2015.xlsx": {"type": "xlsx", "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2016.xlsx": {"type": "xlsx", "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2017.xlsx": {"type": "xlsx", "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2018 - CSV.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2019 - CSV.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2020 - CSV.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2021 - CSV.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:msci"},
    "msci/ESG Ratings Timeseries Expanded 2022.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:msci"},
    # --- Sustainalytics ------------------------------------------------------
    "sustainalytics/sust_indicator_all.zip": {
        "type": "zip", "member": "ipcyxtqygf6aicwh.csv", "reader": {"encoding": "unicode_escape"},
        "used_by": "greenium/green_measures.py:sustainalytics",
        "usecols": ['EntityId', 'FieldName', 'FieldValue', 'File_date', 'FieldDate'],
        "force_categories": ["FieldName"],  # pipeline pivots on these names and pattern-matches E./S./G. prefixes
    },
    "sustainalytics/sust_ref_data.zip": {
        "type": "zip", "member": "vqywhnnzjul6s0cp.csv", "reader": {"encoding": "unicode_escape"},
        "used_by": "greenium/green_measures.py:sustainalytics",
    },
    "sustainalytics/sust_legacy.zip": {
        "type": "zip", "member": "orvxvkrcejp13hs6.csv", "reader": {"encoding": "unicode_escape"},
        "used_by": "greenium/green_measures.py:sustainalytics",
        "usecols": ['Date', 'Company', 'ISIN', 'CapitalIQ_ID', 'CompanyId', 'Country',
                    'total_esg_score', 'environment_score'],
        "roles": {"CompanyId": "sust_companyid_old"},
    },
    "sustainalytics/IDS_TABLE.csv": {"type": "csv", "reader": {"encoding": "latin1"},
                                     "used_by": "greenium/green_measures.py:sustainalytics",
                                     "roles": {"CompanyId": "sust_entityid"}},  # pipeline renames col 3 -> EntityId
    "sustainalytics/G_NAMES.csv": {"type": "csv", "reader": {},
                                   "used_by": "greenium/green_measures.py:sustainalytics"},
    # --- Trucost -------------------------------------------------------------
    "trucost/trucost.zip": {
        "type": "zip", "member": "nfyldit54txkyele.csv", "reader": {},
        "used_by": "greenium/green_measures.py:trucost",
        "usecols": ['gvkey', 'periodenddate', 'di_323615_text', 'di_319407', 'di_319413',
                    'di_319408', 'di_319414', 'di_319409', 'di_319415', 'di_319403_text',
                    'di_329689_text', 'di_368758_text'],
    },
    # --- Morningstar ---------------------------------------------------------
    "morningstar/Sustainable_Funds_Processed.csv": {"type": "csv", "reader": {},
                                                    "used_by": "greenium/aggregate_green_score.py:load_fund_info"},
    "morningstar/Morningstar_holdings_consolidated.csv": {"type": "wide_holdings", "force_categories": ["Detail Holding Type", "Country", "Currency"],
                                                          "used_by": "greenium/aggregate_green_score.py:load_holdings_raw"},
    "morningstar/Morningstar_holdings_consolidated_notArticle9.csv": {"type": "wide_holdings", "force_categories": ["Detail Holding Type", "Country", "Currency"],
                                                                      "used_by": "greenium/aggregate_green_score.py:load_holdings_raw"},
    "morningstar/Morningstar_holdings_consolidated_notArticle9_US.csv": {"type": "wide_holdings", "force_categories": ["Detail Holding Type", "Country", "Currency"],
                                                                         "used_by": "greenium/aggregate_green_score.py:load_holdings_raw"},
    # --- Linking (Capital IQ via WRDS) ---------------------------------------
    "linking/WRDS_GVKEY.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:isin_gvkey_load"},
    "linking/WRDS_ISIN.csv": {"type": "csv", "reader": {}, "used_by": "greenium/green_measures.py:isin_gvkey_load"},
    # --- Subjective expected returns -----------------------------------------
    "sers/subjective_er.csv": {"type": "csv", "reader": {}, "used_by": "greenium/expected_returns.py:expected_returns"},
    "sers/value_line.csv": {"type": "csv", "reader": {}, "used_by": "greenium/expected_returns.py:expected_returns",
                            "usecols": ['eom', 'permno', 'vl_req_ret']},
    "sers/morningstar.csv": {"type": "csv", "reader": {}, "used_by": "greenium/expected_returns.py:expected_returns",
                             "usecols": ['permno', 'eom', 'ms_coe_exc']},
    # --- Bonds ---------------------------------------------------------------
    "bonds/yields_traded_firm.csv": {"type": "csv", "reader": {}, "used_by": "greenium/expected_returns.py:bond_data"},
    # --- Other ---------------------------------------------------------------
    "other/industry_codes.csv": {"type": "csv", "reader": {}, "used_by": "greenium/expected_returns.py:market_data"},
    "other/crsp_dsenames.csv": {"type": "csv", "reader": {}, "used_by": "R/green_tilts.R"},
}

# Identifier roles, keyed by lower-cased column name. The generator draws every
# column with the same role from one shared universe so that merges succeed.
ROLES = {
    "id": "jkp_id", "gvkey": "gvkey", "permno": "permno", "permco": "permco",
    "isin": "isin", "issuer_isin": "isin",
    "companyid": "ciq_companyid", "companyid_old": "ciq_companyid",
    "ciq_id": "ciq_id", "capitaliq_id": "ciq_id",
    "entityid": "sust_entityid", "issuerid": "msci_issuerid",
    "institutionid": "trucost_institutionid",
    "secid": "ms_secid", "fundid": "ms_fundid",
    "cusip": "cusip", "ncusip": "cusip", "issuer_cusip": "cusip",
    "sedol": "sedol", "issuer_sedol": "sedol",
    "ticker": "ticker", "tsymbol": "ticker", "issuer_ticker": "ticker",
    "unique_id": "sust_unique_id", "unique_id_new": "sust_unique_id",
    "excntry": "country", "fic": "country", "countrycode": "country",
    "issuer_cntry_domicile": "country",
    "comnam": "firm_name", "conm": "firm_name", "companyname": "firm_name",
    "company": "firm_name", "entityname": "firm_name", "issuer_name": "firm_name",
    "name": "name",
}

# Quirks the generator must reproduce; documented here so the JSON is
# self-contained.
HINTS = {
    "jkp/Characteristics.zip": "One CSV per country, named <iso3>.csv; all share the same header. "
                               "Pipeline keeps eom >= 19910000 and requires gvkey.",
    "msci/*": "Pipeline reads the first 30 columns (usecols=range(30)). IVA_RATING_DATE format differs by "
              "file: yyyymmdd int (2007-2013, 2015-2018), m/d/yy (2014), m/d/YYYY (2019-2022). "
              "AS_OF_DATE is yyyymmdd int in all files. ISSUER_ISIN must be non-missing to be used.",
    "sustainalytics/sust_indicator_all.zip": "Long format: one row per (EntityId, FieldName, FieldDate). "
                                             "Pipeline pivots on FieldName; the FieldName value set matters.",
    "linking/*": "startdate/enddate are yyyymmdd strings, with literal 'B' (beginning) and 'E' (end) sentinels. "
                 "companyid is the numeric Capital IQ company id; the pipeline builds CIQ_ID = 'IQ' + str(companyid), "
                 "so role ciq_id (IQ-prefixed string) and role ciq_companyid (int) are the same universe.",
    "morningstar/Morningstar_holdings_*.csv": "Wide format: 8 fixed columns then one column per month-end "
                                             "('YYYY-MM-DD 00:00:00') holding the portfolio weight in percent. "
                                             "Rows are (fund secid, holding ISIN). Pipeline keeps "
                                             "Detail Holding Type == 'EQUITY'. NOTE: the real header has no secid "
                                             "column in the fixed 8; the pipeline lower-cases and reads 'secid' — "
                                             "check the profile for the exact fixed-column list.",
    "morningstar/Sustainable_Funds_Processed.csv": "fund_size is a quoted string with thousands separators "
                                                   "('22,705,723.00'). ym is 'YYYY-MM'.",
    "other/crsp_dsenames.csv": "Used in R to attach ticker/comnam by permno over [namedt, nameendt].",
    "trucost/trucost.zip": "di_323615_text (effective date) is a text timestamp like 'Jan 15 2020 12:00AM'; the "
                           "pipeline keeps the first 11 characters and parses them with pd.to_datetime. "
                           "gvkey is missing for most rows in the real file; only rows with gvkey are usable.",
    "dates": "Date profiles carry 'sentinels' when a few values do not match the format (e.g. 'B'/'E' in the "
             "linking tables); the generator should reproduce those at the stated share.",
}

DATE_PATTERNS = [
    ("%Y-%m-%d", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("%Y-%m-%d %H:%M:%S", re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")),
    ("%m/%d/%Y", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
    ("%m/%d/%y", re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")),
    ("%Y-%m", re.compile(r"^\d{4}-\d{2}$")),
    ("%Y%m%d", re.compile(r"^(19|20)\d{6}$")),
    ("%Y%m", re.compile(r"^(19|20)\d{4}$")),
]
DATE_NAME_RE = re.compile(r"date|eom|^ym$|dt$|^year", re.I)
MAX_CATEGORIES = 60


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def count_lines(path, member=None):
    """Row count (excluding header) via wc -l, streaming zip members through unzip -p."""
    if member is None:
        out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, check=True).stdout
    else:
        out = subprocess.run(f"unzip -p '{path}' '{member}' | wc -l", shell=True,
                             capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) - 1


def mask_of(s):
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", s))


def detect_date_format(series, name):
    """Return the date format if (nearly) all non-missing values match one pattern."""
    vals = series.dropna()
    if vals.empty:
        return None
    sample = vals.sample(min(len(vals), 2000), random_state=0)
    as_str = sample.astype(str)
    if pd.api.types.is_float_dtype(sample) and (sample == np.floor(sample)).all():
        as_str = sample.astype("int64").astype(str)
    for fmt, rx in DATE_PATTERNS:
        if fmt in ("%Y%m%d", "%Y%m") and not DATE_NAME_RE.search(name):
            continue  # an 8-digit integer is only a date if the column name says so
        if as_str.map(lambda v: bool(rx.match(v))).mean() > 0.98:
            return fmt
    return None


def profile_column(series, name, force_categories=False):
    n = len(series)
    missing = float(series.isna().mean()) if n else 1.0
    col = {"missing_share": round(missing, 4)}
    role = ROLES.get(name.lower())
    if role:
        col["role"] = role
    vals = series.dropna()
    if vals.empty:
        col["dtype"] = "empty"
        return col

    fmt = detect_date_format(vals, name)
    if fmt:
        col["dtype"] = "date"
        col["format"] = fmt
        col["storage"] = "int" if pd.api.types.is_numeric_dtype(vals) else "str"
        as_str = vals.astype("int64").astype(str) if col["storage"] == "int" else vals.astype(str)
        parsed = pd.to_datetime(as_str, format=fmt, errors="coerce")
        bad = as_str[parsed.isna()]
        if len(bad):
            col["sentinels"] = {"share": round(len(bad) / len(as_str), 5)}
            if bad.nunique() <= 10:
                col["sentinels"]["values"] = {str(k): round(float(v), 5)
                                              for k, v in bad.value_counts(normalize=True).items()}
        parsed = parsed.dropna()
        if not parsed.empty:
            col["min"] = str(parsed.min().date())
            col["max"] = str(parsed.max().date())
        return col

    if pd.api.types.is_bool_dtype(vals):
        col["dtype"] = "bool"
        col["share_true"] = round(float(vals.mean()), 4)
        return col

    if pd.api.types.is_numeric_dtype(vals):
        v = vals.astype(float)
        v = v[np.isfinite(v)]
        is_int = bool((v == np.floor(v)).all()) if len(v) else False
        col["dtype"] = "int" if is_int else "float"
        col["n_unique_sample"] = int(vals.nunique())
        qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        if len(v):
            col.update({
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std()) if len(v) > 1 else 0.0,
                "quantiles": dict(zip([str(q) for q in qs], [float(x) for x in v.quantile(qs)])),
            })
            if (col["n_unique_sample"] <= MAX_CATEGORIES and not role) or force_categories:
                counts = vals.value_counts(normalize=True)
                col["categories"] = {str(int(k)) if is_int else str(k): round(float(p), 5)
                                     for k, p in counts.items()}
        return col

    # strings
    s = vals.astype(str)
    col["dtype"] = "str"
    col["n_unique_sample"] = int(s.nunique())
    lens = s.str.len()
    col["len_min"], col["len_max"] = int(lens.min()), int(lens.max())
    masks = Counter(s.sample(min(len(s), 5000), random_state=0).map(mask_of))
    col["masks"] = {m: round(c / sum(masks.values()), 4) for m, c in masks.most_common(5)}
    identifying = role in ("firm_name", "name", "isin", "cusip", "sedol", "ticker", "gvkey", "permno",
                           "jkp_id", "ciq_companyid", "ciq_id", "sust_entityid", "msci_issuerid",
                           "trucost_institutionid", "ms_secid", "ms_fundid", "sust_unique_id", "permco")
    if (col["n_unique_sample"] <= MAX_CATEGORIES and not identifying) or force_categories:
        counts = s.value_counts(normalize=True)
        col["categories"] = {k: round(float(p), 5) for k, p in counts.items()}
    return col


BOM_MARKS = ("\ufeff", "\u00ef\u00bb\u00bf")  # utf-8 BOM, and the same bytes decoded as latin1


def strip_bom(df):
    """Drop a byte-order mark from the first column name; report whether one was present."""
    first = str(df.columns[0])
    for mark in BOM_MARKS:
        if first.startswith(mark):
            df = df.rename(columns={df.columns[0]: first[len(mark):]})
            return df, True
    return df, False


def profile_frame(df, roles=None, force_categories=()):
    df, bom = strip_bom(df)
    out = {c: profile_column(df[c], c, c in force_categories) for c in df.columns}
    for c, r in (roles or {}).items():
        if c in out:
            out[c]["role"] = r
    return out, bom


def sample_csv(src, n_rows, sample_rows, reader, usecols=None):
    """Uniform random sample of ~sample_rows rows (reads the whole file once).

    When the pipeline reads the file with usecols, profile only those columns:
    that is what the code sees, and pandas then tolerates the few rows with a
    stray extra field (which otherwise raise a tokenizing error)."""
    kw = dict(low_memory=False, **reader)
    if usecols:
        kw["usecols"] = usecols
    if n_rows <= sample_rows:
        return pd.read_csv(src, **kw)
    rng = np.random.default_rng(0)
    frac = sample_rows / n_rows
    keep = rng.random(n_rows + 1) < frac
    keep[0] = False  # never skip the header
    return pd.read_csv(src, skiprows=lambda i: i > 0 and not keep[i], **kw)


def header_of(src, reader):
    df = pd.read_csv(src, nrows=0, **reader)
    df, _ = strip_bom(df)
    return list(df.columns)


def profile_csv(path, spec, sample_rows):
    n_rows = count_lines(path)
    all_cols = header_of(path, spec.get("reader", {}))
    df = sample_csv(path, n_rows, sample_rows, spec.get("reader", {}), spec.get("usecols"))
    cols, bom = profile_frame(df, spec.get("roles"), spec.get("force_categories", ()))
    return {"n_rows": n_rows, "n_cols": len(all_cols), "all_columns": all_cols, "sample_rows": len(df),
            "bom": bom, "columns": cols}


def profile_zip_single(path, spec, sample_rows):
    member = spec["member"]
    n_rows = count_lines(path, member)
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as fh:
            all_cols = header_of(fh, spec.get("reader", {}))
        with zf.open(member) as fh:
            df = sample_csv(fh, n_rows, sample_rows, spec.get("reader", {}), spec.get("usecols"))
    cols, bom = profile_frame(df, spec.get("roles"), spec.get("force_categories", ()))
    return {"members": [member], "n_rows": n_rows, "n_cols": len(all_cols), "all_columns": all_cols,
            "sample_rows": len(df), "bom": bom, "columns": cols}


def profile_zip_multi(path, spec, sample_rows):
    """JKP characteristics: profile the needed columns, pooled over all country files."""
    members, frames = [], []
    with zipfile.ZipFile(path) as zf:
        infos = [i for i in zf.infolist() if i.filename.endswith(".csv")]
        total_bytes = sum(i.file_size for i in infos)
        for info in infos:
            # rows per country proportional to its share of the data, so that
            # the pooled profile is not dominated by the many small countries
            per_member = max(500, int(sample_rows * info.file_size / total_bytes))
            with zf.open(info) as fh:
                head = pd.read_csv(fh, nrows=per_member, low_memory=False)
            # rough row count from the uncompressed size and the sampled bytes/row
            buf = io.StringIO()
            head.head(200).to_csv(buf, index=False, header=False)
            bytes_per_row = max(1, len(buf.getvalue().encode()) / max(1, min(200, len(head))))
            members.append({"name": info.filename, "uncompressed_bytes": info.file_size,
                            "n_rows_est": int(info.file_size / bytes_per_row), "sampled_rows": len(head),
                            "country": head["excntry"].mode().iat[0] if "excntry" in head and len(head) else None})
            frames.append(head[spec["usecols"]])
    df = pd.concat(frames, ignore_index=True)
    with zipfile.ZipFile(path) as zf:
        with zf.open(infos[0]) as fh:
            all_cols = list(pd.read_csv(fh, nrows=0).columns)
    cols, _ = profile_frame(df, spec.get("roles"), spec.get("force_categories", ()))
    return {"members": members, "n_members": len(members), "all_columns": all_cols, "n_cols": len(all_cols),
            "sample_rows": len(df), "columns": cols,
            "note": "columns profiled = the usecols the pipeline reads; all_columns lists the full header"}


def profile_xlsx(path, spec, sample_rows):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows, n_rows, header = [], 0, None
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            header = [str(c) if c is not None else f"Unnamed: {j}" for j, c in enumerate(row)]
            continue
        n_rows += 1
        if len(rows) < sample_rows:
            rows.append(row)
    wb.close()
    df = pd.DataFrame(rows, columns=header).infer_objects()
    # pandas.read_excel pushes cell values through its CSV type inference, so
    # numeric-looking text (and 'NA'-style strings) come out as floats/NaN.
    # Reproduce that so the profile matches what the pipeline sees.
    na_strings = pd.io.parsers.readers.STR_NA_VALUES
    for c in df.columns:
        if df[c].dtype == object:
            col = df[c].mask(df[c].astype(str).str.strip().isin(na_strings))
            num = pd.to_numeric(col, errors="coerce")
            if col.notna().sum() > 0 and num.notna().sum() == col.notna().sum():
                df[c] = num
            else:
                df[c] = col
    cols, _ = profile_frame(df, spec.get("roles"), spec.get("force_categories", ()))
    return {"sheets": wb.sheetnames, "n_rows": n_rows, "n_cols": len(header), "sample_rows": len(df),
            "columns": cols}


def profile_wide_holdings(path, spec, sample_rows):
    header = list(pd.read_csv(path, nrows=0).columns)
    fixed = [c for c in header if not re.match(r"^\d{4}-\d{2}-\d{2}", c)]
    date_cols = [c for c in header if c not in fixed]
    n_rows = count_lines(path)
    df = sample_csv(path, n_rows, sample_rows, {})
    fixed_profile, _ = profile_frame(df[fixed], spec.get("roles"), spec.get("force_categories", ()))
    w = df[date_cols].apply(pd.to_numeric, errors="coerce")
    stacked = w.stack(future_stack=True).dropna()
    nonmissing_per_row = w.notna().sum(axis=1)
    return {"layout": "wide_holdings", "n_rows": n_rows, "n_cols": len(header), "sample_rows": len(df),
            "header": header, "fixed_columns": fixed_profile,
            "date_columns": {"n": len(date_cols), "first": date_cols[0], "last": date_cols[-1],
                             "values": date_cols},
            "weight": profile_column(stacked.reset_index(drop=True), "weight"),
            "nonmissing_dates_per_row": profile_column(nonmissing_per_row, "n_dates")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=300_000)
    ap.add_argument("--only", nargs="*", help="profile only these keys (substring match)")
    args = ap.parse_args()

    schema = {"generated": datetime.now().isoformat(timespec="seconds"),
              "sample_rows_target": args.sample_rows,
              "roles": sorted(set(ROLES.values())), "hints": HINTS, "files": {}}
    if OUT.exists():  # incremental: keep earlier profiles when using --only
        schema["files"] = json.loads(OUT.read_text()).get("files", {})

    for key, spec in FILES.items():
        if args.only and not any(o in key for o in args.only):
            continue
        path = DATA_RAW / key
        if not path.exists():
            log(f"MISSING {key}")
            schema["files"][key] = {"error": "file not found"}
            continue
        t0 = time.time()
        log(f"profiling {key}")
        fn = {"csv": profile_csv, "zip": profile_zip_single, "zip_multi": profile_zip_multi,
              "xlsx": profile_xlsx, "wide_holdings": profile_wide_holdings}[spec["type"]]
        prof = fn(path, spec, args.sample_rows)
        prof.update({"type": spec["type"], "used_by": spec["used_by"], "bytes": path.stat().st_size,
                     "reader": spec.get("reader", {}), "usecols": spec.get("usecols")})
        schema["files"][key] = prof
        log(f"  done in {time.time() - t0:.0f}s")
        OUT.write_text(json.dumps(schema, indent=1, default=str))  # checkpoint after each file

    log(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
