#!/usr/bin/env python3
"""
make_pseudo_data.py — generate pseudo versions of the proprietary raw inputs.

Writes, under data/raw_pseudo/, every file that data/DATA_CLASSIFICATION.md
lists as proprietary, with the same filenames, zip members, column names, and
column types as the real inputs, but with randomly generated observations.
The shapes and distributions come from data/pseudo_schema.json (produced by
scripts/extract_pseudo_schema.py); no real observation is reproduced.

The pseudo firms share one identifier universe (JKP id, gvkey, permno, ISIN,
Capital IQ id, Sustainalytics entity id, Morningstar secid, ...) so that the
pipeline's merges succeed. Firm identifiers are sampled from the *public*
inputs (ICC ids, option-data permnos, Hoberg-Phillips gvkeys) so that the real
public data merge onto the pseudo panel; the pairing between identifiers is
random and carries no information about real firms.

To run the pipeline on pseudo data, copy (or symlink) the generated folders
over data/raw/ next to the public inputs and run `bash run.sh`. The exhibits
will not match the paper.

Run with:  uv run python scripts/make_pseudo_data.py [--seed 1] [--n-us 500] [--n-global 500]
"""

import argparse
import io
import json
import string
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_RAW = REPO / "data" / "raw"
SCHEMA_PATH = REPO / "data" / "pseudo_schema.json"
OUT_DEFAULT = REPO / "data" / "raw_pseudo"

GLOBAL_COUNTRIES = ["GBR", "JPN", "DEU", "FRA", "CAN"]  # all present in CCPI_data.xlsx and country_abbr.xlsx
START, END = "2005-01-31", "2022-12-31"
# Valid GICS sub-industry codes (public classification), 4 per sector.
GICS_CODES = [10101010, 10101020, 10102010, 10102020, 15101010, 15101020, 15102010, 15104010,
              20101010, 20102010, 20103010, 20201010, 25101010, 25201010, 25301010, 25501010,
              30101010, 30201010, 30202010, 30301010, 35101010, 35102010, 35201010, 35202010,
              40101010, 40102010, 40201020, 40301010, 45101010, 45102010, 45103010, 45201020,
              50101010, 50102010, 50201010, 50202010, 55101010, 55102010, 55103010, 55105010,
              60101010, 60102010, 60102020, 60102030]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Generic value engine driven by the schema
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self, schema, rng):
        self.files = schema["files"]
        self.rng = rng

    def profile(self, key, col):
        f = self.files[key]
        cols = f.get("columns") or f.get("fixed_columns")
        return cols.get(col)

    def header(self, key):
        f = self.files[key]
        return f.get("all_columns") or list(f["columns"])

    def draw(self, p, n, missing=None):
        """Draw n values for a column profile p; None entries become missing."""
        rng = self.rng
        miss = p["missing_share"] if missing is None else missing
        dtype = p["dtype"]
        if dtype == "empty":
            return pd.Series([np.nan] * n, dtype=object)
        if "categories" in p:
            cats = list(p["categories"])
            probs = np.array(list(p["categories"].values()), dtype=float)
            probs = probs / probs.sum()
            vals = pd.Series(rng.choice(cats, size=n, p=probs))
            if dtype in ("int", "float"):
                vals = pd.to_numeric(vals)
        elif dtype in ("int", "float"):
            vals = pd.Series(self.numeric(p, n))
        elif dtype == "date":
            vals = pd.Series(self.dates(p, n))
        elif dtype == "bool":
            vals = pd.Series(rng.random(n) < p.get("share_true", 0.5))
        else:
            vals = pd.Series(self.strings(p, n))
        if miss > 0:
            vals = vals.mask(rng.random(n) < miss)
        return vals

    def numeric(self, p, n):
        q = p.get("quantiles")
        if not q:
            return np.full(n, p.get("mean", 0.0))
        xs = [p["min"]] + [q[k] for k in sorted(q, key=float)] + [p["max"]]
        ps = [0.0] + [float(k) for k in sorted(q, key=float)] + [1.0]
        vals = np.interp(self.rng.random(n), ps, xs)
        return np.round(vals) if p["dtype"] == "int" else vals

    def dates(self, p, n):
        lo = pd.Timestamp(p.get("min", "2005-01-01"))
        hi = pd.Timestamp(p.get("max", "2022-12-31"))
        days = self.rng.integers(0, max(1, (hi - lo).days + 1), size=n)
        ts = lo + pd.to_timedelta(days, unit="D")
        return format_dates(ts, p["format"], p.get("storage", "str"))

    def strings(self, p, n):
        masks = p.get("masks") or {"AAAA": 1.0}
        mk = list(masks)
        pr = np.array(list(masks.values()), dtype=float)
        pr = pr / pr.sum()
        chosen = self.rng.choice(mk, size=n, p=pr)
        return [self.from_mask(m) for m in chosen]

    def from_mask(self, mask):
        rng = self.rng
        out = []
        for ch in mask:
            if ch == "A":
                out.append(string.ascii_uppercase[rng.integers(26)])
            elif ch == "9":
                out.append(string.digits[rng.integers(10)])
            else:
                out.append(ch)
        return "".join(out)

    def frame(self, key, n, overrides, header=None, generic=True):
        """DataFrame with the file's full header; `overrides` fill structural
        columns, the remaining profiled columns are drawn generically, and
        unprofiled columns are left empty."""
        cols = header or self.header(key)
        data = {}
        for c in cols:
            if c in overrides:
                v = overrides[c]
                data[c] = v if not np.isscalar(v) else [v] * n
            elif generic and self.profile(key, c) is not None:
                data[c] = self.draw(self.profile(key, c), n)
            else:
                data[c] = [np.nan] * n
        return pd.DataFrame(data, columns=cols)


def format_dates(ts, fmt, storage="str"):
    ts = pd.DatetimeIndex(ts)
    if fmt in ("%Y%m%d", "%Y%m"):
        s = ts.strftime(fmt).astype(int)
        return np.asarray(s) if storage == "int" else np.asarray(s.astype(str))
    return np.asarray(ts.strftime(fmt))


def write_csv(path, df, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kw)
    log(f"  wrote {path.relative_to(path.parents[2])}  ({len(df):,} rows x {df.shape[1]} cols)")


def write_zip(path, members, **kw):
    """members: {member_name: DataFrame}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in members.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False, **kw)
            zf.writestr(name, buf.getvalue())
            log(f"  wrote {path.relative_to(path.parents[2])}:{name}  ({len(df):,} rows x {df.shape[1]} cols)")


def write_xlsx(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("Sheet1")
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in row])
    wb.save(path)
    log(f"  wrote {path.relative_to(path.parents[2])}  ({len(df):,} rows x {df.shape[1]} cols)")


# ----------------------------------------------------------------------------
# Firm universe
# ----------------------------------------------------------------------------
def ids_with_coverage(path, min_months, rng, n, start="2005-01-01", prefer=None):
    """Sample n ids from a public ICC file that have >= min_months of data after `start`.
    Ids in `prefer` (e.g. permnos with option data) are taken first."""
    df = pd.read_csv(path, usecols=["id", "datadate"])
    df = df[df.datadate >= start]
    cnt = df.groupby("id")["datadate"].nunique()
    pool = cnt[cnt >= min_months].index.to_numpy()
    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} ids with >= {min_months} months in {path.name}; lower --n-us/--n-global")
    if prefer is not None:
        first = np.intersect1d(pool, prefer)
        if len(first) >= n:
            return np.sort(rng.choice(first, size=n, replace=False))
        rest = rng.choice(np.setdiff1d(pool, first), size=n - len(first), replace=False)
        return np.sort(np.concatenate([first, rest]))
    return np.sort(rng.choice(pool, size=n, replace=False))


def build_universe(rng, n_us, n_global):
    log("building firm universe from public identifier pools")
    # prefer permnos that also appear in the (public) option data so those merges are non-empty
    with zipfile.ZipFile(DATA_RAW / "vilkov_options" / "MartinWagnerBounds.zip") as zf:
        opt = pd.read_csv(zf.open("MartinWagnerBounds.csv"), usecols=["id", "date"])
    opt_ids = opt[opt.date >= "2005-01-01"]["id"].dropna().astype(int).unique()
    us_ids = ids_with_coverage(DATA_RAW / "icc" / "icc_us.csv", 60, rng, n_us, prefer=opt_ids)
    log(f"  US ids: {len(us_ids)}, of which with option data: {np.isin(us_ids, opt_ids).sum()}")
    g_ids = ids_with_coverage(DATA_RAW / "icc" / "icc_comp.csv", 60, rng, n_global)
    g_gvkeys = g_ids // 1000  # JKP id convention for Compustat securities: gvkey*1000 + iid
    with zipfile.ZipFile(DATA_RAW / "hoberg_phillips" / "tnic3_data.zip") as zf:
        hp = pd.read_csv(zf.open("tnic3_data.txt"), sep="\t", usecols=["gvkey1"])["gvkey1"].unique()
    hp = np.setdiff1d(hp, g_gvkeys)
    us_gvkeys = np.sort(rng.choice(hp, size=n_us, replace=False))

    n = n_us + n_global
    firms = pd.DataFrame({
        "id": np.concatenate([us_ids, g_ids]).astype(int),
        "gvkey": np.concatenate([us_gvkeys, g_gvkeys]).astype(int),
        "permno": np.concatenate([us_ids, np.full(n_global, np.nan)]),
        "excntry": ["USA"] * n_us + [GLOBAL_COUNTRIES[i % len(GLOBAL_COUNTRIES)] for i in range(n_global)],
    })
    firms["iso2"] = firms.excntry.map({"USA": "US", "GBR": "GB", "JPN": "JP", "DEU": "DE", "FRA": "FR", "CAN": "CA"})
    firms["name"] = [f"PSEUDO FIRM {i:04d}" for i in range(n)]
    firms["ticker"] = [f"P{i:04d}" for i in range(n)]
    firms["cusip"] = [f"{rng.integers(10**7, 10**8):08d}{rng.integers(10)}" for _ in range(n)]
    firms["sedol"] = [f"B{rng.integers(10**5, 10**6):06d}" for _ in range(n)]
    firms["isin"] = firms.iso2 + firms.cusip + "0"
    firms["isin"] = firms["isin"].str[:12]
    firms["companyid"] = 100_000 + rng.choice(90_000_000, size=n, replace=False)  # Capital IQ
    firms["entityid"] = 1_008_000_000 + rng.choice(1_000_000, size=n, replace=False)  # Sustainalytics
    firms["companyid_old"] = 20_000 + rng.choice(9_000_000, size=n, replace=False)  # old Sustainalytics id
    firms["issuerid"] = [f"IID{rng.integers(10**9):09d}" for _ in range(n)]  # MSCI
    firms["institutionid"] = 4_000_000 + rng.choice(1_000_000, size=n, replace=False)  # Trucost
    firms["gics"] = rng.choice(GICS_CODES, size=n)
    firms["sic"] = rng.integers(1000, 9000, size=n)
    firms["ff49"] = rng.integers(1, 50, size=n)
    firms["log_me"] = rng.normal(7.0, 1.6, size=n)  # me in USD millions, lognormal firm level
    firms["has_msci"] = rng.random(n) < 0.85
    firms["has_sust"] = rng.random(n) < 0.85
    firms["has_trucost"] = rng.random(n) < 0.75
    firms["has_sers"] = (rng.random(n) < 0.7) & (firms.excntry == "USA")
    firms["has_bonds"] = (rng.random(n) < 0.4) & (firms.excntry == "USA")
    return firms


# ----------------------------------------------------------------------------
# File writers
# ----------------------------------------------------------------------------
def gen_jkp(eng, firms, months, out):
    key = "jkp/Characteristics.zip"
    log("JKP characteristics")
    rng = eng.rng
    members = {}
    for cntry, grp in firms.groupby("excntry"):
        nf, nm = len(grp), len(months)
        panel = pd.DataFrame({"id": np.repeat(grp.id.to_numpy(), nm), "date_": np.tile(months, nf)})
        n = len(panel)
        f = grp.set_index("id")
        ret = rng.normal(0.005, 0.10, size=n)
        panel["ret_exc"] = ret
        panel["ret_exc_lead1m"] = panel.groupby("id")["ret_exc"].shift(-1)
        # market equity: firm level times a random walk in the firm's returns
        cumret = panel.groupby("id")["ret_exc"].cumsum().to_numpy()
        me = np.exp(f.log_me.reindex(panel.id).to_numpy() + cumret * 0.5)
        panel["me"] = me
        panel["me_lag1"] = panel.groupby("id")["me"].shift(1)
        panel["me_company"] = me
        # accounting quantities, positive (the pipeline requires > 0)
        panel["book_equity"] = me * np.exp(rng.normal(-0.5, 0.6, size=n))
        panel["assets"] = me * np.exp(rng.normal(0.3, 0.7, size=n))
        panel["sales"] = me * np.exp(rng.normal(-0.2, 0.8, size=n))
        panel["be_me"] = panel.book_equity / me
        panel["bev_mev"] = np.clip(rng.normal(1.1, 0.4, size=n), 0.05, None)
        panel["ni_me"] = rng.normal(0.05, 0.08, size=n)
        panel["ebitda_mev"] = np.clip(rng.normal(0.10, 0.06, size=n), 0.001, None)
        panel["debt_at"] = np.clip(rng.normal(0.25, 0.15, size=n), 0, 0.95)
        panel["ebit_at"] = rng.normal(0.06, 0.08, size=n)
        panel["ni_be"] = rng.normal(0.08, 0.15, size=n)
        panel["ret_12_1"] = rng.normal(0.08, 0.35, size=n)
        panel["beta_252d"] = rng.normal(1.0, 0.4, size=n)
        panel["cash_at"] = np.clip(rng.normal(0.12, 0.10, size=n), 0, 0.9)
        rank = panel.groupby("date_")["me"].rank(pct=True)
        panel["size_grp"] = pd.cut(rank, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=["nano", "micro", "small", "large", "mega"]).astype(str)
        over = {
            "id": panel.id.to_numpy(),
            "date": pd.DatetimeIndex(panel.date_).strftime("%Y-%m-%d"),
            "eom": format_dates(panel.date_, "%Y%m%d", "int"),
            "gvkey": f.gvkey.reindex(panel.id).to_numpy(),
            "permno": f.permno.reindex(panel.id).to_numpy(),
            "excntry": cntry,
            "crsp_shrcd": rng.choice([10, 11], size=n) if cntry == "USA" else np.full(n, np.nan),
            "ff49": f.ff49.reindex(panel.id).to_numpy(),
            "sic": f.sic.reindex(panel.id).to_numpy(),
            "gics": f.gics.reindex(panel.id).to_numpy(),
        }
        for c in ["me", "me_lag1", "me_company", "book_equity", "sales", "assets", "bev_mev", "ni_me", "ebitda_mev",
                  "size_grp", "ret_exc", "ret_exc_lead1m", "be_me", "debt_at", "ebit_at", "ni_be", "ret_12_1",
                  "beta_252d", "cash_at"]:
            over[c] = panel[c].to_numpy()
        members[f"{cntry.lower()}.csv"] = eng.frame(key, n, over, generic=False)
    write_zip(out / key, members)

    # industry codes: same firm-month panel, same codes
    key2 = "other/industry_codes.csv"
    nf, nm = len(firms), len(months)
    ids = np.repeat(firms.id.to_numpy(), nm)
    f = firms.set_index("id")
    df = eng.frame(key2, len(ids), {
        "id": ids, "eom": np.tile(pd.DatetimeIndex(months).strftime("%Y-%m-%d"), nf),
        "sic": f.sic.reindex(ids).to_numpy(), "ff49": f.ff49.reindex(ids).to_numpy()})
    write_csv(out / key2, df)

    # monthly series: return cutoffs and risk-free rate
    key3 = "jkp/return_cutoffs.csv"
    all_months = pd.date_range("1926-01-31", END, freq="ME")
    df = eng.frame(key3, len(all_months), {"eom": format_dates(all_months, "%Y%m%d", "int")})
    for c in df.columns:  # keep lower cutoffs negative and upper positive
        if c.endswith("_0_1") or c.endswith("_1"):
            df[c] = -df[c].abs()
        elif c.endswith("_99") or c.endswith("_99_9"):
            df[c] = df[c].abs()
    write_csv(out / key3, df)

    key4 = "jkp/rf.csv"
    rf_months = pd.date_range("1963-07-31", "2023-11-30", freq="ME")
    df = eng.frame(key4, len(rf_months), {"date": format_dates(rf_months, "%Y%m", "int")})
    df["rf"] = df["rf"].abs()
    write_csv(out / key4, df)


def gen_crsp_dsenames(eng, firms, out):
    key = "other/crsp_dsenames.csv"
    log("CRSP names")
    us = firms[firms.excntry == "USA"]
    n = len(us)
    df = eng.frame(key, n, {
        "permno": us.permno.astype(int).to_numpy(), "namedt": "1990-01-01", "nameendt": "2023-12-29",
        "siccd": us.sic.to_numpy(), "hsiccd": us.sic.to_numpy(), "ticker": us.ticker.to_numpy(),
        "tsymbol": us.ticker.to_numpy(), "comnam": us.name.to_numpy(), "ncusip": us.cusip.str[:8].to_numpy(),
        "cusip": us.cusip.str[:8].to_numpy(), "shrcd": eng.rng.choice([10, 11], size=n),
        "permco": np.arange(1, n + 1)})
    write_csv(out / key, df)


def gen_linking(eng, firms, out):
    log("Capital IQ linking tables")
    n = len(firms)
    df = eng.frame("linking/WRDS_GVKEY.csv", n, {
        "companyid": firms.companyid.to_numpy(), "gvkey": firms.gvkey.to_numpy(),
        "startdate": "B", "enddate": "E", "companyname": firms.name.to_numpy()})
    write_csv(out / "linking/WRDS_GVKEY.csv", df, encoding="utf-8-sig")
    df = eng.frame("linking/WRDS_ISIN.csv", n, {
        "companyid": firms.companyid.to_numpy(), "isin": firms["isin"].to_numpy(),
        "startdate": "B", "enddate": "E", "companyname": firms.name.to_numpy()})
    write_csv(out / "linking/WRDS_ISIN.csv", df, encoding="utf-8-sig")


MSCI_FILES = [  # (filename, years, IVA_RATING_DATE format, is_xlsx)
    ("ESG Ratings Timeseries Expanded 2007 to 2012.xlsx", range(2007, 2013), "%Y%m%d", True),
    ("ESG Ratings Timeseries Expanded 2013.xlsx", [2013], "%Y%m%d", True),
    ("ESG Ratings Timeseries Expanded 2014.xlsx", [2014], "%m/%d/%y", True),
    ("ESG Ratings Timeseries Expanded 2015.xlsx", [2015], "%Y%m%d", True),
    ("ESG Ratings Timeseries Expanded 2016.xlsx", [2016], "%Y%m%d", True),
    ("ESG Ratings Timeseries Expanded 2017.xlsx", [2017], "%Y%m%d", True),
    ("ESG Ratings Timeseries Expanded 2018 - CSV.csv", [2018], "%Y%m%d", False),
    ("ESG Ratings Timeseries Expanded 2019 - CSV.csv", [2019], "%m/%d/%Y", False),
    ("ESG Ratings Timeseries Expanded 2020 - CSV.csv", [2020], "%m/%d/%Y", False),
    ("ESG Ratings Timeseries Expanded 2021 - CSV.csv", [2021], "%m/%d/%Y", False),
    ("ESG Ratings Timeseries Expanded 2022.csv", [2022], "%m/%d/%Y", False),
]


def gen_msci(eng, firms, out):
    log("MSCI ESG ratings")
    rng = eng.rng
    iss = firms[firms.has_msci].reset_index(drop=True)
    ni = len(iss)
    # persistent firm-level scores, small noise per observation
    base = {c: np.clip(rng.normal(5, 2, size=ni), 0, 10) for c in
            ["E", "S", "G", "CLIMATE", "NATRES", "WASTE", "ENVOPPS", "HUMAN", "IND"]}
    w = rng.dirichlet([3, 3, 3], size=ni) * 100  # pillar weights summing to 100
    ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    for fname, years, iva_fmt, is_xlsx in MSCI_FILES:
        key = f"msci/{fname}"
        header = eng.header(key)
        asof = pd.DatetimeIndex([pd.Timestamp(y, m, 1) for y in years for m in (1, 4, 7, 10)])  # quarterly
        nm = len(asof)
        n = ni * nm
        idx = np.tile(np.arange(ni), nm)
        dates = np.repeat(asof, ni)
        noise = lambda: rng.normal(0, 0.3, size=n)
        sc = {k: np.clip(base[k][idx] + noise(), 0, 10) for k in base}
        wt = w[idx]
        rating_dates = pd.DatetimeIndex(dates) - pd.to_timedelta(rng.integers(0, 300, size=n), unit="D")
        over = {
            "ISSUER_NAME": iss.name.to_numpy()[idx], "ISSUERID": iss.issuerid.to_numpy()[idx],
            "ISSUER_TICKER": iss.ticker.to_numpy()[idx], "ISSUER_CUSIP": iss.cusip.to_numpy()[idx],
            "ISSUER_SEDOL": iss.sedol.to_numpy()[idx], "ISSUER_ISIN": iss["isin"].to_numpy()[idx],
            "ISSUER_CNTRY_DOMICILE": iss.iso2.to_numpy()[idx],
            "AS_OF_DATE": format_dates(dates, "%Y%m%d", "int"),
            "IVA_RATING_DATE": format_dates(rating_dates, iva_fmt, "int" if iva_fmt == "%Y%m%d" else "str"),
            "IVA_COMPANY_RATING": rng.choice(ratings, size=n), "IVA_PREVIOUS_RATING": rng.choice(ratings, size=n),
            "INDUSTRY_ADJUSTED_SCORE": sc["IND"],
            "WEIGHTED_AVERAGE_SCORE": (sc["E"] * wt[:, 0] + sc["S"] * wt[:, 1] + sc["G"] * wt[:, 2]) / 100,
            "ENVIRONMENTAL_PILLAR_SCORE": sc["E"], "ENVIRONMENTAL_PILLAR_WEIGHT": wt[:, 0],
            "SOCIAL_PILLAR_SCORE": sc["S"], "SOCIAL_PILLAR_WEIGHT": wt[:, 1],
            "GOVERNANCE_PILLAR_SCORE": sc["G"], "GOVERNANCE_PILLAR_WEIGHT": wt[:, 2],
            "CLIMATE_CHANGE_THEME_SCORE": sc["CLIMATE"], "CLIMATE_CHANGE_THEME_WEIGHT": wt[:, 0] * 0.4,
            "NATURAL_RES_USE_THEME_SCORE": sc["NATRES"], "NATURAL_RES_USE_THEME_WEIGHT": wt[:, 0] * 0.2,
            "WASTE_MGMT_THEME_SCORE": sc["WASTE"], "WASTE_MGMT_THEME_WEIGHT": wt[:, 0] * 0.2,
            "ENVIRONMENTAL_OPPS_THEME_SCORE": sc["ENVOPPS"], "ENVIRONMENTAL_OPPS_THEME_WEIGHT": wt[:, 0] * 0.2,
            "HUMAN_CAPITAL_THEME_SCORE": sc["HUMAN"],
        }
        # the pipeline reads only the first 30 columns; draw those generically where not overridden,
        # leave the remaining (unused) columns empty to keep the files small
        df = eng.frame(key, n, over, header=header[:30])
        rest = pd.DataFrame({c: [np.nan] * n for c in header[30:]})
        df = pd.concat([df, rest], axis=1)
        if is_xlsx:
            write_xlsx(out / key, df)
        else:
            write_csv(out / key, df)


def gen_sustainalytics(eng, firms, out):
    log("Sustainalytics")
    rng = eng.rng
    ent = firms[firms.has_sust].reset_index(drop=True)
    ne = len(ent)
    ciq = "IQ" + ent.companyid.astype(str)

    # IDS_TABLE: the pipeline renames the four columns positionally to CIQ_ID, CompanyId_old, EntityId, CompanyName
    key = "sustainalytics/IDS_TABLE.csv"
    df = eng.frame(key, ne, {"CIQ_ID": ciq.to_numpy(), "CompanyId_old": ent.companyid_old.to_numpy(),
                             "CompanyId": ent.entityid.to_numpy(), "CompanyName": ent.name.to_numpy()})
    write_csv(out / key, df, encoding="utf-8-sig")

    # G_NAMES: header table, one row per firm, link via ISIN with [year1, year2]
    key = "sustainalytics/G_NAMES.csv"
    df = eng.frame(key, len(firms), {
        "gvkey": firms.gvkey.to_numpy(), "conm": firms.name.to_numpy(), "sedol": firms.sedol.to_numpy(),
        "isin": firms["isin"].to_numpy(), "fic": firms.excntry.to_numpy(), "sic": firms.sic.to_numpy(),
        "gsubind": firms.gics.to_numpy(), "gind": (firms.gics // 100).to_numpy(), "year1": 2000, "year2": 2022})
    write_csv(out / key, df, encoding="utf-8-sig")

    # indicator data (long format): annual file dates, a subset of the real field names
    key = "sustainalytics/sust_indicator_all.zip"
    cats = eng.profile(key, "FieldName")["categories"]
    weight_fields = [f for f in cats if f.endswith("-Weight-Overall-RR") and f[:2] in ("E.", "S.", "G.")
                     and f.replace("-Weight-Overall-RR", "-Weighted Score-Overall-RR") in cats]
    chosen = []
    for pillar in ("E.", "S.", "G."):
        chosen += [f for f in weight_fields if f.startswith(pillar)][:5]
    file_dates = pd.DatetimeIndex(["2019-12-02", "2020-12-01", "2021-12-01", "2022-12-01"])
    rows = []
    base_q = rng.uniform(20, 90, size=(ne, len(chosen)))  # firm-level quality per indicator (0-100)
    for d in file_dates:
        for j, wf in enumerate(chosen):
            sf = wf.replace("-Weight-Overall-RR", "-Weighted Score-Overall-RR")
            weight = rng.uniform(2, 12, size=ne)
            score = weight * np.clip(base_q[:, j] + rng.normal(0, 5, size=ne), 0, 100) / 100
            fd = (d - pd.to_timedelta(rng.integers(0, 120, size=ne), unit="D")).strftime("%Y-%m-%d")
            rows.append(pd.DataFrame({"EntityId": ent.entityid, "FieldName": wf, "FieldValue": weight,
                                      "FieldDate": fd, "File_date": d.strftime("%Y-%m-%d"),
                                      "FieldCluster": "Indicator Weight-Overall", "_j": 2 * j}))
            rows.append(pd.DataFrame({"EntityId": ent.entityid, "FieldName": sf, "FieldValue": score,
                                      "FieldDate": fd, "File_date": d.strftime("%Y-%m-%d"),
                                      "FieldCluster": "Indicator Weighted Score-Overall", "_j": 2 * j + 1}))
    long = pd.concat(rows, ignore_index=True)
    field_ids = 10**11 + rng.choice(10**9, size=2 * len(chosen), replace=False)
    over = {c: long[c].to_numpy() for c in ["EntityId", "FieldName", "FieldValue", "FieldDate", "File_date", "FieldCluster"]}
    over["FieldId"] = field_ids[long._j.to_numpy()]
    df = eng.frame(key, len(long), over)
    write_zip(out / key, {"ipcyxtqygf6aicwh.csv": df})

    # reference data: entity x extraction date
    key = "sustainalytics/sust_ref_data.zip"
    idx = np.tile(np.arange(ne), len(file_dates))
    df = eng.frame(key, len(idx), {
        "EntityId": ent.entityid.to_numpy()[idx], "EntityName": ent.name.to_numpy()[idx],
        "Company_Type": "Public", "ExtractionDate": np.repeat(file_dates.strftime("%Y-%m-%d"), ne),
        "CountryCode": ent.excntry.to_numpy()[idx], "Ticker": ent.ticker.to_numpy()[idx],
        "ISIN": ent["isin"].to_numpy()[idx], "SEDOL": ent.sedol.to_numpy()[idx], "CUSIP": ent.cusip.to_numpy()[idx]})
    write_zip(out / key, {"vqywhnnzjul6s0cp.csv": df})

    # legacy data: annual snapshots 2009-2019, full 77-column header (only the 8 used columns are filled)
    key = "sustainalytics/sust_legacy.zip"
    leg_dates = pd.DatetimeIndex([f"{y}-06-30" for y in range(2009, 2020)])
    idx = np.tile(np.arange(ne), len(leg_dates))
    n = len(idx)
    esg_base = rng.uniform(30, 85, size=ne)
    df = eng.frame(key, n, {
        "Date": np.repeat(leg_dates.strftime("%Y-%m-%d"), ne), "Company": ent.name.to_numpy()[idx],
        "ISIN": ent["isin"].to_numpy()[idx], "CapitalIQ_ID": ciq.to_numpy()[idx],
        # a handful of real rows are shifted and carry an "IQ..." id here; the pipeline drops them with a
        # .str accessor, which needs the column to load as text -- so reproduce a few such rows
        "CompanyId": np.where(rng.random(n) < 0.01, ciq.to_numpy()[idx], ent.companyid_old.astype(str).to_numpy()[idx]),
        "Country": ent.excntry.to_numpy()[idx],
        "total_esg_score": np.clip(esg_base[idx] + rng.normal(0, 4, size=n), 0, 100),
        "environment_score": np.clip(esg_base[idx] + rng.normal(0, 8, size=n), 0, 100)}, generic=False)
    write_zip(out / key, {"orvxvkrcejp13hs6.csv": df})


def gen_trucost(eng, firms, out):
    key = "trucost/trucost.zip"
    log("Trucost")
    rng = eng.rng
    tc = firms[firms.has_trucost].reset_index(drop=True)
    years = np.arange(2005, 2023)
    idx = np.tile(np.arange(len(tc)), len(years))
    yr = np.repeat(years, len(tc))
    n = len(idx)
    s1 = np.exp(rng.normal(11, 2, size=n))
    s2 = np.exp(rng.normal(10.5, 1.8, size=n))
    s3 = np.exp(rng.normal(12, 2, size=n))
    rev = np.exp(rng.normal(7, 1.5, size=n))  # USD millions
    eff = pd.DatetimeIndex([pd.Timestamp(y + 1, 1, 1) for y in yr]) + pd.to_timedelta(rng.integers(90, 270, size=n), unit="D")
    df = eng.frame(key, n, {
        "gvkey": tc.gvkey.to_numpy()[idx], "institutionid": tc.institutionid.to_numpy()[idx],
        "fiscalyear": yr, "periodenddate": [f"{y}-12-31" for y in yr],
        "di_323615_text": [d.strftime("%b %d %Y 12:00AM") for d in eff],
        "di_319413": s1, "di_319414": s2, "di_319415": s3,
        "di_319407": s1 / rev, "di_319408": s2 / rev, "di_319409": s3 / rev})
    write_zip(out / key, {"nfyldit54txkyele.csv": df})


def gen_morningstar(eng, firms, out):
    log("Morningstar sustainable funds and holdings")
    rng = eng.rng
    n_funds = 200
    secids = [f"F{rng.integers(10**8):08d}X" for _ in range(n_funds)]
    fundids = [f"FS{rng.integers(10**8):08d}" for _ in range(n_funds)]
    yms = pd.date_range("2018-10-31", END, freq="ME")
    starts = pd.Timestamp("2018-10-31") + pd.to_timedelta(rng.integers(0, 400, size=n_funds), unit="D")
    starts = pd.DatetimeIndex(np.where(rng.random(n_funds) > 0.3, starts, pd.Timestamp("2018-10-31")))  # many start at the series start
    sizes = np.exp(rng.normal(19, 1.5, size=n_funds))

    key = "morningstar/Sustainable_Funds_Processed.csv"
    idx = np.tile(np.arange(n_funds), len(yms))
    n = len(idx)
    fs = sizes[idx] * np.exp(rng.normal(0, 0.1, size=n))
    fs_str = [f"{v:,.2f}" for v in fs]
    df = eng.frame(key, n, {
        "fundid": np.array(fundids)[idx], "secid": np.array(secids)[idx],
        "name": np.array([f"Pseudo Sustainable Fund {i:03d}" for i in range(n_funds)])[idx],
        "isin": np.array([f"LU{rng.integers(10**9):09d}0" for _ in range(n_funds)])[idx],
        "global_broad_category_group": "Equity", "sustainable_investment_overall": "Yes",
        "sustainable_investment_overall_start_date": starts.strftime("%Y-%m-%d").to_numpy()[idx],
        "ym": np.repeat(yms.strftime("%Y-%m"), n_funds), "fund_size": fs_str, "fund_size_orig": fs_str,
        "inception_date": "2010-01-15"})
    df["obsolete_date"] = np.nan
    write_csv(out / key, df)

    # wide holdings: one row per (fund, holding ISIN), one column per month-end with the weight in percent
    files = ["Morningstar_holdings_consolidated.csv", "Morningstar_holdings_consolidated_notArticle9.csv",
             "Morningstar_holdings_consolidated_notArticle9_US.csv"]
    assign = rng.integers(0, 3, size=n_funds)
    date_cols = [d.strftime("%Y-%m-%d 00:00:00") for d in yms]
    for k, fname in enumerate(files):
        key = f"morningstar/{fname}"
        rows = []
        for fi in np.flatnonzero(assign == k):
            nh = rng.integers(20, 60)
            pool = firms if k != 2 else firms[firms.excntry == "USA"]
            held = pool.sample(nh, random_state=int(rng.integers(2**31)))
            w = rng.dirichlet(np.ones(nh)) * 100
            block = pd.DataFrame({"ISIN": held["isin"].to_numpy(), "Name": held.name.to_numpy(),
                                  "Detail Holding Type": "EQUITY", "Country": held.excntry.to_numpy(),
                                  "Coupon %": np.nan, "Secondary Sector": np.nan,
                                  "Currency": "USD", "Maturity Date": np.nan})
            for j, dc in enumerate(date_cols):
                block[dc] = w * np.exp(rng.normal(0, 0.15, size=nh))
            block["SecId"] = secids[fi]
            rows.append(block)
        df = pd.concat(rows, ignore_index=True)
        # drop some fund-months entirely (missing reports) to mimic the real sparsity
        mask = rng.random((len(df), len(date_cols))) < 0.3
        df[date_cols] = df[date_cols].mask(mask)
        write_csv(out / key, df)


def gen_sers(eng, firms, out):
    log("subjective expected returns")
    rng = eng.rng
    us = firms[firms.has_sers]
    months = pd.date_range(START, "2021-12-31", freq="ME")
    permnos = np.repeat(us.permno.astype(int).to_numpy(), len(months))
    eom = np.tile(months.strftime("%Y-%m-%d"), len(us))
    n = len(permnos)
    for key in ["sers/subjective_er.csv", "sers/value_line.csv", "sers/morningstar.csv"]:
        df = eng.frame(key, n, {"permno": permnos, "eom": eom})
        write_csv(out / key, df)


def gen_bonds(eng, firms, out):
    key = "bonds/yields_traded_firm.csv"
    log("bond yields")
    us = firms[firms.has_bonds]
    months = pd.date_range(START, "2021-11-30", freq="ME")
    gvkeys = np.repeat(us.gvkey.to_numpy(), len(months))
    eom = np.tile(months.strftime("%Y-%m-%d"), len(us))
    n = len(gvkeys)
    df = eng.frame(key, n, {"gvkey": gvkeys, "eom": eom})
    for c in ["debt_mv", "debt_fv", "tmt", "rating_num", "yield", "duration"]:  # the pipeline requires these
        df[c] = df[c].fillna(df[c].median())
    write_csv(out / key, df)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-us", type=int, default=500)
    ap.add_argument("--n-global", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    schema = json.loads(SCHEMA_PATH.read_text())
    eng = Engine(schema, rng)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    firms = build_universe(rng, args.n_us, args.n_global)
    months = pd.date_range(START, END, freq="ME")

    gen_jkp(eng, firms, months, out)
    gen_crsp_dsenames(eng, firms, out)
    gen_linking(eng, firms, out)
    gen_msci(eng, firms, out)
    gen_sustainalytics(eng, firms, out)
    gen_trucost(eng, firms, out)
    gen_morningstar(eng, firms, out)
    gen_sers(eng, firms, out)
    gen_bonds(eng, firms, out)

    (out / "PSEUDO_DATA_README.txt").write_text(
        "Pseudo data generated by scripts/make_pseudo_data.py "
        f"(seed={args.seed}, n_us={args.n_us}, n_global={args.n_global}, {datetime.now():%Y-%m-%d}).\n"
        "Every file mirrors a proprietary raw input: same filename, zip member, columns and types, "
        "random observations. Copy these folders over data/raw/ (keeping the public inputs) and run "
        "`bash run.sh`. The resulting exhibits will not match the paper.\n")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
