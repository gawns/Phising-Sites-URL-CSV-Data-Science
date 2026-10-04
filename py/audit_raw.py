"""Audit data mentah phishing_site_urls.csv -> audit_raw.json

  python audit_raw.py [csv]

Memakai aturan kanonik yang sama persis dengan secure_pipeline.load_clean_data
agar angka tidak berbeda antar laporan.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # root project (folder di atas py/)
LAPORAN_DIR = BASE_DIR / "laporan"
LAPORAN_DIR.mkdir(exist_ok=True)
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "phishing_site_urls.csv"
OUT = LAPORAN_DIR / "audit_raw.json"
LABELS = ["good", "bad"]

# Pola "URL tak punya host yang bisa dipakai": tanpa titik, tanpa host setelah
# //, hanya skema, atau karakter kontrol/spasi.
CONTROL = re.compile(r"[\s\x00-\x1f\x7f]")
SCHEME = re.compile(r"^([a-z][a-z0-9+.\-]*)://", re.I)


def normalize_url(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("#", 1)[0]
    return value.rstrip("/")


def host_of(url: str) -> str:
    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def main():
    raw_bytes = CSV.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8", errors="replace")
    header = text.split("\n", 1)[0].strip()
    delimiter = ";" if header.count(";") > header.count(",") else ","

    df = pd.read_csv(CSV, sep=delimiter, dtype="string", keep_default_na=True,
                     na_values=["", "NA", "N/A", "null", "NULL", "NaN", "nan"])
    out = {
        "csv": str(CSV),
        "sha256": sha,
        "delimiter": delimiter,
        "header": header,
        "columns": list(df.columns),
        "rows_raw": int(len(df)),
        "rows_per_column_non_null": {c: int(df[c].notna().sum()) for c in df.columns},
    }

    url_raw = df["URL"]
    label_raw = df["Label"]
    out["missing"] = {
        "URL_null": int(url_raw.isna().sum()),
        "URL_blank_string": int(url_raw.fillna("").str.strip().eq("").sum()),
        "Label_null": int(label_raw.isna().sum()),
        "Label_blank_string": int(label_raw.fillna("").str.strip().eq("").sum()),
    }

    label_trim = label_raw.fillna("").str.strip()
    out["labels_exact_raw"] = {str(k): int(v) for k, v in label_trim.value_counts(dropna=False).items()}
    label_norm = label_trim.str.lower()
    out["labels_normalized"] = {str(k): int(v) for k, v in label_norm.value_counts(dropna=False).items()}
    out["rows_removed_invalid_label_or_empty_url"] = int(
        (~label_norm.isin(LABELS) | url_raw.fillna("").str.strip().eq("")).sum())
    bad_rate_raw = float((label_norm == "bad").mean())
    out["bad_rate_raw"] = round(bad_rate_raw, 6)
    out["bad_rate_note"] = "rasio pada baris mentah; label tak valid ikut terhitung sebagai bukan bad"

    df2 = df.assign(_u=url_raw.fillna("").str.strip(), _l=label_norm)
    df2 = df2[df2["_u"].ne("") & df2["_l"].isin(LABELS)].copy()

    out["duplicates"] = {"exact_url_string_rows_removed": int(df2.duplicated("_u").sum())}
    df2["_clean"] = df2["_u"].map(normalize_url)
    df2 = df2[df2["_clean"].ne("")].copy()
    canon_dupes = int(df2.duplicated("_clean").sum())
    out["duplicates"]["canonical_url_rows_removed"] = canon_dupes

    conflicts = df2.groupby("_clean")["_l"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    out["conflicts"] = {
        "canonical_urls_with_both_labels": len(conflict_keys),
        "rows_dropped_by_conflict_policy": int(df2["_clean"].isin(conflict_keys).sum()),
        "examples": sorted(conflict_keys)[:5],
    }
    df3 = df2[~df2["_clean"].isin(conflict_keys)].copy()
    before = len(df3)
    df3 = df3.drop_duplicates("_clean", keep="first").reset_index(drop=True)

    out["clean"] = {
        "rows_clean": int(len(df3)),
        "label_counts": {str(k): int(v) for k, v in df3["_l"].value_counts().items()},
        "bad_rate": round(float((df3["_l"] == "bad").mean()), 6),
        "unique_hosts": int(df3["_clean"].map(host_of).replace("", pd.NA).nunique()),
        "unique_urls": int(len(df3)),
        "host_dedup_removes_extra_rows": int(before - len(df3)) - canon_dupes,
    }

    has_scheme = df3["_clean"].str.contains("://")
    sch = df3["_clean"][has_scheme].str.split("://", n=1).str[0].str.lower()
    out["scheme"] = {
        "has_scheme_rows": int(has_scheme.sum()),
        "missing_scheme_rows": int((~has_scheme).sum()),
        "missing_scheme_pct": round(float((~has_scheme).mean() * 100), 4),
        "scheme_counts": {str(k): int(v) for k, v in sch.value_counts().items()},
        "https_rows": int((sch == "https").sum()),
        "http_rows": int((sch == "http").sum()),
        "https_pct_of_dataset": round(float((sch == "https").mean() * 100), 4),
        "http_pct_of_dataset": round(float((sch == "http").mean() * 100), 4),
    }

    host_series = df3["_clean"].map(host_of)
    empty_host = host_series.eq("")
    ws = host_series[~empty_host]
    out["url_validity"] = {
        "empty_host_rows": int(empty_host.sum()),
        "no_dot_in_host_rows": int((~empty_host & ws.eq("")).sum()),
        "host_with_space_or_control": int(df3["_clean"].str.contains(CONTROL).sum()),
        "no_dot_in_host": int((~empty_host & ~ws.str.contains(".")).sum()),
        "ip_literal_hosts": int(ws.str.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}").sum()),
        "unicode_hosts": int(ws.str.contains(r"[^\x00-\x7f]").sum()),
        "userinfo_at_rows": int(df3["_clean"].str.contains("@").sum()),
        "query_rows": int(df3["_clean"].str.contains(r"\?").sum()),
        "path_rows": int(df3["_clean"].str.contains("/").sum()),
        "avg_url_len": round(float(df3["_clean"].str.len().mean()), 2),
        "max_url_len": int(df3["_clean"].str.len().max()),
        "url_len_over_2000": int((df3["_clean"].str.len() > 2000).sum()),
    }
    out["url_validity"]["no_dot_in_host"] = out["url_validity"].pop("no_dot_in_host_rows")
    out["url_validity"]["no_dot_in_host_rows"] = out["url_validity"].pop("no_dot_in_host")

    out["label_balance"] = {
        "good": int((df3["_l"] == "good").sum()),
        "bad": int((df3["_l"] == "bad").sum()),
        "good_pct": round(float((df3["_l"] == "good").mean() * 100), 3),
        "bad_pct": round(float((df3["_l"] == "bad").mean() * 100), 3),
        "imbalance_ratio_good_over_bad": round(
            float((df3["_l"] == "good").sum() / max((df3["_l"] == "bad").sum(), 1)), 3),
    }

    tmp = df3[["_clean", "_l"]].copy()
    tmp.to_csv(BASE_DIR / "_audit_clean_urls.csv", index=False, encoding="utf-8")
    out["artifact"] = "_audit_clean_urls.csv (url_clean,label) untuk audit split"

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()