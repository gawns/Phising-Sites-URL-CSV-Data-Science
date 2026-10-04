r"""Verifikasi independen atas klaim pipeline v3.

  python py\verify_split.py > logtxtetc\verify_split.log 2>&1

Memakai fungsi yang sama dengan secure_pipeline, lalu MENGHITUNG ULANG overlap
dari host/registrable-domain secara terpisah (tidak mempercayai kolom `group`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))  # agar import secure_pipeline tetap ketemu saat dijalankan dari root
import pandas as pd

from secure_pipeline import (BASE_DIR, LABELS, load_clean_data, make_three_way_split,
                            normalize_url, registrable_domain, sha256_file)

CSV = BASE_DIR / "phishing_site_urls.csv"
OUT = BASE_DIR / "laporan" / "verify_split.json"

raw = pd.read_csv(CSV, dtype="string")
raw["URL"] = raw["URL"].fillna("").astype(str)
raw["Label"] = raw["Label"].fillna("").astype(str).str.strip().str.lower()

res = {"csv_sha256": sha256_file(CSV), "rows_raw": int(len(raw))}

# --- provenance angka 42.151 / 507.195 di web_app.py -------------------------
exact_dupes = int(raw.duplicated("URL").sum())
res["legacy_figures"] = {
    "exact_string_duplicate_rows": exact_dupes,
    "rows_after_exact_string_dedup": int(len(raw) - exact_dupes),
}
stripped = raw.assign(_u=raw["URL"].str.strip())
res["legacy_figures"]["exact_duplicate_rows_after_strip"] = int(stripped.duplicated("_u").sum())
res["legacy_figures"]["note"] = ("web_app.py menampilkan 42.151/507.195 = hitungan v1 "
                                 "(duplikat string persis). Pipeline v3 memakai kanonik "
                                 "sehingga angkanya berbeda.")

# --- baris yang hilang di tahap cleaning ------------------------------------
canon = raw["URL"].map(normalize_url)
res["rows_lost_to_empty_canonical"] = int((raw["URL"].str.strip().ne("") & canon.eq("")).sum())
res["rows_lost_to_empty_canonical_examples"] = sorted(
    raw.loc[raw["URL"].str.strip().ne("") & canon.eq(""), "URL"].str.strip().unique().tolist())[:5]

# --- scheme -----------------------------------------------------------------
head = raw["URL"].str.strip().str.lower()
res["scheme"] = {
    "starts_with_http_colon_slash_slash": int(head.str.startswith("http://").sum()),
    "starts_with_https_colon_slash_slash": int(head.str.startswith("https://").sum()),
    "contains_scheme_anywhere": int(head.str.contains("://").sum()),
    "starts_with_any_scheme": int(head.str.match(r"^[a-z][a-z0-9+.\-]*://").sum()),
}
inner = head[head.str.contains("://") & ~head.str.startswith(("http://", "https://"))]
res["scheme"]["contains_but_not_at_start_examples"] = inner.unique().tolist()[:5]
res["scheme"]["is_https_feature_variance"] = 0 if res["scheme"]["starts_with_https_colon_slash_slash"] == 0 else 1

# --- duplikasi per label (mengapa rasio turun dari 28,5% ke 21,8%) ----------
dup_key = raw.assign(_c=canon)
before = dup_key["Label"].value_counts()
uniq_label = dup_key.loc[~dup_key["_c"].duplicated(keep="first")].assign(
    _conflict=lambda d: d["_c"].map(dup_key.groupby("_c")["Label"].nunique()) > 1)
kept = uniq_label[~uniq_label["_conflict"]]
after = kept["Label"].value_counts()
res["dedup_effect_per_label"] = {
    lab: {"raw": int(before.get(lab, 0)), "setelah_dedup_bersih": int(after.get(lab, 0)),
          "hilang": int(before.get(lab, 0) - after.get(lab, 0)),
          "pct_hilang": round(float(1 - after.get(lab, 0) / max(before.get(lab, 0), 1)) * 100, 2)}
    for lab in LABELS}
res["dedup_effect_per_label"]["bad_rate_raw"] = round(float((raw["Label"] == "bad").mean()), 6)
res["dedup_effect_per_label"]["bad_rate_clean"] = round(float((kept["Label"] == "bad").mean()), 6)

# --- konflik label ----------------------------------------------------------
conf = dup_key.groupby("_c")["Label"].agg(["nunique", "count"])
conflicting = conf[conf["nunique"] > 1]
res["label_conflicts"] = {
    "canonical_urls": int(len(conflicting)),
    "rows_affected": int(conflicting["count"].sum()),
    "details": dup_key[dup_key["_c"].isin(conflicting.index)]
                .groupby(["_c", "Label"]).size().to_dict().__repr__()[:800],
}

# --- split: hitung ulang independen -----------------------------------------
df, stats = load_clean_data(CSV)
train_idx, val_idx, test_idx = make_three_way_split(df)
parts = {"train": df.iloc[train_idx].copy(), "validation": df.iloc[val_idx].copy(),
         "test": df.iloc[test_idx].copy()}


def reg(host: str) -> str:
    return registrable_domain(host, "")


for name, part in parts.items():
    part["_host"] = part["url_clean"].map(
        lambda u: (urlsplit(u if "://" in u else "//" + u).hostname or "").lower().rstrip("."))
    part["_reg"] = part["_host"].map(reg)

dom_sets = {n: set(p["_reg"]) for n, p in parts.items()}
host_sets = {n: set(p["_host"]) - {""} for n, p in parts.items()}
res["split_recomputed"] = {
    "rows": {n: int(len(p)) for n, p in parts.items()},
    "bad_rate": {n: round(float(p["target"].mean()), 6) for n, p in parts.items()},
    "registrable_domain_overlap": {
        "train_validation": len(dom_sets["train"] & dom_sets["validation"]),
        "train_test": len(dom_sets["train"] & dom_sets["test"]),
        "validation_test": len(dom_sets["validation"] & dom_sets["test"])},
    "host_overlap": {
        "train_validation": len(host_sets["train"] & host_sets["validation"]),
        "train_test": len(host_sets["train"] & host_sets["test"]),
        "validation_test": len(host_sets["validation"] & host_sets["test"])},
    "index_sets_disjoint": bool(len(set(train_idx) & set(val_idx)) == 0
                                and len(set(train_idx) & set(test_idx)) == 0
                                and len(set(val_idx) & set(test_idx)) == 0
                                and len(train_idx) + len(val_idx) + len(test_idx) == len(df)),
}
# sertakan baris dari setiap split yang terduplikasi antar split (bukti, bukan klaim)
cross = []
for a in parts:
    for b in parts:
        if a < b:
            shared = dom_sets[a] & dom_sets[b]
            if shared:
                cross.append({f"{a}&{b}": sorted(shared)[:5]})
res["split_recomputed"]["leaked_domain_examples"] = cross

# --- distribusi ukuran grup: apakah split acakDn sunkhorn -------------------
sizes = df["group"].value_counts()
mixed = df.groupby("group")["target"].nunique()
res["group_structure"] = {
    "n_registrable_domains": int(df["group"].nunique()),
    "max_urls_in_one_group": int(sizes.max()),
    "groups_with_more_than_100_urls": int((sizes > 100).sum()),
    "groups_with_mixed_labels": int((mixed > 1).sum()),
    "rows_in_mixed_label_groups": int(df["group"].isin(mixed[mixed > 1].index).sum()),
    "effective_sample_note": ("efektif kecil: 502.191 baris berasal dari hanya "
                              f"{df['group'].nunique()} registrable domain"),
}

# --- artefak v3 -------------------------------------------------------------
res["artifacts"] = {
    name: {"exists": (BASE_DIR / name).exists(),
           "size": (BASE_DIR / name).stat().st_size if (BASE_DIR / name).exists() else 0}
    for name in ["phishing_model_v3.joblib", "training_report_v3.json", "model_selection_v3.csv",
                 "final_evaluation_v3.csv", "cm_final_v3.png", "phishing_model.joblib",
                 "model_comparison.csv", "training_report_v3.json"]}
res["artifacts"]["parent_dir_of_this_folder"] = sorted(
    p.name for p in BASE_DIR.parent.glob("*v3*"))

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))