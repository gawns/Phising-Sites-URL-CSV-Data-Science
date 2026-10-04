r"""Pipeline v3: canonical dedup + registrable-domain train/val/test.

  python py\secure_pipeline.py audit phishing_site_urls.csv
  python py\secure_pipeline.py train phishing_site_urls.csv
  python py\secure_pipeline.py predict URL [URL ...]

Model dipilih memakai validation set. Final test hanya dipakai sekali setelah
model dibekukan. Artefak yang disimpan adalah model yang dievaluasi pada test.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import platform
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent  # root project (folder di atas py/)
LAPORAN_DIR = BASE_DIR / "laporan"
LAPORAN_DIR.mkdir(exist_ok=True)
CSV_DEFAULT = BASE_DIR / "phishing_site_urls.csv"
MODEL_PATH = BASE_DIR / "phishing_model_v3.joblib"
REPORT_PATH = LAPORAN_DIR / "training_report_v3.json"
SELECTION_PATH = LAPORAN_DIR / "model_selection_v3.csv"
FINAL_PATH = LAPORAN_DIR / "final_evaluation_v3.csv"
LABELS = ["good", "bad"]
PIPELINE_VERSION = "domain-grouped-v3"
# Gunakan snapshot Public Suffix List bawaan paket; jangan mengakses jaringan.
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_url(value: object) -> str:
    """Representasi konsisten untuk deduplikasi, TF-IDF, dan inference."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("#", 1)[0]  # fragment tidak pernah dikirim ke server
    return value.rstrip("/")


def hostname(url: str) -> str:
    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def registrable_domain(host: str, fallback_key: str) -> str:
    """eTLD+1 dari PSL; IP/host tanpa suffix tetap dikelompokkan secara aman."""
    if not host:
        return "__malformed__:" + hashlib.sha1(fallback_key.encode("utf-8", "ignore")).hexdigest()
    try:
        ipaddress.ip_address(host)
        return "ip:" + host
    except ValueError:
        pass
    result = TLD_EXTRACT(host)
    domain = result.top_domain_under_public_suffix
    return domain or host


def extract_features(url: str) -> dict:
    """Baseline fitur struktural. Scheme tidak dipakai karena hampir selalu hilang."""
    host = hostname(url)
    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        path = parsed.path or ""
    except ValueError:
        path = ""
    return {
        "url_length": len(url), "host_length": len(host), "path_length": len(path),
        "num_dots": url.count("."), "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url), "num_slashes": url.count("/"),
        "num_params": url.count("="), "has_at": int("@" in url),
        "has_ip": int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))),
        "num_subdomains": max(host.count(".") - 1, 0),
    }


def load_clean_data(csv_path: Path):
    raw = pd.read_csv(csv_path, dtype={"URL": "string", "Label": "string"})
    if not {"URL", "Label"}.issubset(raw.columns):
        raise ValueError("CSV harus memiliki kolom URL dan Label")
    rows_raw = len(raw)
    df = raw[["URL", "Label"]].copy()
    df["URL"] = df["URL"].fillna("").astype(str).str.strip()
    df["Label"] = df["Label"].fillna("").astype(str).str.strip().str.lower()
    invalid = ~df["Label"].isin(LABELS) | df["URL"].eq("")
    invalid_rows = int(invalid.sum())
    df = df.loc[~invalid].copy()
    df["url_clean"] = df["URL"].map(normalize_url)
    empty_clean = int(df["url_clean"].eq("").sum())
    df = df.loc[df["url_clean"].ne("")].copy()

    conflicts = df.groupby("url_clean")["Label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    conflict_rows = int(df["url_clean"].isin(conflict_keys).sum())
    df = df.loc[~df["url_clean"].isin(conflict_keys)].copy()
    before_dedupe = len(df)
    df = df.drop_duplicates("url_clean", keep="first").reset_index(drop=True)
    duplicate_rows = before_dedupe - len(df)
    df["target"] = (df["Label"] == "bad").astype(int)
    df["host"] = df["url_clean"].map(hostname)
    unique_hosts = df[["host", "url_clean"]].drop_duplicates("host")
    host_groups = {row.host: registrable_domain(row.host, row.url_clean)
                   for row in unique_hosts.itertuples(index=False)}
    # URL malformed memakai hash URL sendiri sehingga tidak membentuk satu grup raksasa.
    malformed = df["host"].eq("")
    df["group"] = df["host"].map(host_groups)
    df.loc[malformed, "group"] = df.loc[malformed, "url_clean"].map(
        lambda value: registrable_domain("", value))

    stats = {
        "rows_raw": rows_raw,
        "rows_removed_invalid_label_or_empty_url": invalid_rows + empty_clean,
        "conflicting_canonical_urls": len(conflict_keys),
        "rows_removed_conflicting_canonical_urls": conflict_rows,
        "duplicate_rows_removed_after_canonicalization": int(duplicate_rows),
        "rows_clean": len(df),
        "unique_hosts": int(df["host"].nunique()),
        "unique_registrable_domain_groups": int(df["group"].nunique()),
        "label_counts": df["Label"].value_counts().to_dict(),
        "malformed_or_empty_host_rows": int(malformed.sum()),
    }
    return df, stats


def select_balanced_fold(df: pd.DataFrame, n_splits: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target_size = len(df) / n_splits
    prevalence = df["target"].mean()
    candidates = list(splitter.split(df["url_clean"], df["target"], groups=df["group"]))
    return min(candidates, key=lambda pair:
               abs(len(pair[1]) - target_size) / target_size
               + 4 * abs(df.iloc[pair[1]]["target"].mean() - prevalence))


def make_three_way_split(df: pd.DataFrame):
    dev_idx, test_idx = select_balanced_fold(df, n_splits=5, seed=42)
    dev = df.iloc[dev_idx].reset_index(drop=False).rename(columns={"index": "original_index"})
    train_local, val_local = select_balanced_fold(dev, n_splits=4, seed=84)
    train_idx = dev.iloc[train_local]["original_index"].to_numpy()
    val_idx = dev.iloc[val_local]["original_index"].to_numpy()
    return train_idx, val_idx, test_idx


def model_factory(name: str):
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": lambda: MultinomialNB(),
        "Linear SVM": lambda: LinearSVC(class_weight="balanced"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                          class_weight="balanced", random_state=42),
    }[name]()


def scores_for(model, x):
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return None


def evaluate(name, stage, y_true, y_pred, scores, feature_kind):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "Model": name, "Stage": stage, "Fitur": feature_kind, "N": int(len(y_true)),
        "Precision (bad)": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall (bad)": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 (bad)": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Akurasi": round(accuracy_score(y_true, y_pred), 4),
        "Balanced Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
        "FP/100k": round(fp / len(y_true) * 100000, 1),
        "FN/100k": round(fn / len(y_true) * 100000, 1),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if scores is not None:
        result["ROC-AUC"] = round(roc_auc_score(y_true, scores), 4)
        result["PR-AUC"] = round(average_precision_score(y_true, scores), 4)
    return result


def audit(csv_path: Path):
    df, stats = load_clean_data(csv_path)
    stats["dataset_sha256"] = sha256_file(csv_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Canonical duplicate tersisa:", int(df.duplicated("url_clean").sum()))
    return stats


def train(csv_path: Path):
    df, stats = load_clean_data(csv_path)
    train_idx, val_idx, test_idx = make_three_way_split(df)
    parts = {"train": df.iloc[train_idx], "validation": df.iloc[val_idx], "test": df.iloc[test_idx]}
    group_sets = {name: set(part["group"]) for name, part in parts.items()}
    overlaps = {"train_validation": len(group_sets["train"] & group_sets["validation"]),
                "train_test": len(group_sets["train"] & group_sets["test"]),
                "validation_test": len(group_sets["validation"] & group_sets["test"])}
    if any(overlaps.values()):
        raise RuntimeError(f"Domain leakage ditemukan: {overlaps}")
    split_info = {name: {"rows": len(part), "bad_rate": round(part["target"].mean(), 4),
                         "domain_groups": int(part["group"].nunique())}
                  for name, part in parts.items()}
    split_info["overlap"] = overlaps
    print(json.dumps({"dataset": stats, "split": split_info}, indent=2, ensure_ascii=False))

    # Model selection: hanya train -> validation.
    selector_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=300_000,
                                   min_df=2, sublinear_tf=True)
    x_train = selector_vec.fit_transform(parts["train"]["url_clean"])
    x_val = selector_vec.transform(parts["validation"]["url_clean"])
    feature_cols = list(extract_features("example.com").keys())
    structural = pd.DataFrame(df["url_clean"].map(extract_features).tolist(), columns=feature_cols)
    selection_results = []
    for name in ["Logistic Regression", "Naive Bayes", "Linear SVM", "Random Forest"]:
        print(f"Model selection: {name}...")
        model = model_factory(name)
        if name == "Random Forest":
            model.fit(structural.iloc[train_idx], parts["train"]["target"])
            x_eval = structural.iloc[val_idx]
        else:
            model.fit(x_train, parts["train"]["target"])
            x_eval = x_val
        pred = model.predict(x_eval)
        selection_results.append(evaluate(name, "validation", parts["validation"]["target"],
                                          pred, scores_for(model, x_eval),
                                          "Fitur buatan" if name == "Random Forest" else "TF-IDF char n-gram"))
    selection = pd.DataFrame(selection_results)
    selection.to_csv(SELECTION_PATH, index=False)
    best = max(selection_results, key=lambda row: (row["F1 (bad)"], row["Recall (bad)"], row["Precision (bad)"]))
    best_name = best["Model"]
    print("\n=== Model selection (validation only) ===\n", selection.to_string(index=False))
    print("Selected:", best_name)

    # Final model: fit train+validation, lalu buka test satu kali.
    dev_idx = pd.Index(train_idx).append(pd.Index(val_idx)).to_numpy()
    dev = df.iloc[dev_idx]
    final_model = model_factory(best_name)
    final_vec = None
    if best_name == "Random Forest":
        x_dev, x_test = structural.iloc[dev_idx], structural.iloc[test_idx]
    else:
        final_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=300_000,
                                    min_df=2, sublinear_tf=True)
        x_dev = final_vec.fit_transform(dev["url_clean"])
        x_test = final_vec.transform(parts["test"]["url_clean"])
    final_model.fit(x_dev, dev["target"])
    final_pred = final_model.predict(x_test)
    final_result = evaluate(best_name, "final_test", parts["test"]["target"], final_pred,
                            scores_for(final_model, x_test),
                            "Fitur buatan" if best_name == "Random Forest" else "TF-IDF char n-gram")
    pd.DataFrame([final_result]).to_csv(FINAL_PATH, index=False)
    print("\n=== Final untouched test ===\n", pd.DataFrame([final_result]).to_string(index=False))

    bundle = {"name": best_name, "pipeline_version": PIPELINE_VERSION,
              "features": best_name == "Random Forest", "feature_columns": feature_cols,
              "model": final_model, "vectorizer": final_vec, "trained_on_rows": len(dev),
              "test_rows_not_trained": len(parts["test"]), "training_stats": stats,
              "split": split_info, "final_metrics": final_result}
    joblib.dump(bundle, MODEL_PATH, compress=3)

    cm = confusion_matrix(parts["test"]["target"], final_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Final unseen-domain test")
    fig.tight_layout(); fig.savefig(LAPORAN_DIR / "cm_final_v3.png", dpi=120); plt.close(fig)

    report = {"pipeline_version": PIPELINE_VERSION,
              "dataset": {**stats, "sha256": sha256_file(csv_path)},
              "split": split_info, "model_selection": selection_results,
              "selected_model": best_name, "final_test": final_result,
              "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                              "scikit_learn": sklearn.__version__, "tldextract": tldextract.__version__},
              "warning": "Prediksi statistik, bukan jaminan keamanan. Tidak ada temporal holdout karena dataset tidak memiliki timestamp."}
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", MODEL_PATH)


def predict(urls):
    bundle = joblib.load(MODEL_PATH)
    for raw in urls:
        raw = raw.strip()
        x = (pd.DataFrame([extract_features(normalize_url(raw))], columns=bundle["feature_columns"])
             if bundle["features"] else bundle["vectorizer"].transform([normalize_url(raw)]))
        pred = int(bundle["model"].predict(x)[0])
        print(f"{raw}\n  -> {'bad (phishing)' if pred else 'good (aman)'}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"audit", "train", "predict"}:
        print(__doc__); raise SystemExit(2)
    command = sys.argv[1]
    csv_path = Path(sys.argv[2]) if command in {"audit", "train"} and len(sys.argv) > 2 else CSV_DEFAULT
    if command == "audit": audit(csv_path)
    elif command == "train": train(csv_path)
    elif len(sys.argv) >= 3: predict(sys.argv[2:])
    else: raise SystemExit("Berikan minimal satu URL")
