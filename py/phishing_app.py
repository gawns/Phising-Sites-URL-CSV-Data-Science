r"""Klasifikasi Phishing Site URLs — pipeline sesuai dokumen alur kerja.

Pakai:
  python py\phishing_app.py train [path_csv]   -> latih semua model, simpan best ke model.joblib
  python py\phishing_app.py predict URL [URL..] -> prediksi URL baru dengan model terbaik
"""
import sys, re, joblib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent.parent  # root project (folder di atas py/)
LAPORAN_DIR = BASE_DIR / "laporan"
LAPORAN_DIR.mkdir(exist_ok=True)
CSV_DEFAULT = str(BASE_DIR / "phishing_site_urls.csv")
MODEL_PATH = str(BASE_DIR / "phishing_model.joblib")
LABELS = ["good", "bad"]


def extract_features(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc
        path = parsed.path
    except ValueError:  # URL tidak valid (mis. "http://[::1" tanpa penutup)
        host, path = "", ""
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_slashes": url.count("/"),
        "num_params": url.count("="),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.search(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "num_subdomains": max(host.count(".") - 1, 0),
        "is_https": int(url.startswith("https")),
    }


def train(csv_path):
    # 5. Load
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print("Missing value:\n", df.isnull().sum().to_string())

    # 6. EDA
    print("Duplikat URL:", df.duplicated(subset="URL").sum())
    print(df["Label"].value_counts().to_string())
    df["url_length"] = df["URL"].str.len()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x="Label", data=df, ax=ax[0])
    ax[0].set_title("Distribusi Label")
    sns.boxplot(x="Label", y="url_length", data=df, ax=ax[1])
    ax[1].set_yscale("log")
    ax[1].set_title("Panjang URL per Label")
    fig.savefig(Path(__file__).resolve().parent.parent / "laporan" / "eda.png", dpi=100)

    # 7. Preprocessing
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    df["URL"] = df["URL"].str.strip().str.lower()
    df["url_clean"] = df["URL"].str.replace(r"^https?://", "", regex=True)  # dataset 99.98% tanpa scheme; tanpa ini "https://" salah ajar jadi sinyal phishing (khusus TF-IDF; fitur buatan tetap pakai scheme utk is_https)
    df["target"] = (df["Label"] == "bad").astype(int)
    print("Setelah hapus duplikat:", len(df))

    # 9. Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["url_clean"], df["target"], test_size=0.2, stratify=df["target"], random_state=42)

    # 8.1 TF-IDF — fit hanya di data latih
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=300_000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 10. Training
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }
    results = []
    for name, m in models.items():
        print(f"Training {name}...")
        m.fit(X_train_vec, y_train)
        y_pred = m.predict(X_test_vec)
        results.append(evaluate(name, "TF-IDF char n-gram", y_test, y_pred))
        if name == "Logistic Regression":
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=LABELS)
            plt.title("Confusion Matrix - Logistic Regression")
            plt.savefig(Path(__file__).resolve().parent.parent / "laporan" / "cm_lr.png", dpi=100, bbox_inches="tight")
            plt.close()

    # 8.2 + 10.4 Random Forest dengan fitur buatan
    print("Extracting hand-crafted features...")
    features_df = pd.DataFrame(df["URL"].apply(extract_features).tolist())
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        features_df, df["target"], test_size=0.2, stratify=df["target"], random_state=42)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42)
    print("Training Random Forest...")
    rf.fit(Xf_train, yf_train)
    results.append(evaluate("Random Forest", "Fitur buatan", yf_test, rf.predict(Xf_test)))

    # 11.3 Tabel perbandingan
    table = pd.DataFrame(results).set_index("Model")
    print("\n=== Tabel Perbandingan Model ===")
    print(table.to_string())
    table.to_csv("model_comparison.csv")

    # Model terbaik = F1 kelas bad tertinggi (recall bad prioritaskan untuk phishing)
    best = max(results, key=lambda r: r["F1 (bad)"])
    best_model = models.get(best["Model"], rf)
    joblib.dump({"model": best_model, "vectorizer": vectorizer if best["Model"] != "Random Forest" else None,
                 "features": best["Model"] == "Random Forest", "name": best["Model"]}, MODEL_PATH)
    print(f"\nModel terbaik: {best['Model']} (F1 bad = {best['F1 (bad)']:.4f}) -> disimpan ke {MODEL_PATH}")
    print("Catatan: recall kelas 'bad' penting — false negative lebih berbahaya daripada false positive.")


def evaluate(name, feature_kind, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tn + tp) / cm.sum()
    print(f"\n{name}\n{classification_report(y_true, y_pred, target_names=LABELS)}")
    return {"Model": name, "Fitur": feature_kind, "Precision (bad)": round(prec, 4),
            "Recall (bad)": round(rec, 4), "F1 (bad)": round(f1, 4), "Akurasi": round(acc, 4)}


def predict(urls):
    bundle = joblib.load(MODEL_PATH)
    model, vec, use_features = bundle["model"], bundle["vectorizer"], bundle["features"]
    print(f"Model: {bundle['name']}")
    for u in urls:
        u = re.sub(r"^https?://", "", u.strip().lower())
        if use_features:
            pred = model.predict(pd.DataFrame([extract_features(u)]))[0]
        else:
            pred = model.predict(vec.transform([u]))[0]
        print(f"{u}\n  -> {'bad (phishing)' if pred == 1 else 'good (aman)'}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        sys.exit(__doc__)
    if sys.argv[1] == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else CSV_DEFAULT)
    else:
        if len(sys.argv) < 3:
            sys.exit("give at least one URL")
        predict(sys.argv[2:])
