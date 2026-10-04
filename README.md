# Phishing URL Detector

Aplikasi klasifikasi URL berdasarkan dataset `phishing_site_urls.csv` dan alur pada dokumen.

## Struktur Folder

- `py/`: kode Python — `phishing_app.py` (EDA, training, prediksi CLI), `web_app.py` (web + JSON API), `secure_pipeline.py` (pipeline v3), `verify_split.py`, `audit_raw.py`.
- `laporan/`: data laporan — `training_report_v3.json`, `audit_raw.json`, grafik PNG, tabel seleksi/evaluasi model.
- `logtxtetc/`: log berformat teks — `audit_run.log`, `training_v3.log`, `verify_split.log`.
- Root: dataset CSV (`phishing_site_urls.csv`), `phishing_model.joblib`, dan `model_comparison.csv` (keluaran training).

## Instalasi

```powershell
cd C:\Users\USER\Downloads\datasetttttt datasains
python -m pip install -r requirements.txt
```

## Jalankan web app

Klik `run_web.bat`, atau:

```powershell
python web_app.py
```

Lalu buka http://127.0.0.1:8000.

## Training ulang

```powershell
python py\phishing_app.py train phishing_site_urls.csv
```

Training dataset penuh memerlukan RAM dan waktu cukup besar.

## Prediksi melalui CLI

```powershell
python py\phishing_app.py predict "https://www.wikipedia.org" "http://secure-login-paypal.verify-account.xyz/update"
```

## JSON API

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d '{"urls":["https://www.wikipedia.org"]}'
```

## Validasi & Pembersihan Data

Hasil audit `phishing_site_urls.csv` (SHA256 `23a5800c…e8262`, identik dengan `training_report_v3.json`):

| Tahap | Hasil |
|---|---|
| Baris mentah | 549.346 (good 392.924, bad 156.422) |
| URL kosong / label invalid | 1 baris dibuang |
| URL kanonik dengan label ganda | 1 kasus (`tommyhumphreys.com/`, good vs bad) — 3 baris dibuang |
| Duplikat setelah kanonikalisasi | 47.151 baris dihapus |
| Baris bersih | 502.191 (good 392.879 = 78,2%, bad 109.312 = 21,8%) |

Pipeline v3 (`secure_pipeline.py`) memakai split **train / validation / test** berbasis registrable domain (eTLD+1 via Public Suffix List):

- Train 301.315, validation 100.439, test 100.437 baris — bad rate merata 21,77% di semua split.
- Overlap domain antar split terverifikasi **0** (train–validation, train–test, validation–test); training berhenti otomatis bila leakage terdeteksi.
- Model dipilih hanya lewat validation set; test set dibuka satu kali setelah model dibekukan.

Model terpilih: **Linear SVM** (TF-IDF char n-gram). Di test set: F1 kelas bad 0,9201, recall 0,9057, precision 0,9350, ROC-AUC 0,9915.

## Permasalahan & Keterbatasan

1. **Duplikat timpang antar kelas** — 83.685 baris duplikat berasal dari kelas bad, hanya 86 dari good. Bad rate turun 28,5% → 21,8% setelah dedup. Tanpa dedup, model akan "menghafal" URL phishing populer dan terlihat terlalu bagus secara palsu.
2. **Konflik label** — satu URL diberi label good sekaligus bad; ambigu, sehingga seluruh barisnya dibuang, bukan dipilih sepihak.
3. **Bias scheme** — 99,98% URL tanpa `http(s)://`, jadi fitur `is_https` praktis tidak berguna dan bisa jadi sinyal palsu; pipeline v3 menghapus scheme sebelum TF-IDF.
4. **Tanpa holdout waktu** — dataset tidak punya timestamp, sehingga tidak bisa diuji pada "phishing masa depan". Skor test kemungkinan optimistis untuk URL benar-benar baru.
5. **Ketidakseimbangan kelas** — rasio good:bad = 3,59:1. Semua model memakai `class_weight="balanced"`; F1 kelas bad yang jadi kriteria pemilihan karena false negative (phishing lolos) lebih berbahaya.
6. **Random Forest lemah** — fitur struktural buatan hanya mencapai F1 0,6427, jauh di bawah TF-IDF; pada dataset ini pola teks URL jauh lebih informatif daripada statistik panjang/karakter.
7. **Prediksi statistik** — hasil model bukan jaminan keamanan; jangan dijadikan satu-satunya dasar keputusan keamanan.
