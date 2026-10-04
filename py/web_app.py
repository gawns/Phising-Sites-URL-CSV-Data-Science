"""Web app deteksi URL phishing.

Lokal : python py/web_app.py   -> http://127.0.0.1:8000
Vercel: entrypoint py.web_app:Handler (lihat pyproject.toml)
"""
import html
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # agar "from phishing_app" ketemu saat dijalankan dari root
from phishing_app import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent  # root project (folder di atas py/)
MODEL_PATH = BASE_DIR / "phishing_model.joblib"
RESULT_PATH = BASE_DIR / "model_comparison.csv"
REPORT_PATH = BASE_DIR / "laporan" / "training_report_v3.json"

PAGE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phishing URL Detector</title>
<style>
:root{--navy:#0b1220;--card:#111c30;--cyan:#28d7e5;--text:#e9f3ff;--muted:#91a4bd;--bad:#ff647c;--good:#54e09a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 10%,#163055 0,var(--navy) 40%);color:var(--text);font:16px system-ui,Segoe UI,sans-serif;min-height:100vh}.wrap{max-width:980px;margin:auto;padding:42px 20px}h1{font-size:clamp(30px,5vw,55px);margin:0}h1 span{color:var(--cyan)}.lead{color:var(--muted);font-size:18px}.card{background:rgba(17,28,48,.92);border:1px solid #263854;border-radius:18px;padding:24px;margin-top:25px;box-shadow:0 18px 55px #03060c88}textarea{width:100%;min-height:155px;background:#08111f;border:1px solid #334b6d;border-radius:12px;color:white;padding:15px;font:15px ui-monospace,Consolas;resize:vertical}button{background:linear-gradient(135deg,var(--cyan),#4694ff);border:0;border-radius:10px;padding:13px 22px;font-weight:800;color:#05101d;cursor:pointer;margin-top:12px}button:hover{filter:brightness(1.1)}.result{padding:13px 15px;margin-top:10px;border-radius:10px;word-break:break-all}.result.bad{background:#451c2a;border-left:5px solid var(--bad)}.result.good{background:#163a30;border-left:5px solid var(--good)}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px;border-bottom:1px solid #2b3b54;text-align:right}th:first-child,td:first-child{text-align:left}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stat{background:#0a1526;padding:16px;border-radius:12px}.stat b{display:block;font-size:25px;color:var(--cyan)}small,.note{color:var(--muted)}@media(max-width:650px){.stats{grid-template-columns:1fr}table{font-size:12px}.card{padding:16px}}
</style></head>
<body><main class="wrap">
<h1>Phishing URL <span>Detector</span></h1>
<p class="lead">Klasifikasi URL <b>good</b> atau <b>bad</b> menggunakan TF-IDF character n-gram dan model terbaik.</p>
<section class="card">
<h2>Periksa URL</h2><p class="note">Satu URL per baris. Hasil model adalah prediksi, bukan jaminan keamanan.</p>
<form method="post"><textarea name="urls" placeholder="https://www.wikipedia.org&#10;http://secure-login-paypal.verify-account.xyz/update">{{INPUT}}</textarea><br><button type="submit">Analisis URL</button></form>
{{RESULTS}}
</section>
<section class="card"><h2>Ringkasan Dataset</h2><div class="stats"><div class="stat"><b>549.346</b>baris awal</div><div class="stat"><b>47.151</b>duplikat URL</div><div class="stat"><b>502.191</b>baris setelah deduplikasi</div></div></section>
<section class="card"><h2>Evaluasi Model</h2>{{TABLE}}<p class="note">Model terpilih: <b>{{MODEL}}</b>. Prioritas evaluasi adalah recall/F1 kelas bad karena false negative lebih berbahaya.</p></section>
</main></body></html>"""


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.bundle = None

    def _load(self):
        if self.bundle is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model tidak ditemukan di {self.model_path}. "
                    "Jalankan training atau pastikan file phishing_model.joblib ikut ter-deploy."
                )
            self.bundle = joblib.load(self.model_path)  # lazy-load: aman untuk serverless (Vercel)
        return self.bundle

    @property
    def name(self):
        return self._load()["name"]

    def predict(self, url):
        bundle = self._load()
        original = url.strip()
        cleaned = re.sub(r"^https?://", "", original.lower())
        if bundle["features"]:
            x = pd.DataFrame([extract_features(original.lower())])
        else:
            x = bundle["vectorizer"].transform([cleaned])
        prediction = int(bundle["model"].predict(x)[0])
        return {"url": original, "label": "bad" if prediction else "good", "phishing": bool(prediction)}


PREDICTOR = Predictor(MODEL_PATH)  # tidak memuat model saat import -> cold start Vercel tetap bisa jalan


def comparison_table():
    if RESULT_PATH.exists():
        df = pd.read_csv(RESULT_PATH)
        headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
        rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
            rows.append(f"<tr>{cells}</tr>")
        return f"<div style='overflow:auto'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    if REPORT_PATH.exists():  # fallback: hasil evaluasi dari laporan training v3
        data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        candidates = data.get("model_selection", []) + ([data["final_test"]] if "final_test" in data else [])
        cols = ["Model", "Fitur", "Precision (bad)", "Recall (bad)", "F1 (bad)", "Akurasi", "ROC-AUC"]
        headers = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        rows = []
        for rec in candidates:
            cells = "".join(f"<td>{html.escape(str(rec.get(c, '-')))}</td>" for c in cols)
            rows.append(f"<tr>{cells}</tr>")
        return f"<div style='overflow:auto'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    return "<p>Hasil evaluasi belum tersedia.</p>"


def render(raw=""):
    urls = [line.strip() for line in raw.splitlines() if line.strip()][:50]
    blocks = []
    for url in urls:
        result = PREDICTOR.predict(url)
        label = "BAD — terindikasi phishing" if result["phishing"] else "GOOD — terindikasi aman"
        blocks.append(f'<div class="result {result["label"]}"><b>{label}</b><br><small>{html.escape(url)}</small></div>')
    return (PAGE.replace("{{INPUT}}", html.escape(raw))
                .replace("{{RESULTS}}", "".join(blocks))
                .replace("{{TABLE}}", comparison_table())
                .replace("{{MODEL}}", html.escape(PREDICTOR.name)))


class Handler(BaseHTTPRequestHandler):
    def send_content(self, status, body, content_type="text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.send_content(200, render())
        elif self.path == "/api/health":
            self.send_content(200, json.dumps({"status": "ok", "model": PREDICTOR.name}), "application/json")
        else:
            self.send_content(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path == "/":
            length = int(self.headers.get("Content-Length", 0))
            values = parse_qs(self.rfile.read(length).decode("utf-8"))
            self.send_content(200, render(values.get("urls", [""])[0]))
        elif self.path == "/api/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                urls = payload.get("urls", [payload.get("url", "")])
                if isinstance(urls, str):
                    urls = [urls]
                results = [PREDICTOR.predict(str(url)) for url in urls if str(url).strip()][:50]
                self.send_content(200, json.dumps({"model": PREDICTOR.name, "results": results}), "application/json")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_content(400, json.dumps({"error": str(exc)}), "application/json")
        else:
            self.send_content(404, "Not found", "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.environ.get("PORT", 8000))
    print(f"Phishing URL Detector berjalan di http://{host}:{port}")
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
