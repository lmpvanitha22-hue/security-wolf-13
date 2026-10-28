"""
Phishing URL Detector — Flask Application
Combines model training, prediction pipeline, and web serving.
Returns separate risk scores for Logistic Regression and Random Forest.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Blacklist setup
# ---------------------------------------------------------------------------
BLACKLIST_DOMAINS = [
    "paypal-security-update.com",
    "login-secure-bank.net",
    "verify-amazon-account.info",
]
with open("blacklist.json", "w") as f:
    json.dump(BLACKLIST_DOMAINS, f, indent=4)


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["url_length", "domain_length", "qty_dot", "qty_hyphen", "is_ip", "has_at"]

def extract_features(url: str) -> dict:
    """Extract numeric features from a URL for ML classification."""
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else parsed.path
    return {
        "url_length":    len(url),
        "domain_length": len(domain),
        "qty_dot":       url.count("."),
        "qty_hyphen":    url.count("-"),
        "is_ip":         1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0,
        "has_at":        1 if "@" in url else 0,
    }


# ---------------------------------------------------------------------------
# 3. Training data & model fitting
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # --- Safe URLs (label 0) ---
    ("https://www.google.com",        0),
    ("https://www.github.com",        0),
    ("https://www.wikipedia.org",     0),
    ("https://www.nytimes.com",       0),
    ("https://www.amazon.com",        0),
    ("https://www.microsoft.com",     0),
    ("https://www.apple.com",         0),
    ("https://www.reddit.com",        0),
    ("https://www.linkedin.com",      0),
    ("https://www.stackoverflow.com", 0),
    # --- Phishing / suspicious URLs (label 1) ---
    ("https://login-secure-paypal-verification.com/webscr",  1),
    ("http://verify-your-bank-account-now.info/update",      1),
    ("http://192.168.1.105/index.php?login=true",            1),
    ("https://amazon-security-alert-center.com/login",       1),
    ("http://secure-update-signin@hotmail-verify.com",       1),
    ("http://get-free-coins-fast.xyz/claim",                 1),
    ("https://netflix-account-onhold.net/renew",             1),
    ("http://10.0.0.1/admin-login",                          1),
    ("https://apple-id-locked-verify.com/account",           1),
    ("http://free-gift-card-generator.online/redeem",        1),
]

X = pd.DataFrame([extract_features(url) for url, _ in TRAINING_DATA])
y = np.array([label for _, label in TRAINING_DATA])

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
MODEL_METRICS = {}
for _name, _model in (("Logistic Regression", LogisticRegression(max_iter=1000)),
                      ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))):
    _predictions = cross_val_predict(_model, X, y, cv=_cv)
    MODEL_METRICS[_name] = {
        "accuracy": round(accuracy_score(y, _predictions) * 100, 1),
        "precision": round(precision_score(y, _predictions, zero_division=0) * 100, 1),
        "recall": round(recall_score(y, _predictions, zero_division=0) * 100, 1),
        "f1_score": round(f1_score(y, _predictions, zero_division=0) * 100, 1),
    }

print("Models trained -- LR + RandomForest ready.")
print(f"   Training samples: {len(TRAINING_DATA)}  |  Features: {FEATURE_NAMES}")


# ---------------------------------------------------------------------------
# 4. Prediction helpers
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Ensure URL has a scheme so urlparse works correctly."""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    return url


def check_blacklist(url: str) -> bool:
    """Return True if the URL's domain matches a known-bad entry."""
    domain = urlparse(url).netloc.lower()
    try:
        with open("blacklist.json") as f:
            blacklist = json.load(f)
    except FileNotFoundError:
        blacklist = []
    return any(bad in domain for bad in blacklist)


def classify(pct: float) -> str:
    """Map a risk percentage to a verdict label."""
    if pct < 35:
        return "SAFE"
    if pct < 70:
        return "SUSPICIOUS"
    return "PHISHING"


def build_reasons(feats: dict) -> list[str]:
    """Generate shared human-readable detection insights from extracted features."""
    reasons = []
    if feats["url_length"] > 50:
        reasons.append(f"Abnormally long URL ({feats['url_length']} characters) — often used to hide the real destination.")
    if feats["qty_hyphen"] >= 2:
        reasons.append("Multiple hyphens in the domain — a common trick to mimic legitimate brand names (e.g. paypal-secure-login.com).")
    if feats["is_ip"]:
        reasons.append("An IP address is used instead of a domain name — legitimate sites almost never do this.")
    if feats["has_at"]:
        reasons.append("The '@' symbol was detected — browsers ignore everything before '@', redirecting you to a different site.")
    if feats["qty_dot"] > 4:
        reasons.append(f"Excessive subdomains ({feats['qty_dot']} dots) — attackers add layers to make a URL look official.")
    if not reasons:
        reasons.append("No structural anomalies detected. URL structure appears normal.")
    return reasons


def build_lr_reasons(feats: dict, prob: float) -> list[str]:
    """
    Logistic Regression-specific reasons.
    LR is a linear model — it weights each feature independently.
    Explain which features pushed the linear score high or low.
    """
    reasons = []

    # High risk contributors
    if prob >= 70:
        if feats["qty_hyphen"] >= 2:
            reasons.append(
                f"LR heavily weighted the {feats['qty_hyphen']} hyphens — "
                "this single feature has a strong positive coefficient for phishing in a linear model."
            )
        if feats["url_length"] > 50:
            reasons.append(
                f"URL length ({feats['url_length']} chars) pushed the LR score up — "
                "the model learned that longer URLs correlate with obfuscation attempts."
            )
        if feats["has_at"]:
            reasons.append(
                "The '@' symbol carries a very high LR coefficient — "
                "it is almost exclusively seen in malicious URLs."
            )
        if feats["is_ip"]:
            reasons.append(
                "Raw IP address in the URL has a strong positive weight in the LR model — "
                "no legitimate brand uses a bare IP as its address."
            )
        if feats["qty_dot"] > 4:
            reasons.append(
                f"High dot count ({feats['qty_dot']}) raised the LR probability — "
                "deep subdomain nesting is a known phishing pattern."
            )
        if not reasons:
            reasons.append(
                f"The combined linear feature weights produced a high phishing probability ({prob}%). "
                "No single dominant feature, but the overall URL profile matches phishing patterns."
            )

    # Medium risk
    elif prob >= 35:
        if feats["qty_hyphen"] == 1:
            reasons.append(
                "One hyphen in the domain slightly raised the LR score — "
                "not conclusive alone, but combined with other features it adds suspicion."
            )
        if feats["url_length"] > 30:
            reasons.append(
                f"Moderately long URL ({feats['url_length']} chars) contributed a partial positive weight in LR."
            )
        if feats["qty_dot"] >= 3:
            reasons.append(
                f"Three or more dots ({feats['qty_dot']}) added a mild positive weight — "
                "LR sees this as a weak phishing signal."
            )
        if not reasons:
            reasons.append(
                f"LR produced a borderline score ({prob}%) — "
                "no strongly weighted features, but the URL profile is not entirely clean."
            )

    # Low risk
    else:
        reasons.append(
            f"LR score is low ({prob}%) — the URL's features (short length, no hyphens, "
            "no IP, no '@') align closely with the safe URLs in training data."
        )
        if feats["domain_length"] <= 15:
            reasons.append(
                f"Short domain length ({feats['domain_length']} chars) is a negative indicator for phishing "
                "in the LR model — legitimate brands tend to have concise domains."
            )

    return reasons


def build_rf_reasons(feats: dict, prob: float) -> list[str]:
    """
    Random Forest-specific reasons.
    RF uses an ensemble of decision trees — it captures non-linear interactions.
    Explain the tree-based decision logic.
    """
    reasons = []

    # High risk
    if prob >= 70:
        if feats["qty_hyphen"] >= 2 and feats["url_length"] > 40:
            reasons.append(
                f"The RF trees found the combination of {feats['qty_hyphen']} hyphens AND "
                f"long URL ({feats['url_length']} chars) together — this interaction is a strong "
                "phishing pattern the ensemble consistently splits on."
            )
        elif feats["qty_hyphen"] >= 2:
            reasons.append(
                f"Multiple hyphens ({feats['qty_hyphen']}) triggered a majority of RF decision trees "
                "to classify this as phishing — the ensemble reached high agreement."
            )
        if feats["is_ip"]:
            reasons.append(
                "Every RF tree that encountered an IP-address flag voted PHISHING — "
                "the ensemble is unanimous on this feature."
            )
        if feats["has_at"]:
            reasons.append(
                "The '@' flag caused near-unanimous agreement across RF trees — "
                "this pattern does not appear in any safe URL in the training set."
            )
        if feats["qty_dot"] > 4:
            reasons.append(
                f"Deep subdomain nesting ({feats['qty_dot']} dots) caused most RF trees to "
                "branch toward the phishing leaf node."
            )
        if not reasons:
            reasons.append(
                f"The RF ensemble voted {prob}% confidence for PHISHING — "
                "multiple trees independently identified this URL's feature combination as suspicious."
            )

    # Medium risk
    elif prob >= 35:
        reasons.append(
            f"The RF ensemble was split ({prob}% phishing confidence) — "
            "some trees voted safe, others voted phishing, indicating ambiguous features."
        )
        if feats["qty_hyphen"] == 1:
            reasons.append(
                "One hyphen caused a subset of RF trees to branch toward phishing — "
                "not enough for consensus but enough to raise the aggregate score."
            )
        if feats["url_length"] > 30:
            reasons.append(
                f"URL length of {feats['url_length']} chars pushed some trees toward the phishing branch, "
                "but others considered it within normal range."
            )
        if not reasons:
            reasons.append(
                "The RF trees found mixed signals — the URL has some characteristics of both "
                "safe and phishing URLs in the training data."
            )

    # Low risk
    else:
        reasons.append(
            f"The RF ensemble voted {100 - prob}% confidence for SAFE — "
            "the majority of decision trees reached the safe leaf node."
        )
        if feats["domain_length"] <= 15 and feats["qty_hyphen"] == 0:
            reasons.append(
                "Short domain with no hyphens — this combination consistently routes "
                "to safe leaf nodes across the RF ensemble."
            )
        if feats["qty_dot"] <= 2:
            reasons.append(
                f"Low dot count ({feats['qty_dot']}) kept RF trees on the safe branch — "
                "minimal subdomain nesting is a hallmark of legitimate URLs."
            )

    return reasons


# ---------------------------------------------------------------------------
# 5. Main prediction function
# ---------------------------------------------------------------------------
def predict_phishing(url: str) -> dict:
    """
    Full pipeline. Returns:
      status        — overall verdict (SAFE / SUSPICIOUS / PHISHING)
      risk_score    — combined (averaged) risk %
      lr_prob       — Logistic Regression risk %
      lr_status     — LR individual verdict
      rf_prob       — Random Forest risk %
      rf_status     — RF individual verdict
      reasons       — list of human-readable insights
    """
    normalized = normalize_url(url)

    # --- Blacklist fast-path ---
    if check_blacklist(normalized):
        return {
            "status":     "PHISHING",
            "risk_score": 100.0,
            "lr_prob":    None,
            "lr_status":  None,
            "rf_prob":    None,
            "rf_status":  None,
            "reasons": [
                "Domain matched a known phishing entry in the system blacklist.",
                "This URL has been reported as malicious. Do not visit it.",
            ],
        }

    # --- Feature extraction ---
    feats   = extract_features(normalized)
    feat_df = pd.DataFrame([feats])

    # --- Individual model probabilities (phishing class = 1) ---
    lr_prob = round(float(lr_model.predict_proba(feat_df)[0][1]) * 100, 1)
    rf_prob = round(float(rf_model.predict_proba(feat_df)[0][1]) * 100, 1)

    # --- Individual verdicts ---
    lr_status = classify(lr_prob)
    rf_status = classify(rf_prob)

    # --- Combined (averaged) score & overall verdict ---
    combined  = round((lr_prob + rf_prob) / 2, 1)
    status    = classify(combined)

    return {
        "status":      status,
        "risk_score":  combined,
        "lr_prob":     lr_prob,
        "lr_status":   lr_status,
        "lr_reasons":  build_lr_reasons(feats, lr_prob),
        "rf_prob":     rf_prob,
        "rf_status":   rf_status,
        "rf_reasons":  build_rf_reasons(feats, rf_prob),
        "reasons":     build_reasons(feats),
    }


def predict_phishing_batch(urls: list[str]) -> list[dict]:
    results = [None] * len(urls)
    pending = []

    for index, url in enumerate(urls):
        normalized = normalize_url(url)
        if check_blacklist(normalized):
            results[index] = predict_phishing(url)
        else:
            pending.append((index, extract_features(normalized)))

    if pending:
        feat_df = pd.DataFrame([feats for _, feats in pending])
        lr_probs = lr_model.predict_proba(feat_df)[:, 1]
        rf_probs = rf_model.predict_proba(feat_df)[:, 1]

        for (index, feats), lr_raw, rf_raw in zip(pending, lr_probs, rf_probs):
            lr_prob = round(float(lr_raw) * 100, 1)
            rf_prob = round(float(rf_raw) * 100, 1)
            lr_status = classify(lr_prob)
            rf_status = classify(rf_prob)
            combined = round((lr_prob + rf_prob) / 2, 1)
            results[index] = {
                "status": classify(combined),
                "risk_score": combined,
                "lr_prob": lr_prob,
                "lr_status": lr_status,
                "lr_reasons": build_lr_reasons(feats, lr_prob),
                "rf_prob": rf_prob,
                "rf_status": rf_status,
                "rf_reasons": build_rf_reasons(feats, rf_prob),
                "reasons": build_reasons(feats),
            }

    return results


# Quick self-test on startup
_tests = [
    ("https://www.google.com",              "SAFE"),
    ("http://paypal-security-update.com",   "PHISHING"),   # blacklist hit
    ("http://login-secure-paypal-verification.com/webscr", "PHISHING"),
]
print("\n  Self-test results:")
for _url, _expected in _tests:
    _r = predict_phishing(_url)
    _ok = "OK" if _r["status"] == _expected else "FAIL"
    print(f"  {_ok} {_url[:55]:<55}  ->  {_r['status']}  (combined {_r['risk_score']}%)")
# ---------------------------------------------------------------------------
# 6. Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
MAX_BATCH_URLS = 1000


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/app-assets/<path:filename>")
def serve_app_asset(filename):
    if filename == "app.js":
        return send_from_directory(os.path.dirname(__file__), filename)
    if filename not in {"analyzer.css", "style.css"}:
        return "Not found", 404
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), filename)


@app.route("/pdf")
def serve_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), "Phishing_URL_Detector_Code.pdf")
    return send_file(pdf_path, mimetype="application/pdf",
                     download_name="Phishing_URL_Detector_Code.pdf")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object containing 'url' or 'urls'."}), 400

    if "urls" in body:
        submitted_urls = body["urls"]
        if isinstance(submitted_urls, str):
            submitted_urls = submitted_urls.splitlines()
        if not isinstance(submitted_urls, list) or any(not isinstance(url, str) for url in submitted_urls):
            return jsonify({"error": "'urls' must be a list of URL strings or newline-separated text."}), 400
        urls = [url.strip() for url in submitted_urls if url.strip()]
        if not urls:
            return jsonify({"error": "Enter at least one URL."}), 400
        if len(urls) > MAX_BATCH_URLS:
            return jsonify({"error": f"Analyze up to {MAX_BATCH_URLS} URLs per request."}), 400
        return jsonify({
            "count": len(urls),
            "results": [
                {"url": url, **result}
                for url, result in zip(urls, predict_phishing_batch(urls))
            ],
        })

    url = body.get("url")
    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "Missing or empty 'url' field."}), 400
    return jsonify(predict_phishing(url))


@app.route("/metrics")
def metrics():
    return jsonify({
        "models": MODEL_METRICS,
        "sample_count": len(TRAINING_DATA),
        "evaluation": "5-fold stratified cross-validation",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server at http://127.0.0.1:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
