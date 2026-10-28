"""Conventional entry point for local runs and WSGI hosting."""

from phishing_detector import *


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server at http://127.0.0.1:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)