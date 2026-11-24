"""Serve a local expert review gallery with byte-range seeking for the FPV videos."""
import argparse
from pathlib import Path

from flask import Flask, redirect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--port", type=int, default=7912)
    args = parser.parse_args()
    root = args.dataset.resolve(strict=True)
    if not (root / "review.html").is_file():
        parser.error(f"Missing review.html in {root}")
    app = Flask(__name__, static_folder=str(root), static_url_path="")

    @app.get("/")
    def index():
        return redirect("/review.html")

    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
