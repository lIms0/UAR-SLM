import argparse
import time
import urllib3
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def build_headers(url: str):
    p = urlparse(url)
    domain = f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else "https://www.google.com"
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": domain,
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

def save_image(content: bytes, out_path: Path):
    img = Image.open(BytesIO(content))
    if img.mode != "RGB":
        img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--failed-status", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_csv(args.report)
    targets = df[df["status"] == args.failed_status].copy()

    print("targets =", len(targets))

    logs = []

    for _, row in tqdm(targets.iterrows(), total=len(targets)):
        url = str(row["url"])
        out_path = Path(row["path"])

        if out_path.exists():
            logs.append({
                "id": row.get("id"),
                "kind": row.get("kind"),
                "url": url,
                "path": str(out_path),
                "status": "already_exists"
            })
            continue

        ok = False
        last_err = None

        for attempt in range(args.retries):
            try:
                with requests.Session() as s:
                    s.headers.update(build_headers(url))
                    r = s.get(url, timeout=args.timeout, verify=False, allow_redirects=True, stream=True)
                    r.raise_for_status()
                    save_image(r.content, out_path)

                logs.append({
                    "id": row.get("id"),
                    "kind": row.get("kind"),
                    "url": url,
                    "path": str(out_path),
                    "status": f"downloaded_attempt_{attempt+1}"
                })
                ok = True
                break
            except Exception as e:
                last_err = repr(e)
                time.sleep(3 * (attempt + 1))

        if not ok:
            logs.append({
                "id": row.get("id"),
                "kind": row.get("kind"),
                "url": url,
                "path": str(out_path),
                "status": f"still_failed_from_{args.failed_status}",
                "error": last_err
            })

    out = pd.DataFrame(logs)
    out.to_csv(args.output, index=False)

    print()
    print("saved:", args.output)
    if not out.empty:
        print(out["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()