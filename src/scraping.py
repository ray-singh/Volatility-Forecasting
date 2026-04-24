"""
Downloads the most recent ZIP file from:
https://quote-saver.bycsi.com/orderbook/inverse/BTCUSD/

Then:
1. Finds the latest .zip link on the page
2. Downloads it
3. Extracts it
4. Finds the CSV inside
5. Prints the CSV path

Requirements:
pip install requests beautifulsoup4
"""

import os
import zipfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://quote-saver.bycsi.com/orderbook/inverse/BTCUSD/"
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "extracted"


def get_latest_zip_url():
    response = requests.get(BASE_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    zip_links = []

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".zip"):
            full_url = urljoin(BASE_URL, href)
            zip_links.append(full_url)

    if not zip_links:
        raise Exception("No ZIP files found on the page.")

    # Usually directory listings are sorted oldest -> newest,
    # so the most recent is typically the last ZIP file.
    latest_zip = zip_links[-1]
    return latest_zip


def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    filename = url.split("/")[-1]
    filepath = os.path.join(save_dir, filename)

    print(f"Downloading: {url}")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Saved ZIP to: {filepath}")
    return filepath


def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Extracted to: {extract_dir}")


def find_data_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".data"):
                return os.path.join(root, file)
    return None


def main():
    latest_zip_url = get_latest_zip_url()
    print(f"Latest ZIP found: {latest_zip_url}")

    zip_path = download_file(latest_zip_url, DOWNLOAD_DIR)

    extract_zip(zip_path, EXTRACT_DIR)

    data_path = find_data_file(EXTRACT_DIR)

    if data_path:
        print(f"DATA file found: {data_path}")
    else:
        print("No DATA file found after extraction.")


if __name__ == "__main__":
    main()
