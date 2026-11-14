"""Download the IPL ball-by-ball JSON archive from Cricsheet (open data, ODC-By)."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://cricsheet.org/downloads/ipl_json.zip"
DEST = Path("data/raw")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (death-overs-project)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        for m in members:
            (DEST / Path(m).name).write_bytes(zf.read(m))
    print(f"Extracted {len(members)} match files to {DEST.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed: {exc}\nGet it manually from https://cricsheet.org/downloads/ "
                 f"and unzip the JSON files into {DEST}/")
