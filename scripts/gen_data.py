"""
scripts/gen_data.py
-------------------
Attempts to download the AI4I 2020 Predictive Maintenance dataset from UCI / Kaggle.
Falls back to a reproducible synthetic dataset with the same schema.
Outputs: data/ai4i2020.csv
"""
import pathlib, sys, io, zipfile, urllib.request, urllib.error
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT = DATA_DIR / "ai4i2020.csv"

SEED = 42
N = 10_000

COLS = [
    "UDI","Product ID","Type",
    "Air temperature [K]","Process temperature [K]",
    "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
    "Machine failure","TWF","HDF","PWF","OSF","RNF"
]

def synthetic_dataset(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_L = int(n * 0.60); n_M = int(n * 0.30); n_H = n - n_L - n_M
    types = ["L"] * n_L + ["M"] * n_M + ["H"] * n_H
    rng.shuffle(types)

    air_temp   = rng.normal(300.0, 2.0, n)
    proc_temp  = air_temp + rng.normal(10.0, 1.0, n)
    rot_speed  = rng.normal(1538, 179, n).clip(1000, 2000)
    torque     = rng.normal(40, 10, n).clip(3, 80)
    tool_wear  = rng.integers(0, 253, n).astype(float)

    # failure modes
    twf = ((tool_wear > 200) & (rng.random(n) < 0.15)).astype(int)
    hdf = (((proc_temp - air_temp) < 8.6) & (rot_speed < 1380) & (rng.random(n) < 0.20)).astype(int)
    pwf = ((torque * rot_speed < 3500) | (torque * rot_speed > 9000)) .astype(int) * (rng.random(n) < 0.10).astype(int)
    osf_base = np.where(np.array(types) == "L", 11000, np.where(np.array(types) == "M", 12000, 13000))
    osf = ((tool_wear * torque) > osf_base).astype(int)
    rnf = (rng.random(n) < 0.001).astype(int)
    failure = ((twf | hdf | pwf | osf | rnf) > 0).astype(int)

    df = pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"{t}{rng.integers(10000,99999)}" for t in types],
        "Type": types,
        "Air temperature [K]": air_temp.round(1),
        "Process temperature [K]": proc_temp.round(1),
        "Rotational speed [rpm]": rot_speed.round(0).astype(int),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": tool_wear.astype(int),
        "Machine failure": failure,
        "TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf,
    })
    return df


def try_uci_download() -> bool:
    url = (
        "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                return False
            with z.open(csv_names[0]) as f:
                df = pd.read_csv(f)
        df.to_csv(OUT, index=False)
        print(f"[gen_data] Downloaded from UCI -> {OUT}")
        return True
    except Exception as e:
        print(f"[gen_data] UCI download failed ({e}), using synthetic data.")
        return False


def main():
    if OUT.exists():
        print(f"[gen_data] {OUT} already exists, skipping generation.")
        return
    success = try_uci_download()
    if not success:
        df = synthetic_dataset()
        df.to_csv(OUT, index=False)
        print(f"[gen_data] Synthetic dataset ({len(df)} rows) -> {OUT}")


if __name__ == "__main__":
    main()
