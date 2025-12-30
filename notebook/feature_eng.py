from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

"""
feature_eng.py (pandas, laptop-friendly)
----------------------------------------
Build supervised dataset for "failure within next H hours" and rolling features.

Strategy:
- Process each machineID separately to stay memory-safe on local Windows.
- Write incrementally (append-like) to a folder of CSV parts.

Output:
  data/features_h{H}.csvdir/   (a folder containing CSV parts)
"""

# ----------------------------
# Config
# ----------------------------
HORIZON_HOURS = 24
ROLL_HOURS = 24
MAX_MACHINES = None  # set None for all

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Predictive Maintenance/
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = DATA_DIR / f"features_h{HORIZON_HOURS}.csvdir"


# ----------------------------
# IO
# ----------------------------
def read_csv(stem: str) -> pd.DataFrame:
    path = DATA_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


# ----------------------------
# Event helpers (match Spark intent)
# ----------------------------
def next_event_hours(base_times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """
    For each base time, find the next event time strictly after it and return hours to it.
    If no future event exists, returns np.inf.
    Arrays must be sorted ascending datetime64[ns].
    """
    idx = np.searchsorted(event_times, base_times, side="right")  # strictly after
    out = np.full(base_times.shape[0], np.inf, dtype="float64")
    valid = idx < event_times.shape[0]
    out[valid] = (event_times[idx[valid]] - base_times[valid]) / np.timedelta64(1, "h")
    return out


def count_events_last_hours(base_times: np.ndarray, event_times: np.ndarray, hours: int = 24) -> np.ndarray:
    """
    Count events in (t-hours, t] for each base time t.
    Mirrors Spark:
      event_time <= datetime AND event_time > datetime - INTERVAL hours HOURS
    """
    if event_times.size == 0:
        return np.zeros(base_times.shape[0], dtype="int64")

    right = np.searchsorted(event_times, base_times, side="right")  # events <= t
    left_bound = base_times - np.timedelta64(hours, "h")
    left = np.searchsorted(event_times, left_bound, side="right")   # events <= t-hours
    return (right - left).astype("int64")


# ----------------------------
# Core per-machine feature build
# ----------------------------
def build_features_for_machine(
    machine_id: int,
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    sensor_cols = ["volt", "rotate", "pressure", "vibration"]

    tel_m = telemetry.loc[
        telemetry["machineID"] == machine_id,
        ["machineID", "datetime", *sensor_cols],
    ].copy()

    if tel_m.empty:
        return tel_m

    tel_m = tel_m.sort_values("datetime").drop_duplicates(["machineID", "datetime"])

    fail_m = failures.loc[failures["machineID"] == machine_id, ["datetime"]].drop_duplicates().sort_values("datetime")
    err_m = errors.loc[errors["machineID"] == machine_id, ["datetime"]].drop_duplicates().sort_values("datetime")
    mnt_m = maint.loc[maint["machineID"] == machine_id, ["datetime"]].drop_duplicates().sort_values("datetime")

    mac_m = machines.loc[machines["machineID"] == machine_id].copy()

    base_times = tel_m["datetime"].to_numpy(dtype="datetime64[ns]")

    # ---- Label (failure within horizon) ----
    failure_times = (
        fail_m["datetime"].to_numpy(dtype="datetime64[ns]")
        if not fail_m.empty else np.array([], dtype="datetime64[ns]")
    )
    dt_hours_to_next_failure = next_event_hours(base_times, failure_times)
    tel_m["label"] = (dt_hours_to_next_failure <= float(HORIZON_HOURS)).astype("int64")

    # ---- Rolling features ----
    # Spark used rowsBetween(-ROLL_HOURS, 0). In the Azure PdM dataset telemetry is hourly,
    # so rows and hours align. Here we implement time-based rolling (more robust) using "ROLL_HOURS H".
    tel_m = tel_m.set_index("datetime")

    roll_window = f"{ROLL_HOURS}H"
    for c in sensor_cols:
        r = tel_m[c].rolling(roll_window, min_periods=1)
        tel_m[f"{c}_mean_{ROLL_HOURS}h"] = r.mean()
        tel_m[f"{c}_std_{ROLL_HOURS}h"] = r.std(ddof=0)  # stddev_pop equivalent
        tel_m[f"{c}_min_{ROLL_HOURS}h"] = r.min()
        tel_m[f"{c}_max_{ROLL_HOURS}h"] = r.max()
        tel_m[f"{c}_delta_1"] = tel_m[c].diff(1)

    tel_m = tel_m.reset_index()  # bring datetime back as a column

    # ---- Event counts (last 24h) ----
    err_times = (
        err_m["datetime"].to_numpy(dtype="datetime64[ns]")
        if not err_m.empty else np.array([], dtype="datetime64[ns]")
    )
    mnt_times = (
        mnt_m["datetime"].to_numpy(dtype="datetime64[ns]")
        if not mnt_m.empty else np.array([], dtype="datetime64[ns]")
    )

    tel_m["errors_last_24h"] = count_events_last_hours(base_times, err_times, 24)
    tel_m["maint_last_24h"] = count_events_last_hours(base_times, mnt_times, 24)

    # ---- Join metadata ----
    if not mac_m.empty:
        tel_m = tel_m.merge(mac_m, on="machineID", how="left")

    # ---- Drop early nulls (match Spark feat.na.drop) ----
    feature_cols = []
    for c in sensor_cols:
        feature_cols += [
            c,
            f"{c}_mean_{ROLL_HOURS}h",
            f"{c}_std_{ROLL_HOURS}h",
            f"{c}_min_{ROLL_HOURS}h",
            f"{c}_max_{ROLL_HOURS}h",
            f"{c}_delta_1",
        ]
    feature_cols += ["errors_last_24h", "maint_last_24h"]

    if not mac_m.empty:
        feature_cols += [col for col in mac_m.columns if col != "machineID"]

    tel_m = tel_m.dropna(subset=feature_cols + ["label"])

    return tel_m


def main() -> None:
    # Load once
    telemetry = read_csv("PdM_telemetry")
    failures = read_csv("PdM_failures")
    errors = read_csv("PdM_errors")
    maint = read_csv("PdM_maint")
    machines = read_csv("PdM_machines")

    # Types (match Spark casting intent)
    for df in (telemetry, failures, errors, maint):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["machineID"] = df["machineID"].astype("int64")
    machines["machineID"] = machines["machineID"].astype("int64")

    # Drop invalid timestamps
    telemetry = telemetry.dropna(subset=["datetime"])
    failures = failures.dropna(subset=["datetime"])
    errors = errors.dropna(subset=["datetime"])
    maint = maint.dropna(subset=["datetime"])

    # Machine list
    machine_ids = sorted(telemetry["machineID"].unique().tolist())
    if MAX_MACHINES is not None:
        machine_ids = machine_ids[:MAX_MACHINES]

    # Reset output folder (Spark-like)
    if OUT_PATH.exists():
        shutil.rmtree(OUT_PATH, ignore_errors=True)
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    written = 0
    for mid in machine_ids:
        df_mid = build_features_for_machine(mid, telemetry, failures, errors, maint, machines)

        part_name = f"part-{written:05d}-machineID-{mid}.csv"
        out_file = OUT_PATH / part_name

        df_mid.to_csv(out_file, index=False)
        written += 1

        print(f"Written machineID={mid} ({written}/{len(machine_ids)})")

    print("=" * 72)
    print("FEATURE ENGINEERING COMPLETED (per-machine mode, pandas)")
    print(f"Output: {OUT_PATH}")
    print(f"Machines processed: {written}")
    print(f"Horizon: {HORIZON_HOURS}h | Rolling: {ROLL_HOURS}h")
    print("=" * 72)


if __name__ == "__main__":
    main()