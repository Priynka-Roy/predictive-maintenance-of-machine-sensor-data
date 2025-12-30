from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# Config (kept consistent)
# ----------------------------
HORIZON_HOURS = 24
ROLL_HOURS = 24
MAX_MACHINES = None # set None later

# ----------------------------
# Paths (kept consistent)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Predictive Maintenance/
DATA_DIR = PROJECT_ROOT / "data"
OUT_FILE = DATA_DIR / f"features_h{HORIZON_HOURS}.csv"


# ----------------------------
# IO
# ----------------------------
def read_csv(stem: str) -> pd.DataFrame:
    path = DATA_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


# ----------------------------
# Feature helpers
# ----------------------------
def next_event_hours(base_times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """
    For each base time, find the next event time strictly after it and return the
    difference in hours. If no future event exists, returns np.inf.
    base_times, event_times must be sorted ascending datetime64[ns].
    """
    # searchsorted gives insertion index to keep order; for "next strictly after",
    # use side="right" so exact ties are considered not-future.
    idx = np.searchsorted(event_times, base_times, side="right")
    out = np.full(base_times.shape[0], np.inf, dtype="float64")
    valid = idx < event_times.shape[0]
    # timedelta64 -> hours
    out[valid] = (event_times[idx[valid]] - base_times[valid]) / np.timedelta64(1, "h")
    return out


def count_events_last_hours(base_times: np.ndarray, event_times: np.ndarray, hours: int) -> np.ndarray:
    """
    For each base time t, count events where (t - hours, t] (strictly greater than t-hours, <= t),
    matching the Spark logic:
      event_time <= datetime AND event_time > datetime - INTERVAL hours HOURS
    """
    if event_times.size == 0:
        return np.zeros(base_times.shape[0], dtype="int64")

    right = np.searchsorted(event_times, base_times, side="right")  # events <= t
    left_bound = base_times - np.timedelta64(hours, "h")
    left = np.searchsorted(event_times, left_bound, side="right")   # events <= (t-hours)
    return (right - left).astype("int64")


def build_features_for_machine(
    machine_id: int,
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    sensor_cols = ["volt", "rotate", "pressure", "vibration"]

    tel_m = telemetry.loc[telemetry["machineID"] == machine_id, ["machineID", "datetime", *sensor_cols]].copy()
    tel_m = tel_m.sort_values("datetime").drop_duplicates(["machineID", "datetime"])

    fail_m = failures.loc[failures["machineID"] == machine_id, ["datetime"]].drop_duplicates().sort_values("datetime")
    err_m = errors.loc[errors["machineID"] == machine_id, ["datetime"]].drop_duplicates().sort_values("datetime")
    mnt_m = maint.loc[maint["machineID"] == machine_id, ["datetime"]].drop_duplicates().sort_values("datetime")

    mac_m = machines.loc[machines["machineID"] == machine_id].copy()

    # If no telemetry, nothing to do
    if tel_m.empty:
        return tel_m

    base_times = tel_m["datetime"].to_numpy(dtype="datetime64[ns]")

    # ----------------------------
    # Label: failure within horizon (same intent as Spark)
    # ----------------------------
    failure_times = fail_m["datetime"].to_numpy(dtype="datetime64[ns]") if not fail_m.empty else np.array([], dtype="datetime64[ns]")
    dt_hours_to_next_failure = next_event_hours(base_times, failure_times)
    tel_m["label"] = (dt_hours_to_next_failure <= float(HORIZON_HOURS)).astype("int64")

    # ----------------------------
    # Rolling sensor features (time-based, per machine)
    # ----------------------------
    # Use time-based rolling window of ROLL_HOURS hours, inclusive of current timestamp.
    # This matches the intended "last 24 hours" behavior for rolling stats.
    tel_m = tel_m.set_index("datetime")

    roll_window = f"{ROLL_HOURS}H"
    for c in sensor_cols:
        r = tel_m[c].rolling(roll_window, min_periods=1)
        tel_m[f"{c}_mean_{ROLL_HOURS}h"] = r.mean()
        tel_m[f"{c}_std_{ROLL_HOURS}h"] = r.std(ddof=0)  # population stddev to mirror stddev_pop
        tel_m[f"{c}_min_{ROLL_HOURS}h"] = r.min()
        tel_m[f"{c}_max_{ROLL_HOURS}h"] = r.max()
        tel_m[f"{c}_delta_1"] = tel_m[c].diff(1)

    # ----------------------------
    # Event counts last 24h (errors, maint) - same window logic as Spark
    # ----------------------------
    err_times = err_m["datetime"].to_numpy(dtype="datetime64[ns]") if not err_m.empty else np.array([], dtype="datetime64[ns]")
    mnt_times = mnt_m["datetime"].to_numpy(dtype="datetime64[ns]") if not mnt_m.empty else np.array([], dtype="datetime64[ns]")

    tel_m["errors_last_24h"] = count_events_last_hours(base_times, err_times, 24)
    tel_m["maint_last_24h"] = count_events_last_hours(base_times, mnt_times, 24)

    # Back to columns
    tel_m = tel_m.reset_index()  # brings datetime back

    # ----------------------------
    # Metadata join (same intent)
    # ----------------------------
    if not mac_m.empty:
        tel_m = tel_m.merge(mac_m, on="machineID", how="left")

    # ----------------------------
    # Drop early nulls (same intent)
    # ----------------------------
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

    # add metadata columns (everything from mac_m except machineID)
    if not mac_m.empty:
        feature_cols += [col for col in mac_m.columns if col != "machineID"]

    tel_m = tel_m.dropna(subset=feature_cols + ["label"])

    return tel_m


def main() -> None:
    # Load data
    telemetry = read_csv("PdM_telemetry")
    failures = read_csv("PdM_failures")
    errors = read_csv("PdM_errors")
    maint = read_csv("PdM_maint")
    machines = read_csv("PdM_machines")

    # Types (mirror Spark casting)
    for df in (telemetry, failures, errors, maint):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["machineID"] = df["machineID"].astype("int64")

    machines["machineID"] = machines["machineID"].astype("int64")

    # Basic sanity: drop rows with invalid timestamps
    telemetry = telemetry.dropna(subset=["datetime"])
    failures = failures.dropna(subset=["datetime"])
    errors = errors.dropna(subset=["datetime"])
    maint = maint.dropna(subset=["datetime"])

    # machine ids
    machine_ids = sorted(telemetry["machineID"].unique().tolist())
    if MAX_MACHINES is not None:
        machine_ids = machine_ids[:MAX_MACHINES]

    # Reset output
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    written = 0
    for mid in machine_ids:
        df_mid = build_features_for_machine(mid, telemetry, failures, errors, maint, machines)

        # Append to CSV (same “per machine” behavior)
        df_mid.to_csv(OUT_FILE, mode="a", header=(written == 0), index=False)

        written += 1
        print(f"Written machineID={mid} ({written}/{len(machine_ids)})")

    print("=" * 72)
    print("FEATURE ENGINEERING COMPLETED (pandas)")
    print(f"Output CSV: {OUT_FILE}")
    print(f"Machines processed: {written}")
    print("=" * 72)


if __name__ == "__main__":
    main()