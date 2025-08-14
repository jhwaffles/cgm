import numpy as np
import pandas as pd

def calculate_gmi(mean_bg):
    """Estimate A1C from mean BG."""
    return round(3.31 + 0.02392 * mean_bg, 2)

def time_in_zone(glucose_series):
    #Calculates percentage of data points spent in defined glucose zones. This is a PROXY for percentage time spent in zones.
    total = glucose_series.dropna().shape[0]
    if total == 0:
        return {
            ">250": 0.0,
            "181–250": 0.0,
            "70–180": 0.0,
            "54–69": 0.0,
            "<54": 0.0
        }

    zones = {
        ">250": (glucose_series > 250).sum(),
        "181–250": ((glucose_series > 180) & (glucose_series <= 250)).sum(),
        "70–180": ((glucose_series >= 70) & (glucose_series <= 180)).sum(),
        "54–69": ((glucose_series >= 54) & (glucose_series < 70)).sum(),
        "<54": (glucose_series < 54).sum()
    }

    # Convert counts to percentages
    zone_percentages = {k: round(v / total * 100, 1) for k, v in zones.items()}
    return zone_percentages

def compute_risk_trace(df, glucose_col="glucose", time_col="timestamp", freq='1h'):
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"Expected column '{time_col}' not in DataFrame. Available columns: {df.columns.tolist()}")
    df = df.dropna(subset=[glucose_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col])

    def compute_risk(bg_values):
        rl_list = []
        rh_list = []
        for bg in bg_values:
            if bg <= 0:
                continue
            f = 1.509 * ((np.log(bg)) ** 1.084 - 5.381)
            r = 10 ** (f ** 2)
            if f < 0:
                rl_list.append(r)
            else:
                rh_list.append(r)
        return pd.Series({
            "LBGI": np.mean(rl_list) if rl_list else 0,
            "HBGI": np.mean(rh_list) if rh_list else 0
        })

    grouped = (
        df
        .set_index(time_col)
        .resample(freq)
        .apply(lambda group: pd.Series(compute_risk(group[glucose_col])))
        .reset_index()
    )
    print("[DEBUG] df_risk (grouped) head:")
    print(grouped.head())
    print("[DEBUG] df_risk columns:", grouped.columns.tolist())
    return grouped  

def compute_rate_of_change(glucose_series, time_series, interval_minutes=15):
    df = pd.DataFrame({"glucose": glucose_series, "timestamp": pd.to_datetime(time_series)})
    df = df.sort_values("timestamp").dropna()

    # Resample to get 15-minute spaced glucose values
    df.set_index("timestamp", inplace=True)
    df_resampled = df.resample(f"{interval_minutes}min").mean().dropna().reset_index()

    # Compute rate of change across 15-min intervals
    df_resampled["delta_glucose"] = df_resampled["glucose"].diff()
    df_resampled["delta_time"] = df_resampled["timestamp"].diff().dt.total_seconds() / 60
    df_resampled["rate_of_change"] = df_resampled["delta_glucose"] / df_resampled["delta_time"]

    return df_resampled["rate_of_change"].dropna()

def compute_poincare_data(glucose_series, time_series, lag=1, interval='15min'):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(time_series),
        "glucose": glucose_series
    }).dropna()

    df = df.set_index("timestamp").resample(interval).mean().dropna().reset_index()

    g1 = df["glucose"][:-lag].values
    g2 = df["glucose"][lag:].values

    return pd.DataFrame({
        f"BG_t": g1,
        f"BG_t+{lag}": g2
    })