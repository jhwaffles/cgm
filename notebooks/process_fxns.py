import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#event functions
def create_event_windows(df_events, window_duration=pd.Timedelta(hours=2)):
    df_events = df_events.sort_values(by='timestamp').reset_index(drop=True)
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])

    event_windows = []

    for i in range(len(df_events)):
        event_time = df_events.loc[i, 'timestamp']
        note = df_events.loc[i].get('Notes', pd.NA)

        window_start = event_time
        # Default end: event_time + 2 hours
        default_end = event_time + window_duration

        # If there's a next event, use that as the cap
        if i < len(df_events) - 1:
            next_event_time = df_events.loc[i + 1, 'timestamp']
            window_end = min(default_end, next_event_time)
        else:
            window_end = default_end

        event_windows.append({
            'event_time': event_time,
            'event_note': note,
            'window_start': window_start,
            'window_end': window_end
        })

    return pd.DataFrame(event_windows)

def rolling_slopes(times, glucose, window_size=3):
    """
    Compute slopes over rolling windows of given size.
    Returns an array of slopes (one per window).
    """
    slopes = []
    for i in range(len(times) - window_size + 1):
        t_win = times[i:i + window_size].reshape(-1, 1)
        g_win = glucose[i:i + window_size]
        if len(np.unique(t_win)) < 2:  # avoid divide-by-zero in flat time
            continue
        model = LinearRegression().fit(t_win, g_win)
        slopes.append(model.coef_[0])
    return slopes

def estimate_baseline_glucose(df_cgm, time_col="timestamp", glucose_col="glucose", percentile=50):
    if df_cgm.empty:
        return np.nan

    df = df_cgm[[time_col, glucose_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    # Step 1: Resample into 1-hour bins and compute the min of each bin
    hourly_mins = df.resample("1h").min().dropna()

    if hourly_mins.empty:
        return np.nan

    # Step 2: Take the bottom N% of hourly mins and average
    sorted_vals = hourly_mins[glucose_col].sort_values()
    cutoff = int(len(sorted_vals) * (percentile / 100))
    if cutoff == 0:
        return sorted_vals.mean()
    return sorted_vals.iloc[:cutoff].mean()

def compute_window_metrics(df_cgm, start, end, baseline):
    """
    Given a CGM dataframe and a time window, return glucose metrics.
    Assumes df_cgm['timestamp'] and df_cgm['glucose'] exist.
    """
    # Filter CGM data in the window
    mask = (df_cgm['timestamp'] >= start) & (df_cgm['timestamp'] <= end)
    window_data = df_cgm.loc[mask].copy()

    if window_data.empty:
        return {
            'glucose_max': np.nan,
            'glucose_min': np.nan,
            'glucose_auc': np.nan,
            'glucose_rate_rise': np.nan,
            'glucose_rate_fall': np.nan
        }

    # Sort just in case
    window_data = window_data.sort_values(by='timestamp')
    
    # Calculate time deltas in hours for AUC
    times = (window_data['timestamp'] - window_data['timestamp'].iloc[0]).dt.total_seconds() / 60.0/ 60.0
    glucose = window_data['glucose'].values

    # Basic metrics
    g_max = glucose.max()
    g_min = glucose.min()

    adjusted_glucose =np.maximum(glucose - baseline, 0 )
    auc = np.trapezoid(adjusted_glucose, times)

    # Rate of change. finds sliding window slopes for 3 consecutive points. Takes the max.
    times_np = np.array(times)
    glucose_np=np.array(glucose)
    slopes = rolling_slopes(times_np,glucose_np,3)
    rate_rise = np.max(slopes) if len(slopes) > 0 else np.nan
    rate_fall = np.min(slopes) if len(slopes) > 0 else np.nan

    return {
        'glucose_max': g_max,
        'glucose_min': g_min,
        'glucose_auc': auc,
        'glucose_rate_rise': rate_rise,
        'glucose_rate_fall': rate_fall
    }

def get_time_category(dt):
    hour = dt.hour
    if 6 <= hour < 10:
        return 0  # early morning
    elif 10 <= hour < 14:
        return 1  # morning
    elif 14 <= hour < 18:
        return 2  # afternoon
    elif 18 <= hour < 22:
        return 3  # later afternoon
    else:
        return 4 # late night
    
def compute_metrics_for_all_windows(df_event_windows, df_clean, df_events,baseline):
    metrics = []
    df_exercise = df_events[df_events['event_type']=='exercise'].sort_values('timestamp')

    for _, row in df_event_windows.iterrows():
        start = row['window_start']
        end = row['window_end']
        event_time = row['event_time']
        event_info = row.to_dict()

        # Add: exercise_within_3h
        prior_exercise = df_exercise[df_exercise['timestamp'] < event_time]
        if not prior_exercise.empty:
            last_ex = prior_exercise['timestamp'].max()
            exercise_within_3h = int((event_time - last_ex) <= pd.Timedelta(hours=3))
        else:
            exercise_within_3h = 0

        # Add: meal_time_category
        meal_time_category = get_time_category(event_time)

        m = compute_window_metrics(df_clean, start, end, baseline)
        metrics.append({
            **event_info, 
            **m,
            **m,
            'exercise_within_3h': exercise_within_3h,
            'meal_time_category': meal_time_category
            })

    return pd.DataFrame(metrics)

##overall metrics

# --- GMI Calculation ---
def calculate_gmi(mean_bg):
    """Estimate A1C from mean BG."""
    return round(3.31 + 0.02392 * mean_bg, 2)

# --- Time-in-Range Zoning ---
def zone_tir_metrics(time_in_ranges):
    """
    Classify glucose exposure into risk zones.
    time_in_ranges: dict with keys '<54', '54-70', '70-180', '181-250', '>250'
    """
    zones = {}
    tir = time_in_ranges.get('70-180', 0)
    if tir >= 70:
        zones['TIR'] = 'green'
    elif tir >= 50:
        zones['TIR'] = 'yellow'
    else:
        zones['TIR'] = 'red'

    hypo = time_in_ranges.get('<54', 0) + time_in_ranges.get('54-70', 0)
    if hypo <= 4:
        zones['Hypo'] = 'green'
    elif hypo <= 10:
        zones['Hypo'] = 'yellow'
    else:
        zones['Hypo'] = 'red'

    hyper = time_in_ranges.get('181-250', 0) + time_in_ranges.get('>250', 0)
    if hyper <= 25:
        zones['Hyper'] = 'green'
    elif hyper <= 40:
        zones['Hyper'] = 'yellow'
    else:
        zones['Hyper'] = 'red'

    if time_in_ranges.get('<54', 0) <= 1:
        zones['<54'] = 'green'
    elif time_in_ranges.get('<54', 0) <= 2:
        zones['<54'] = 'yellow'
    else:
        zones['<54'] = 'red'

    if time_in_ranges.get('>250', 0) <= 5:
        zones['>250'] = 'green'
    elif time_in_ranges.get('>250', 0) <= 10:
        zones['>250'] = 'yellow'
    else:
        zones['>250'] = 'red'

    return zones

# --- IQR Calculation ---
def compute_iqr(glucose_values):
    q75, q25 = np.percentile(glucose_values, [75 ,25])
    return round(q75 - q25, 2)

# --- Rate of Change SD ---
def compute_sd_of_rate(glucose_values, time_interval_minutes=5):
    deltas = np.diff(glucose_values) / time_interval_minutes
    return round(np.std(deltas), 3)

# --- LBGI / HBGI / BGRI Calculation ---
def compute_f(bg):
    return 1.509 * ((np.log(bg))**1.084 - 5.381)

def compute_risk(bg):
    f_bg = compute_f(bg)
    risk = 10 ** (f_bg ** 2)
    rl = risk if f_bg < 0 else 0
    rh = risk if f_bg > 0 else 0
    return rl, rh

def compute_LBGI_HBGI(glucose_values):
    rl_list = []
    rh_list = []
    for bg in glucose_values:
        if bg > 0:
            rl, rh = compute_risk(bg)
            rl_list.append(rl)
            rh_list.append(rh)
    LBGI = round(np.mean(rl_list), 2)
    HBGI = round(np.mean(rh_list), 2)
    BGRI = round(LBGI + HBGI, 2)
    return LBGI, HBGI, BGRI

# --- Risk Zoning for LBGI/HBGI/BGRI ---
def zone_bg_risk(lbgi, hbgi):
    zones = {}
    if lbgi <= 1.1:
        zones['LBGI'] = 'green'
    elif lbgi <= 2.5:
        zones['LBGI'] = 'yellow'
    else:
        zones['LBGI'] = 'red'

    if hbgi <= 4.5:
        zones['HBGI'] = 'green'
    elif hbgi <= 9:
        zones['HBGI'] = 'yellow'
    else:
        zones['HBGI'] = 'red'

    bgri = lbgi + hbgi
    if bgri <= 6:
        zones['BGRI'] = 'green'
    elif bgri <= 10:
        zones['BGRI'] = 'yellow'
    else:
        zones['BGRI'] = 'red'

    return zones

# --- Event Count Zoning ---
def zone_event_counts(low_events, high_events):
    total_events = low_events + high_events
    if total_events <= 1:
        return 'green'
    elif total_events <= 3:
        return 'yellow'
    else:
        return 'red'
