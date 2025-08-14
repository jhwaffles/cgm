import pandas as pd
from datetime import datetime, time
from event_metrics import estimate_baseline_glucose, create_event_windows, compute_metrics_for_all_windows


def clean_cgm_df(df):
    df['timestamp'] = pd.to_datetime(df['Device Timestamp'], format="%m/%d/%Y %H:%M", errors='coerce')
    df['glucose'] = pd.to_numeric(df['Historic Glucose mg/dL'], errors='coerce')
    df['time_since_midnight'] = df['timestamp'] - df['timestamp'].dt.normalize()
    df['hours_since_midnight'] = df['time_since_midnight'].dt.total_seconds() / 3600
    df['date'] = df['timestamp'].dt.date
    df=df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

def filter_by_date_range(df, start_date, end_date):
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)
    return df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

def extract_events(df_filtered):  #doesnt need any glucose data, just extract time events.
    try:
        if 'Record Type' not in df_filtered.columns or 'Notes' not in df_filtered.columns:
            print("[DEBUG] extract_events: Missing 'Record Type' or 'Notes'")
            return pd.DataFrame()

        print("[DEBUG] extract_events: df_filtered.columns =", df_filtered.columns.tolist())
        print("[DEBUG] extract_events: unique 'Record Type' values =", df_filtered['Record Type'].unique())
        df_events = df_filtered[df_filtered['Record Type'].isin([6, 7])].copy()
        df_events = df_events[df_events['Notes'] != 'Exercise']
        df_events['event_type'] = df_events['Record Type'].map({6: 'food', 7: 'exercise'})
        df_events = df_events.dropna(subset=['timestamp', 'Notes']).sort_values('timestamp').reset_index(drop=True)

        print(f"[DEBUG] extract_events: Found {len(df_events)} events")
        return df_events

    except Exception as e:
        print("[DEBUG] extract_events ERROR:", e)
        return pd.DataFrame()  # fallback so pipeline doesn't crash

# 2 flows:
# for glucose signature plots and calculating glucose metrics
# df_raw 
#    → df_clean 
#       → df_glucose                    # Filter: glucose not null
#          → df_glucose_filtered       # Filter: timestamp within selected date range
# for generating event windows
# df_raw 
#    → df_clean 
#       → df_filtered                  # Filter: timestamp within selected date range
#          → df_events                 # Filter: Record Type 6 or 7
#             → df_food_events        # Subset: just food events (can do exercise later)
#                → event_windows      # Generate time windows (e.g., +2h from meal)
#                   → df_event_metrics # Compute AUC, peak, etc.

def run_event_metrics_pipeline(df_raw, start_date, end_date):
    df_clean = clean_cgm_df(df_raw)
    df_glucose = df_clean[~df_clean['glucose'].isna()].copy()  

    df_filtered = filter_by_date_range(df_clean, start_date, end_date) 
    df_glucose_filtered = filter_by_date_range(df_glucose, start_date, end_date) 

    df_events = extract_events(df_filtered)

    if df_events.empty:
        print("[DEBUG] run_event_metrics_pipeline: No events found")
        return {
            "df_clean": df_clean,
            "df_glucose_filtered": df_glucose_filtered,
            "df_events": pd.DataFrame(),
            "event_windows": pd.DataFrame(),
            "df_event_metrics": pd.DataFrame(),
            "baseline": estimate_baseline_glucose(df_glucose_filtered)
        }

    df_food_events = df_events[df_events['event_type'] == 'food']

    baseline = estimate_baseline_glucose(df_filtered)
    event_windows = create_event_windows(df_food_events)
    df_event_metrics = compute_metrics_for_all_windows(event_windows, df_glucose_filtered, df_events, baseline)

    return {
        "df_clean": df_clean,
        "df_glucose_filtered": df_glucose_filtered,
        "df_events": df_events,
        "event_windows": event_windows,
        "df_event_metrics": df_event_metrics,
        "baseline": baseline
    }