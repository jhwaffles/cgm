import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import norm

def create_signature_plot(df):
    if df.empty:
        return go.Figure()

    if 'hours_rounded' not in df.columns:
        df['hours_rounded'] = df['hours_since_midnight'].round().astype(int)

    grouped = df.groupby('hours_rounded')['glucose']
    summary = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack().reset_index()
    summary.columns = ['hours_rounded', 'p05', 'p25', 'p50', 'p75', 'p95']

    fig = go.Figure()

    # --- 5th–95th percentile band (light shading) ---
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=summary['p95'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=summary['p05'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,150,255,0.1)',  # Lighter blue
        line=dict(width=0),
        name='5–95% Range',
        hoverinfo='skip'
    ))

    # --- 25th–75th percentile band (darker shading) ---
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=summary['p75'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=summary['p25'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,150,255,0.3)',  # Darker blue
        line=dict(width=0),
        name='25–75% Range',
        hoverinfo='skip'
    ))

    # --- 50th percentile (median) line ---
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=summary['p50'],
        mode='lines+markers',
        name='Median Glucose (50%)',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    # --- Target range lines ---
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=[70] * len(summary),
        mode='lines',
        name='Lower Target (70)',
        line=dict(color='green', dash='solid', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=summary['hours_rounded'],
        y=[180] * len(summary),
        mode='lines',
        name='Upper Target (180)',
        line=dict(color='green', dash='solid', width=2)
    ))

    # --- Layout styling ---
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 25, 2)),
            ticktext=[f'{h % 12 or 12} {"AM" if h < 12 else "PM"}' for h in range(0, 25, 2)],
            title='Time of Day',
            gridcolor='lightblue'
        ),
        yaxis=dict(
            title='Glucose (mg/dL)',
            gridcolor='lightblue'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    return fig



def create_event_plot(df_day, df_events, day_str, baseline_val, show_events=False):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = list(df_day['timestamp'].dt.to_pydatetime()),
        y=df_day['glucose'],
        mode='lines+markers',
        name='Glucose',
        hovertemplate='Time: %{x}<br>Glucose: %{y} mg/dL'
    ))

    if show_events:
        for _, row in df_events.iterrows():
            print(f"[DEBUG] Processing row: {row['event_note']} from {row['window_start']} to {row['window_end']}")

            # Only show events for selected day
            if str(row['window_start'].date()) != day_str:
                continue  

            mask = (df_day['timestamp'] >= row['window_start']) & (df_day['timestamp'] <= row['window_end'])

            #one segment for each event. can capture 8-10 readings per event.
            segment = df_day.loc[mask] 
            print(f"[DEBUG] Segment rows for event '{row['event_note']}': {len(segment)}")
            print(f"[DEBUG] Segment preview:\n{segment[['timestamp', 'glucose']].head()}")


            above = segment[segment['glucose'] > baseline_val]
            if above.empty:
                continue
            print(f"[DEBUG] Points above baseline: {len(above)}")

            #makes a closed polygon in the plot. we just do another conversion to pydatetime and put it in a list to pass it into plotly safely.
            fill_x = list(above['timestamp'].dt.to_pydatetime()) + list(above['timestamp'].dt.to_pydatetime()[::-1])
            fill_y = list(above['glucose']) + [baseline_val] * len(above)
            n_points = len(fill_x)

            #this is passed to the hover template. each point in the 'curve' gets the same hover info.
            customdata = np.array([[
                row['event_note'],
                row['window_start'].strftime("%Y-%m-%d %H:%M"),
                row['window_end'].strftime("%Y-%m-%d %H:%M"),
                row['glucose_max'],
                row['glucose_auc']
            ]] * n_points)
            print(f"[DEBUG] AUC trace for '{row['event_note']}' — fill_x: {len(fill_x)}, fill_y: {len(fill_y)}")

            fig.add_trace(go.Scatter(
                x=fill_x,
                y=fill_y,
                fill='toself',
                mode='lines+markers',
                marker=dict(size=1, color='rgba(0,0,0,0)'),
                line=dict(color='rgba(255, 165, 0, 0.2)'),
                fillcolor='rgba(255, 165, 0, 0.3)',
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" +
                    "Start: %{customdata[1]}<br>" +
                    "End: %{customdata[2]}<br>" +
                    "Max Glucose: %{customdata[3]} mg/dL<br>" +
                    "AUC above baseline: %{customdata[4]:.1f}<extra></extra>"
                ),
                showlegend=False
            ))

    fig.update_layout(
        title=f'Glucose Events for {day_str}',
        xaxis_title='Timestamp',
        yaxis_title='Glucose (mg/dL)',
        hovermode='closest'
    )
    return fig

def plot_risk_trace(df_risk):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(df_risk["timestamp"].dt.to_pydatetime()),
        y=-df_risk["LBGI"], #flip for below zero bar
        name="LBGI",
        marker_color="orange"
    ))
    fig.add_trace(go.Bar(
        x=list(df_risk["timestamp"].dt.to_pydatetime()),
        y=df_risk["HBGI"],
        name="HBGI",
        marker_color="red"
    ))
    fig.update_layout(
        xaxis=dict(
            title="Time",
            tickformat="%b %d",  # e.g., Jul 04 06:00
            tickangle=45,
            tickmode="linear", 
            dtick=86400000,  #every 24 hours
            showgrid=True,
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title="Risk Index",
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray"
        ),
        barmode="relative",
        plot_bgcolor="white",
    )
    return fig

def plot_roc_histogram(roc_series, nbins=50):
    roc_series = roc_series.dropna()

    # Compute histogram
    hist_vals, bin_edges = np.histogram(roc_series, bins=nbins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Fit normal distribution
    mu = roc_series.mean()
    sigma = roc_series.std()
    normal_y = norm.pdf(bin_centers, loc=mu, scale=sigma)

    # Create Plotly figure
    fig = go.Figure()

    # Histogram trace
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_vals,
        name="Rate of Change Histogram",
        marker_color="lightblue",
        opacity=0.7
    ))

    # Normal curve trace
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=normal_y,
        name=f"Normal Fit (σ = {sigma:.2f})",
        mode="lines",
        line=dict(color="black", dash="dash")
    ))

    # Layout tweaks
    fig.update_layout(
        xaxis_title="Rate of Change (mg/dL/min)",
        yaxis_title="Density",
        bargap=0.1,
        plot_bgcolor="white",
        legend=dict(x=0.7, y=0.95),
    )

    return fig

def compute_sd1_sd2(x, y):
    diffs = x - y
    sums = x + y

    SD1 = np.std(diffs / np.sqrt(2))
    SD2 = np.std(sums / np.sqrt(2))
    return SD1, SD2

def generate_ellipse(mean_x, mean_y, SD1, SD2, n_std=1.0, resolution=100):
    t = np.linspace(0, 2 * np.pi, resolution)
    ellipse = np.array([n_std * SD2 * np.cos(t), n_std * SD1 * np.sin(t)])

    # Rotate by 45 degrees (identity line)
    rotation_matrix = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4),  np.cos(np.pi/4)]
    ])
    rotated = rotation_matrix @ ellipse

    x_ellipse = rotated[0] + mean_x
    y_ellipse = rotated[1] + mean_y
    return x_ellipse, y_ellipse

def plot_poincare(df):
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    SD1, SD2 = compute_sd1_sd2(x, y)
    x_ellipse, y_ellipse = generate_ellipse(mean_x, mean_y, SD1, SD2, n_std=2.45)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="BG Pairs",
        marker=dict(color="blue", opacity=0.5),
        showlegend=False
    ))

    # Identity line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="gray", dash="dash"),
        name="Line of Identity"
    )

    # Ellipse trace
    fig.add_trace(go.Scatter(
        x=x_ellipse,
        y=y_ellipse,
        mode="lines",
        name=f"95% Ellipse (SD1={SD1:.2f}, SD2={SD2:.2f})",
        line=dict(color="orange", width=2)
    ))

    fig.update_layout(
        autosize=True,
        title="Poincaré Plot with Confidence Ellipse",
        xaxis=dict(
            title=df.columns[0],
            showgrid=True,
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title=df.columns[1],
            showgrid=True,
            gridcolor="lightgray"
        ),
        plot_bgcolor="white",
        showlegend=True
    )
    return fig
