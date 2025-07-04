import pandas as pd
import numpy as np
from datetime import datetime, time
from shiny import App, render, ui, reactive
from shinywidgets import render_widget, output_widget
from process_fxns import create_event_windows, compute_metrics_for_all_windows,estimate_baseline_glucose,calculate_gmi,compute_iqr,compute_sd_of_rate,compute_LBGI_HBGI,time_in_zone
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go

#input file
#select time filter
#output graph with values so user day by day
#output metrics + comparison with population values
#can take peak/llm analysis offline

app_ui = ui.page_fluid(
    ui.panel_title("CGM Report"),
    #Section 1 - Overview
    ui.layout_columns(
        #left column, inputs and metrics (25%)
        ui.card(
            ui.input_file("file", "Upload CGM CSV"),
            ui.input_date_range("date_range", "Date Range", start="2022-07-01", end="2022-07-14")
        ),
        #right column, Signature Plot (75%)
        ui.card(
            output_widget("signature_plot"),
            ui.layout_columns(
                ui.div(ui.output_ui("metrics_summary_ui"))
            ),
            title="Summary Metrics",
            full_screen=True
        ),
        

        col_widths=(3, 9)
    ),
    #Section 2 - Day to Day View
    ui.input_selectize("selected_days","Choose a day",choices=[],multiple=True),
    output_widget("daily_plot")
)

# Server logic
def server(input, output, session):
    @reactive.Calc
    def df_raw():
        file = input.file()
        if not file:
            return pd.DataFrame()
        df = pd.read_csv(file[0]["datapath"], skiprows=1)
        return df

    @reactive.Calc
    def df_clean():
        df=df_raw()
        if df.empty:
            return pd.DataFrame()
        try:
            df['timestamp'] = pd.to_datetime(df['Device Timestamp'], format="%m/%d/%Y %H:%M", errors='coerce')
            df['glucose'] = pd.to_numeric(df['Historic Glucose mg/dL'], errors='coerce')
            df['time_since_midnight'] = df['timestamp'] - df['timestamp'].dt.normalize()
            df['hours_since_midnight'] = df['time_since_midnight'].dt.total_seconds() / 3600
            df['date'] = df['timestamp'].dt.date
            df = df.dropna(subset=['timestamp', 'glucose']).sort_values('timestamp').reset_index(drop=True)
            return df
        
        except Exception as e:
            print("Error in df_clean:", e)
            return pd.DataFrame()

    @reactive.Effect
    def update_date_range():
        df = df_clean()
        if df.empty:
            return pd.DataFrame()

        # Determine min and max dates from timestamp
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        # Update the date range input UI
        ui.update_date_range(
            "date_range",
            start=min_date,
            end=max_date
        )
        
    @reactive.Calc
    def df_filtered():
        df = df_clean()
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        start_date, end_date = input.date_range()
    
        start_dt = datetime.combine(start_date, time.min)
        end_dt = datetime.combine(end_date, time.max)
        return df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
    
    # Rendered as styled UI, not raw table
    @output
    @render.ui
    def metrics_summary_ui():
        df = df_filtered()
        if df.empty:
            return ui.div("No data loaded")

        baseline = estimate_baseline_glucose(df)
        mean_glucose = df['glucose'].mean()
        gmi = calculate_gmi(mean_glucose)
        zones = time_in_zone(df['glucose'])
        iqr = compute_iqr(df['glucose'])
        roc_sd = compute_sd_of_rate(df['glucose'])
        lbgi, hbgi, bgri = compute_LBGI_HBGI(df['glucose'])

        def color_zone(label):
            if label == "70–180":
                return "#d4edda"  # green
            elif label in ["181–250", "54–69"]:
                return "#fff3cd"  # yellow
            else:
                return "#f8d7da"  # red

        def color_glucose(val):
            if 80 <= val <= 100:
                return "#d4edda"  # green
            elif 70 <= val < 80 or 100 < val <= 130:
                return "#fff3cd"  # yellow
            else:
                return "#f8d7da"  # red (too low or too high)

        def color_gmi(val):
            if val < 6.0:
                return "#d4edda"  # green-ish
            elif val < 6.5:
                return "#fff3cd"  # yellow-ish
            return "#f8d7da"      # red-ish
        
        def color_zone(label):
            if label == "70–180":
                return "#d4edda"  # green
            elif label in ["181–250", "54–69"]:
                return "#fff3cd"  # yellow
            else:
                return "#f8d7da"  # red

        def tooltip(text, explanation):
            return f"<span title='{explanation}'>{text} ℹ️</span>"

        return ui.HTML(f"""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;'>
        <!-- Column 1: Basic -->
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
            <h5 style='margin-bottom: 10px;'>Basic Metrics</h5>
            <div style='background-color: {color_glucose(baseline)}; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('Baseline Glucose', 'Typical pre-meal glucose level. Ideal: 80–100 mg/dL. Source: ADA Standards of Medical Care in Diabetes, 2024')}</b>: {baseline:.1f} mg/dL
            </div>
            <div style='background-color: {color_glucose(mean_glucose)}; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('Mean Glucose', 'Average glucose across time range. Target: 70–130 mg/dL. Source: ADA Standards of Medical Care in Diabetes, 2024')}</b>: {mean_glucose:.1f} mg/dL
            </div>
            <div style='background-color: {color_gmi(gmi)}; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('GMI', 'Glucose Management Indicator: estimates A1C. GMI < 6.0% is ideal.')}</b>: {gmi:.2f}
            </div>
        </div>

        <!-- Column 2: Time in Range -->
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
            <h5 style='margin-bottom: 10px;'>Time in Range</h5>
            {''.join([
                f"<div style='background-color: {color_zone(label)}; padding: 5px; border-radius: 3px;'>"
                f"<b>{tooltip(label + ' mg/dL', zone_tooltips.get(label, ''))}</b>: {pct:.1f}%</div>"
                for label, pct in zones.items()
            ])}
        </div>

        <!-- Column 3: Advanced Metrics -->
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
            <h5 style='margin-bottom: 10px;'>Advanced Metrics</h5>
            <div style='background-color: #e9ecef; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('IQR', 'Interquartile Range of glucose values. Smaller = more stable.')}</b>: {iqr:.1f} mg/dL
            </div>
            <div style='background-color: #e9ecef; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('Rate of Change SD', 'Standard deviation of glucose rate-of-change. Higher = more variability.')}</b>: {roc_sd:.2f} mg/dL/min
            </div>
            <div style='background-color: #e9ecef; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('LBGI', 'Low Blood Glucose Index: Risk of hypoglycemia. <2.5 is desirable.')}</b>: {lbgi:.2f}
            </div>
            <div style='background-color: #e9ecef; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('HBGI', 'High Blood Glucose Index: Risk of hyperglycemia. <2.5 is desirable.')}</b>: {hbgi:.2f}
            </div>
            <div style='background-color: #e9ecef; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('BGRI', 'Blood Glucose Risk Index: Combined LBGI + HBGI.')}</b>: {bgri:.2f}
            </div>
        </div>
    </div>
    """)

    zone_tooltips = {
        ">250": "Very high glucose. Increased risk of hyperglycemia complications.",
        "181–250": "Above target range. Generally too high.",
        "70–180": "Target range. Goal is >70% of time here.",
        "54–69": "Below range. Mild hypoglycemia.",
        "<54": "Severe low glucose. Increased immediate risk."
    }

    @reactive.Effect
    def update_day_choices():
        df=df_filtered()
        if df.empty:
            return
        unique_days=sorted(df['date'].astype(str).unique())
        ui.update_selectize("selected_days",choices=unique_days)

    @output
    @render_widget
    def signature_plot():
        df = df_filtered()
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
            title='Ambulatory Glucose Profile (AGP)',
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


    @reactive.Calc
    def df_selected_days():
        df=df_filtered()
        selected=input.selected_days()
        if not selected or df.empty:
            return pd.DataFrame()
        return df[df['date'].astype(str).isin(selected)]

    @output
    @render_widget
    def event_plot():
        df = df_raw()
        df_filtered_data=df_filtered()
        if df.empty:
            return go.Figure()
# Identify and clean event rows
        df_events = df[df['Record Type'].isin([6, 7])].copy()
        df_events = df_events.drop(df_events[df_events['Notes'] == 'Exercise'].index)
        df_events['event_type'] = df_events['Record Type'].map({6: 'food', 7: 'exercise'})
        df_events = df_events.dropna(subset=['timestamp', 'Notes']).sort_values('timestamp').reset_index(drop=True)

        # Compute baseline and event windows
        baseline = estimate_baseline_glucose(df)
        df_food_event_windows = create_event_windows(df_events[df_events['event_type'] == 'food'].copy())
        # df_event_metrics = compute_metrics_for_all_windows(df_food_event_windows, df_filtered_data, df_events, baseline)

        # # Create plot
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     x=df['timestamp'],
        #     y=df['glucose'],
        #     mode='lines+markers',
        #     name='Glucose',
        #     hovertemplate='Time: %{x}<br>Glucose: %{y} mg/dL'
        # ))

        # for _, row in df_event_metrics.iterrows():
        #     mask = (df['timestamp'] >= row['window_start']) & (df['timestamp'] <= row['window_end'])
        #     segment = df.loc[mask]
        #     above = segment[segment['glucose'] > baseline]
        #     if above.empty:
        #         continue

        #     fill_x = list(above['timestamp']) + list(above['timestamp'][::-1])
        #     fill_y = list(above['glucose']) + [baseline] * len(above)
        #     n_points = len(fill_x)
        #     customdata = np.array([[row['event_note'], row['window_start'], row['window_end'],
        #                             row['glucose_max'], row['glucose_auc']]] * n_points)

        #     fig.add_trace(go.Scatter(
        #         x=fill_x,
        #         y=fill_y,
        #         fill='toself',
        #         mode='lines+markers',
        #         marker=dict(size=1, color='rgba(0,0,0,0)'),
        #         line=dict(color='rgba(255, 165, 0, 0.2)'),
        #         fillcolor='rgba(255, 165, 0, 0.3)',
        #         customdata=customdata,
        #         hovertemplate=(
        #             "<b>%{customdata[0]}</b><br>" +
        #             "Start: %{customdata[1]}<br>" +
        #             "End: %{customdata[2]}<br>" +
        #             "Max Glucose: %{customdata[3]} mg/dL<br>" +
        #             "AUC above baseline: %{customdata[4]:.1f}<extra></extra>"
        #         ),
        #         showlegend=False
        #     ))

        # fig.update_layout(
        #     title='Glucose Readings with Highlighted Event Windows',
        #     xaxis_title='Timestamp',
        #     yaxis_title='Glucose (mg/dL)',
        #     hovermode='closest'
        # )

        # return fig


    @output
    @render_widget
    def daily_plot():
        df = df_selected_days()
        if df.empty:
            return px.scatter(title="No data available")
        fig = px.line(
            df,
            x="hours_since_midnight",
            y="glucose",
            color="date",
            markers=True,
            title="Glucose Trends Over 24 Hours by Day"
        )
        fig.update_layout(
            plot_bgcolor='white',       # ✅ White plotting area
            paper_bgcolor='white',      # ✅ White outer background
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 25, 2)),
                ticktext=[f'{h % 12 or 12} {"AM" if h < 12 else "PM"}' for h in range(0, 25, 2)],
                title='Time of Day',
                gridcolor='lightblue',  # ✅ Light blue gridlines
                zeroline=False
            ),
            yaxis=dict(
                title='Glucose (mg/dL)',
                gridcolor='lightblue',  # ✅ Light blue gridlines
                zeroline=False
            ),
        )
        return fig

app = App(app_ui, server)
#calculate aggregated statistic
#Baseline

#for each peak
#area under curve
#peak height, duration of peak, rate of decrease