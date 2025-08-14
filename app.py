import pandas as pd
import numpy as np
import os
from datetime import datetime, time
from shiny import App, render, ui, reactive
from shinywidgets import render_widget, output_widget
from glucose_metrics import calculate_gmi,time_in_zone, compute_risk_trace, compute_rate_of_change, compute_poincare_data
from ui_components import generate_metrics_summary_ui, signature_plot_title, risk_trace_title, roc_histogram_title, poincare_plot_title
from plots import create_signature_plot,create_event_plot, plot_risk_trace, plot_roc_histogram, plot_poincare
from processing_pipeline import run_event_metrics_pipeline
generate_metrics_summary_ui
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go


#output metrics + comparison with population values. put some better context on these metrics. 1. what is the Ambuloatory glucose profile? population metrics?
#output graph with values so user day by day. incorporate better baseline algorithm? Put this on to-do:
#can take peak/llm analysis offline (download). Add download button.  DONE
#give 'example' anonymized profiles. DONE
#put together a readme.

app_ui = ui.page_fluid(
    ui.panel_title("CGM Report"),
    ui.page_navbar(
        ui.nav_panel("Overall Summary",
            ui.markdown("Upload your CGM CSV file and select a date range. View an aggregate glucose profile (Ambulatory Glucose Profile), baseline glucose, GMI, and time-in-range."),
            ui.layout_columns(
                # Left column: inputs
                ui.card(
                    ui.input_file("file", "Upload CGM CSV"),
                    ui.input_date_range("date_range", "Date Range", start="2022-07-01", end="2022-07-14")
                ),
                # Right column: signature plot + metrics
                ui.card(
                    ui.HTML(f"<h5>{signature_plot_title()}</h5>"),  #from ui_components.py
                    output_widget("signature_plot"),
                    ui.output_ui("metrics_summary_ui"),
                    full_screen=True
                ),
                col_widths=(3, 9)
            )
        ),

        ui.nav_panel("Day-by-Day View",
            ui.markdown("Select a specific day to view glucose traces. Optionally overlay logged events to see post-meal glucose responses."),
            ui.layout_columns(
                ui.card(
                    ui.h5("Select a Day"),
                    ui.output_ui("day_buttons"),
                    ui.input_checkbox("show_logged_events", "Display Logged Events", value=True),
                ),
                ui.card(
                    output_widget("event_plot"),
                    title="Glucose Events for Selected Day",
                    full_screen=True
                ),
                col_widths=(3, 9)
            )
        ),

        ui.nav_panel("Event Metrics Table",
            ui.markdown("Browse a table of calculated metrics for each logged food event. Export to CSV for further analysis."),
            ui.card(
                ui.h5("Event AUC Metrics"),
                ui.output_data_frame("event_table"),
                ui.download_button("download_event_metrics", "Download as CSV", class_="mb-3")

            )
        ),
        ui.nav_panel("Advanced Metrics",
            ui.markdown("Explore risk indices (LBGI/HBGI), rate-of-change histogram, and a Poincaré plot showing glycemic variability."),
            ui.card(
                ui.output_ui("risk_trace_title_ui"),
                output_widget("risk_trace_plot"),
                ui.hr(),
                ui.output_ui("roc_histogram_title_ui"),
                output_widget("roc_histogram"),
                ui.hr(),
                ui.output_ui("poincare_plot_title_ui"),
                output_widget("poincare_plot")
            )
        ),
        ui.nav_panel("About",
            ui.card(
                ui.HTML("""
                    <h3>Overview</h3>
                    <p>
                    This app helps clinicians and patients analyze uploaded Continuous Glucose Monitoring (CGM) data. It enables both high-level summary and detailed event-by-event insights into glycemic patterns.
                    </p>

                    <h3>Summary Metrics (AGP Overview)</h3>
                    <p>The Ambulatory Glucose Profile (AGP) summarizes glucose trends across the day, typically using 10–14 days of data.</p>
                    <p><strong>Key metrics include:</strong></p>
                    <ul>
                    <li><strong>Baseline Glucose:</strong> Typical pre-meal glucose level</li>
                    <li><strong>Mean Glucose</strong></li>
                    <li><strong>GMI (Glucose Management Indicator):</strong> An estimate of A1C</li>
                    <li><strong>Time in Range:</strong> % time spent in target zones (e.g., 70–180 mg/dL)</li>
                    </ul>
                    <p><em>Reference: Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range (ADA, 2019).</em></p>

                    <h3>Day-by-Day View</h3>
                    <ul>
                    <li>Allows users to explore individual days.</li>
                    <li>A “Display Logged Events” checkbox overlays 2-hour windows after logged food entries.</li>
                    <li>Each event is highlighted above a calculated glucose baseline to visualize postprandial excursions.</li>
                    </ul>

                    <h3>Event Metrics Table</h3>
                    <p>Summarizes glucose features from each logged food event.</p>
                    <p><strong>Each 2-hour window includes:</strong></p>
                    <ul>
                    <li>Maximum and minimum glucose</li>
                    <li>Area under the curve (AUC) above baseline</li>
                    <li>Max rate of rise and fall</li>
                    <li>Whether exercise occurred within 3 hours</li>
                    <li>Meal time category (e.g., breakfast, lunch)</li>
                    </ul>
                    <p>The table can be downloaded for further analysis.</p>
                    <p><em>Reference: Zeevi et al., <i>Personalized Nutrition by Prediction of Glycemic Responses</i>, Cell, 2015.</em></p>

                    <h3>Advanced Metrics</h3>
                    <p>Three research-grade tools to quantify glucose variability:</p>
                    <ul>
                    <li><strong>Risk Trace</strong>: Plots LBGI (Low BG Index) and HBGI (High BG Index) over time. These reflect the frequency and severity of hypo- and hyperglycemia risk. Based on log-transformed risk scoring of glucose.</li>
                    <li><strong>Rate of Change Histogram</strong>: Histogram of glucose slopes (mg/dL/min), downsampled to 15-min intervals to reduce noise.</li>
                    <li><strong>Poincaré Plot</strong>: Visualizes short- and long-term variability (SD1, SD2). Ellipse shows 95% confidence coverage.</li>
                    </ul>
                    <p><em>Reference: Clarke and Kovatchev, <i>Statistical Tools to Analyze Continuous Glucose Monitor Data</i>, Diabetes Technology & Therapeutics, 2008.</em></p>
                """)
            )
        )
    )
)
# Server logic
def server(input, output, session):
    selected_day_value = reactive.Value(None)
    previous_file_name = reactive.Value(None)

    @reactive.Calc
    def data_bundle():
        file = input.file()

        # If user uploads a file, use it
        if file:
            df_raw = pd.read_csv(file[0]["datapath"], skiprows=1)
            print("[DEBUG] Using user-uploaded file.")
        else:
            return {}

        start_date, end_date = input.date_range()
        return run_event_metrics_pipeline(df_raw, start_date, end_date)


    @reactive.Calc
    def df_glucose_filtered():
        return data_bundle().get("df_glucose_filtered", pd.DataFrame())

    @reactive.Calc
    def df_event_metrics():
        return data_bundle().get("df_event_metrics", pd.DataFrame())
        
    @reactive.Calc
    def baseline():
        return data_bundle().get("baseline", None)
    
    # Rendered as styled UI, not raw table
    @output
    @render.ui
    def metrics_summary_ui():
        df = df_glucose_filtered()
        if df.empty:
            return ui.div("No data loaded")

        baseline_val = baseline()
        mean_glucose = df['glucose'].mean()
        gmi = calculate_gmi(mean_glucose)
        zones = time_in_zone(df['glucose'])
        return generate_metrics_summary_ui(baseline_val, mean_glucose, gmi, zones)
    
    @reactive.Effect
    def update_date_range():
        file = input.file()
        if not file:
            return

        current_file_name = file[0]["name"]
        if previous_file_name.get() == current_file_name:
            return  # Same file — do not reset date range

        previous_file_name.set(current_file_name)  # Update to new file

        df = data_bundle().get("df_clean", pd.DataFrame())
        if df.empty or 'timestamp' not in df.columns:
            return

        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        print(f"[DEBUG] New file uploaded: resetting date range to {min_date} - {max_date}")

        ui.update_date_range(
            "date_range",
            start=min_date,
            end=max_date,
            min=min_date,
            max=max_date
        )

    @reactive.Effect
    def update_day_choices():
        df=df_glucose_filtered()
        if df.empty:
            return
        unique_days=sorted(df['date'].astype(str).unique())
        ui.update_selectize("selected_days",choices=unique_days)

    @output
    @render_widget
    def signature_plot():
        df = df_glucose_filtered()
        return create_signature_plot(df)

    @reactive.Calc
    def available_days():  #only in selected filter
        df = df_glucose_filtered()
        if df.empty:
            return []
        return sorted(df['date'].astype(str).unique())
    
    @output
    @render.ui
    def day_buttons():
        days = available_days()
        selected_day = selected_day_value.get()

        if not days:
            return ui.div("No days available")

        # Create two columns
        column1 = []
        column2 = []

        for i, day in enumerate(days):
            is_selected = (day == selected_day)
            btn_class = "btn btn-primary mb-2 w-100" if is_selected else "btn btn-outline-secondary mb-2 w-100"

            btn = ui.input_action_link(f"day_select_{i}", label=day, class_=btn_class)

            # Alternate between columns
            if i % 2 == 0:
                column1.append(btn)
            else:
                column2.append(btn)

        return ui.div(
            ui.div(*column1, class_="col"),
            ui.div(*column2, class_="col"),
            class_="row"
        )
    
    @reactive.Effect
    def handle_day_selection():
        days = available_days()
        for i, day in enumerate(days):
            if input[f"day_select_{i}"]() > 0:
                selected_day_value.set(day)
                print(f"[DEBUG] Day selected: {day}")

    @output
    @render_widget
    def event_plot():
        df=df_glucose_filtered().copy()
        df_events = df_event_metrics()
        day_str = selected_day_value.get()
        show_events = input.show_logged_events()

        if not day_str:
            return go.Figure()
        df_day = df[df['date'].astype(str) == day_str]
        baseline_val = baseline()
        return create_event_plot(df_day, df_events, day_str, baseline_val, show_events)

    @output
    @render.data_frame
    def event_table():
        return df_event_metrics()

    @render.download(filename="event_metrics.csv")
    def download_event_metrics():
        df = df_event_metrics()
        yield df.to_csv(index=False)

    @output
    @render.ui
    def risk_trace_title_ui():
        return ui.HTML(f"<h5>{risk_trace_title()}</h5>")

    @output
    @render.ui
    def roc_histogram_title_ui():
        return ui.HTML(f"<h5>{roc_histogram_title()}</h5>")

    @output
    @render.ui
    def poincare_plot_title_ui():
        return ui.HTML(f"<h5>{poincare_plot_title()}</h5>")
    
    @output
    @render_widget
    def risk_trace_plot():
        df = df_glucose_filtered()
        if df.empty:
            return go.Figure()
        df_risk = compute_risk_trace(df)

        return plot_risk_trace(df_risk)

    @output
    @render_widget
    def roc_histogram():
        df = df_glucose_filtered()
        if df.empty:
            return go.Figure()
        roc = compute_rate_of_change(df["glucose"], df["timestamp"])
        return plot_roc_histogram(roc)

    @output
    @render_widget
    def poincare_plot():
        df = df_glucose_filtered()
        if df.empty:
            return go.Figure()
        poincare_df = compute_poincare_data(glucose_series=df["glucose"],time_series=df["timestamp"])
        return plot_poincare(poincare_df)

app = App(app_ui, server)