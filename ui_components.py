from shiny import ui

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
    return f'<span title="{explanation}" style="cursor: help;">{text}<span style="font-weight: bold; color: #007bff; padding-left: 4px;">ℹ️</span></span>'

zone_tooltips = {
    ">250": "Percent time spent in >250 mg/dL range (very high). Increased risk of hyperglycemia complications.",
    "181–250": "Percent time spent in 181-250 mg/dL range (Above target range). Generally too high.",
    "70–180": "Percent time spent in 70-180 mg/dL range (Target range). Goal is >70% of time here.",
    "54–69": "Percent time spent in 54-69 mg/dL range (Mild hypoglycemia).",
    "<54": "Percent time spent in <54 mg/dL (Severe low glucose). Increased immediate risk."
}

def signature_plot_title():
    return tooltip(
        "Ambulatory Glucose Profile (AGP)",
        "Overview of user's average glucose levels, summarized across multiple days. Helps identify daily patterns. Typically based on 10–14 days of CGM data."
    )

def generate_metrics_summary_ui(baseline, mean_glucose, gmi, zones):
    return ui.HTML(f"""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;'>
        <!-- Column 1: Basic -->
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
            <h5 style='margin-bottom: 10px;'>Basic Metrics</h5>
            <div style='background-color: {color_glucose(baseline)}; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('Baseline Glucose', 'Estimated by averaging the lowest 30% of hourly minimum glucose values. Reflects typical pre-meal or fasting levels. Ideal: 80–100 mg/dL. (ADA, 2024)')}</b>: {baseline:.1f} mg/dL
            </div>
            <div style='background-color: {color_glucose(mean_glucose)}; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('Mean Glucose', 'Arithmetic average of all glucose values in the selected date range. Target: 70–130 mg/dL. Source: ADA Standards of Medical Care in Diabetes, 2024')}</b>: {mean_glucose:.1f} mg/dL
            </div>
            <div style='background-color: {color_gmi(gmi)}; padding: 5px; border-radius: 3px;'>
                <b>{tooltip('GMI','Converts mean glucose to an estimated A1C value. GMI < 6.0% is ideal. Derived from: Bergenstal et al., 2018.')}</b>: {gmi:.2f}
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
    </div>
    """)

def risk_trace_title():
    return tooltip(
        "Risk Indices (LBGI / HBGI) Over Time",
        "Risk indices quantifying severity of low/high glucose excursions using LBGI and HBGI. Derived from log-transformed risk equations. (Clarke & Kovatchev, 2008)"
    )

def roc_histogram_title():
    return tooltip(
        "Histogram of Glucose Rate of Change",
        "Distribution of glucose change rates, sampled every 15 minutes. Captures stability vs volatility in glycemic patterns."
    )

def poincare_plot_title():
    return tooltip(
        "Poincaré Plot",
        "Scatterplot of glucose at time t vs time t-1. Visualizes short-term (SD1) vs long-term (SD2) variability. A confidence ellipse highlights the spread."
    )