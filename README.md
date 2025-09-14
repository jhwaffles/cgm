This app lets clinicians/users analyze cgm data: https://0198a743-8aac-6e01-b377-b7c90b9f1907.share.connect.posit.cloud/

app.py - main file for shiny interactive app. contains ui, and server logic, and reactive functions.
processing_pipeline.py - main processing file which uses pandas. imports .csv files and outputs clean dataframes for further analysis.
event_metrics.py - contains functions for calculating event windows and metrics
glucose_metrics.py - contains functions for calculating in glucose metrics in standard report. Also contains advanced metrics
plots.py - interactively plotly graphs.
ui_components.py - supporting file for shiny app.

Regression and Random Forest models can be found in /notebooks