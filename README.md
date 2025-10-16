Îlots de chaleur & précipitations — LATMOS project

This repository hosts the PDF report “Îlots de chaleur et précipitations” produced at LATMOS (CNRS). The work explores statistical and ML methods to study interactions between urban heat islands (UHIs) and precipitation patterns over an urban region.

Report: Rapport_Îlots_de_chaleur_et_précipitations.pdf
 (in French). 

Project at a glance

Goal: quantify links between UHI intensity and rainfall occurrence/intensity, and characterize spatio-temporal organization of precipitation around urban areas.

Data (typical sources):

Urban/temperature indicators (e.g., station networks or reanalysis / satellite LST).

Radar/rain-gauge precipitation fields (event-level and aggregated).

Methods (high-level):

Preprocessing & quality control of meteorological series.

Spatio-temporal event definition and clustering (e.g., DBSCAN), with optional sequence similarity (e.g., DTW).

Feature extraction: UHI metrics, upwind/downwind stratification, urban vs rural buffers.

Statistical association tests and simple predictive baselines.

Outputs: figures/tables linking UHI strength to rainfall probability, intensity and localization; case studies & sensitivity checks.

The exact dataset list and parameters are documented in the report PDF; this README provides a structured summary for GitHub.
