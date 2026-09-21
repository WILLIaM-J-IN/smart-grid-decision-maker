# Grid-Reasoner: Power Transformer and AI Dispatch System

This repository contains the core implementation for the Grid-Reasoner project, focusing on power grid state forecasting and automated intelligent dispatching. The system leverages a custom Channel-Independent Transformer to predict future power generation across various energy sources and utilizes a Large Language Model (LLM) to determine optimal Battery Energy Storage System (BESS) operations.

## Core Features

*   **Probabilistic Forecasting Model:** Implements a Channel-Independent Transformer to predict grid data across 9 targets (Total Load, Solar, Wind, Coal, Gas, Nuclear, Oil, Hydro, Other) with quantile outputs (10th, 50th, 90th percentiles).
*   **Enhanced Custom Loss Functions:** Incorporates specialized optimization metrics, including coverage penalty loss to enforce prediction intervals, interval width loss to prevent excessively narrow bounds, and an enhanced consistency loss for boundary constraints.
*   **LLM-Based Dispatch Engine:** Integrates the Gemini API to act as an expert grid dispatcher. The engine analyzes forecasted grid states, such as total load and fossil fuel ratios, to output structured JSON actions (CHARGE, DISCHARGE, or STANDBY) along with logical reasoning.
*   **Visualization Dashboard:** Automatically generates comprehensive monitoring panels using Matplotlib. The visualizations include total load trends, fuel composition stack plots, AI dispatch logic summaries, and generation deviation error rates.

## Getting Started

To initialize the environment, run predictions, and trigger the AI dispatch engine, execute the main script:

```bash
python train.py
