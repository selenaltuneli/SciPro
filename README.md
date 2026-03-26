This project focuses on forecasting daily ATM cash withdrawals and preparing model outputs for cash replenishment planning.  
The workflow starts with raw ATM and branch data, continues with feature engineering, and ends with multiple forecasting models and optimization-oriented outputs.

## Project Structure

The project is organized as a step-by-step pipeline.

### Data Preparation

- `1_merge_atm_and_branch_data.py`  
  Merges ATM and branch sheets into a single master table.

- `2_add_location_data.py`  
  Adds branch latitude and longitude information from the GeoJSON file.

- `3_holidays_added.py`  
  Extracts official holidays and school holidays and adds holiday-related features.

- `4_location_features.py`  
  Creates location-based features such as postcode and functional zone by using geographic services.

- `5_add_time_and_lag_features.py`  
  Adds calendar-based variables and lag features for ATM withdrawals.

- `6_fill_missing_atm_values.py`  
  Fills missing ATM values for selected branch-only records.

- `0_add_data.py`  
  Adds missing weekend forecast rows to the prediction output file.

### Forecasting Models

- `7_LightGBM.py`  
  Weekly rolling forecast model based on LightGBM. 

- `8_XGBoost.py`  
  Weekly rolling forecast model based on XGBoost.

- `9_XGBoost_Scenario_E.py`  
  Fixed 3-day re-optimization XGBoost model for Scenario 3.

- `11_XGBoost_Scenario_E_Daily.py`  
  Daily re-optimization XGBoost model for Scenario 4.

- `12_ANN.py`  
  Weekly ANN-based forecasting model.

- `13_SARIMAX.py`  
  Weekly SARIMAX-based forecasting model with several corrections and fallback strategies.

### Optimization

- `Optimization Final.py`  
  Main optimization model used after forecasting.

- `pipeline_Scenario0.py`  
  Prepares Scenario 0 planning inputs by computing ATM-level average demand over the available historical window.

### Evaluation and Reporting

- `optimization_metrics_revised.py`  
  Calculates run-level and scenario-level optimization performance metrics from result files.

- `forecast and real world reflected metrics.py`  
  Compares forecast-based demand metrics with realized service outcomes.

## Main Dataset

- `ATM_Branch_Data_Final_filled.xlsx`  
  This is the main input dataset used by the forecasting models.

## Recommended Workflow

The scripts are intended to be used in the following order:

1. Merge ATM and branch data.
2. Add location information.
3. Add holiday features.
4. Add location-based features.
5. Add time and lag features.
6. Fill missing ATM values if needed.
7. Train forecasting models.
8. Use the prediction outputs in the optimization stage.
9. Evaluate optimization and realized-service metrics when needed.
