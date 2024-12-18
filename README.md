# Dynamic Projection Monthly Linear with Seasonality

This script performs time series forecasting using Facebook Prophet, specifically designed for monthly linear projections with seasonal components.

## Description

This tool generates forecasts for customer data, particularly suitable for:
- Non-seasonal industries (e.g., healthcare, utilities)
- Customers transitioning from on-premises to cloud infrastructure
- Linear growth expectations

## Prerequisites

### Required Packages
```python
Prophet
SQLAlchemy==1.4.46
pandasql
numpy
pandas
matplotlib
```

### Input Requirements
The input CSV file must contain the following fields:
- key_field
- month_start (format: YYYY-MM-DD)
- sum_recs

### Output Format
The script generates a CSV file with the following columns:
- ds (date)
- key_field
- yhat (projection)
- conf_score
- updated_on

## Configuration

Key parameters that can be adjusted:
```python
req_obs = 0              # Minimum required observations
cap_growth = 1.05        # Growth cap
fperiods = 14           # Forecast periods
cap_pctile = 1          # Percentile cap
```

## Usage

1. Set the following variables:
```python
customer_id = ''
output_bucket = 's3://'
input_file = 's3://whatever.csv'
output_file = output_bucket + 'dynamic_projection_monthly_linear_w_season_' + customer_id + '.csv'
```

2. Run the script to:
- Process input data
- Generate forecasts
- Create visualizations
- Output results to specified S3 location

## Features

- Handles negative values (converts to zero)
- Includes confidence scoring
- Generates both individual and aggregate forecasts
- Provides visualization of historical data and forecasts
- Supports S3 integration

## Output Visualization

The script generates several plots:
- Aggregate forecast
- Historical data
- Forward-looking projections
- Combined history and forecast visualization

## Notes

- Confidence score is a composite of mean absolute percentage and ratio of observations to full year
- The script automatically estimates full last month based on daily average
- Includes error handling for forecast generation
