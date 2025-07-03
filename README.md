# Crypto Market Data Analysis & Prediction

A Python script that analyzes cryptocurrency market data, predicts prices using statistical models, and visualizes results.

## Description

This project fetches cryptocurrency market data from an API, processes the data using statistical methods and machine learning models (e.g., linear regression), and generates visualizations of the analysis. The results are stored in a MySQL database.

Key features:
- Fetches real-time cryptocurrency data
- Performs statistical analysis (normality tests, skewness, returns)
- Generates technical indicators (EMA, SMA, RSI)
- Predicts future prices using logarithmic regression
- Visualizes price data and predictions
- Stores processed data in a MySQL database
- Performs future price prediction using LSTM model

## Prerequisites

Before running this script, ensure you have the following:

1. **Python 3.6+ installed**
2. Required libraries:
   - NumPy
   - Pandas
   - Matplotlib
   - Scipy
   - Statsmodels
   - Requests

3. Access to a MySQL database or configuration for one

4. API keys (if required for data fetching)

## Installation

1. Clone this repository:
   

`bash`  
   git clone https://github.com/ericwkim1/crypto_market_analysis.git
   cd crypto_market_analysis.git
   



2. Install dependencies:
   

`bash`  
   pip install numpy pandas matplotlib scipy statsmodels requests
   



## Configuration

1. Create a `.env` file in the root directory with the following variables:
   


   MYSQL_HOST=your_database_host  
   MYSQL_USER=your_database_user  
   MYSQL_PASSWORD=your_database_password  
   MYSQL_DB=your_database_name  
   



2. Set your API keys in the environment or modify the script accordingly.

## Usage

1. Run the script:
   

`bash`  
   python crypto_data_analysis.py
   



2. The script will:
   - Fetch cryptocurrency data
   - Process and analyze the data
   - Generate visualizations
   - Store results in the MySQL database

## Data Processing

The script processes the following metrics for each cryptocurrency:
- Moving averages (SMA, EMA)
- Relative Strength Index (RSI)
- Normality tests
- Skewness
- Returns analysis (daily and weekly)
- Price predictions with prediction intervals

## Visualization

The script generates plots that include:
- Actual prices vs predicted prices
- Technical indicators (EMA, SMA)
- Prediction intervals
- Key statistical metrics

Plots are saved in the `graphs` directory with timestamps and symbols for organization.

## Directory Structure


crypto_market_analysis/  
├── main.py               # Main script  
├── README.md             # This README file  
└── graphs/              # Directory to save generated plots  



## Notes

- **API Limits**: Be mindful of API rate limits when fetching data.
- **Data Freshness**: The script fetches fresh data periodically; adjust timing as needed.
- **Database Connection**: Ensure database credentials are correctly configured.
- **Error Handling**: The script includes basic error handling, but refine it based on your needs. 
