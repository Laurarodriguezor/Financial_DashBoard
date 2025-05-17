Stock Market Dashboard
An interactive stock market dashboard built with Streamlit and Plotly, providing users with visualizations and metrics to analyze stock performance.
Features

Interactive stock price visualization with moving averages and Bollinger bands
Key performance metrics organized by category:

Return metrics: Total return, annualized return, daily returns
Risk metrics: Volatility, maximum drawdown, Value at Risk
Technical indicators: RSI, MACD, Moving averages


Daily returns analysis with multiple visualization formats:

Color-coded data table
Calendar heatmap
Bar chart


Historical data analysis with customizable date ranges

Setup

Clone this repository
Install requirements: pip install -r requirements.txt
Run the app: streamlit run Home.py

Pages

Home: Overview with stock price, key metrics, and return distribution
Daily Returns: Detailed analysis of daily returns with multiple visualizations

Data Source
This dashboard uses data from Yahoo Finance via the yfinance package. Date ranges should preferably be set to historical periods (before 2024) for most reliable data.
Requirements

Python 3.8+
Internet connection (for fetching stock data)
Web browser (for viewing the dashboard)

Design Considerations
This dashboard was designed to address specific requirements:

Providing essential metrics for understanding stock performance
Organizing metrics by category for easy comprehension
Using both cards and tables to display information effectively
Employing color coding to indicate performance (green/red)
Creating a data table with the most relevant columns
Using appropriate formatting and conditional styling
