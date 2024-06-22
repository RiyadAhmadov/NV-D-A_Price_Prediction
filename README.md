# NVIDIA Price Prediction

!['NVIDIA Price Prediction'](https://image.cnbcfm.com/api/v1/image/107067261-1653514547587-gettyimages-1234001062-RAFAPRESS_15072021-9654.jpeg?v=1707772252)

This project aims to predict the price of NVIDIA stock using Triple Exponential Smoothing (Holt-Winters). The dataset is scraped from Yahoo Finance using Selenium, and various time series analysis techniques are applied to create a predictive model.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Scraping](#data-scraping)
- [Data Analysis](#data-analysis)
- [Time Series Modeling](#time-series-modeling)
- [Installation](#installation)
- [Results](#results)

## Project Overview
The goal of this project is to predict the future prices of NVIDIA stock. We use historical stock prices scraped from Yahoo Finance and apply time series analysis using the Holt-Winters method for prediction. 

## Data Scraping
We use Selenium to automate the process of fetching historical stock prices from Yahoo Finance. The data is then processed and saved for further analysis.

## Data Analysis
The dataset is analyzed to understand its structure and to visualize the historical price trends. Various statistical techniques are applied to prepare the data for modeling.

## Time Series Modeling
The Holt-Winters method (Triple Exponential Smoothing) is used to create the predictive model. This method captures level, trend, and seasonality components of the time series data.

## Installation
To run this project, you need to have Python installed on your system along with the following libraries:
- pandas
- numpy
- selenium
- matplotlib
- seaborn
- BeautifulSoup
- statsmodels
- scikit-learn

## Results
The results include visualizations of the historical stock prices and the predicted future prices using the Holt-Winters method. The performance of the model is evaluated using Mean Absolute Error (MAE).
