# Tesla Stock Price Prediction

This project demonstrates the use of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), for predicting future stock prices of Tesla Inc. Stock price prediction is a complex task due to the highly volatile and dynamic nature of stock markets. The aim here is to develop a model that captures trends in historical data and provides reasonable forecasts for future stock prices.

## Table of Contents
- [Overview](#overview)
- [Objective](#objective)
- [Concepts Used](#concepts-used)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [API Usage](#api-usage)
- [Running the Project](#running-the-project)
- [Conclusion](#conclusion)

## Overview

Stock price prediction is a time-series forecasting problem that involves analyzing patterns in historical stock prices to predict future movements. This project implements a deep learning approach using LSTM, a powerful architecture designed for learning long-term dependencies in sequential data. The predictions are further served via an API using FastAPI, allowing for seamless interaction with the model.
 
## Objective

The main objective of this project is to predict the future stock prices of Tesla Inc. using historical data. By utilizing the `Close` prices over the past 60 days, the model attempts to forecast the next closing price. The model is integrated into a FastAPI application, where users can input historical data and receive stock price predictions in real-time.

## Concepts Used

### 1. **Time Series Forecasting**
   Time series data is a sequence of data points collected at consistent time intervals. Stock prices are inherently sequential, and predicting them involves using historical values to forecast future prices. LSTM networks are particularly suited for this task because they can remember long-term dependencies in data.

### 2. **Recurrent Neural Networks (RNN)**
   RNNs are designed to recognize patterns in sequential data by maintaining a hidden state that captures past information. However, traditional RNNs struggle with long-term dependencies due to vanishing gradients, which led to the development of LSTMs.

### 3. **Long Short-Term Memory (LSTM)**
   LSTMs are a type of RNN that can learn long-term dependencies in data by using memory cells and gates that control information flow. This architecture is ideal for time series problems like stock price prediction, as it helps the model focus on relevant information over long sequences of input data.

## Requirements

The following packages are required to run the project:
- Python 3.12
- FastAPI
- Uvicorn
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Joblib

You can install the necessary dependencies by creating a `requirements.txt` file:


## Dataset
The dataset contains historical stock prices for Tesla and includes the following features:

Date
Open
High
Low
Close
Adjusted Close
Volume
You can obtain the dataset from sources like Yahoo Finance or Kaggle.

## Model Architecture

This LSTM model is designed for time-series forecasting, specifically for predicting stock prices. The first two layers are LSTM layers, which capture sequential patterns from 60 timesteps of stock price data. The first LSTM layer has 50 units and returns the full sequence, while the second LSTM layer with 64 units outputs only the final state, summarizing the sequence into one representation.

The model then passes this information through Dense layers with 32 and 16 neurons, reducing the data's dimensionality to focus on key patterns. The final Dense layer outputs a single predicted value, representing the next stock price. The model uses the 'adam' optimizer and 'mean_squared_error' loss function, which are well-suited for regression tasks like stock price prediction.

## Data Preprocessing
Before training the LSTM model, the dataset is preprocessed to ensure the data is suitable for the model:

Scaling: The stock prices are scaled using MinMaxScaler to bring all values into a common range, typically between 0 and 1. This helps in speeding up the convergence of the LSTM model.

Splitting: The dataset is split into training and testing sets, and the last 60 values of the Close price are used to predict the next price in the sequence

## API Usage
The model is deployed as a FastAPI service, allowing users to send requests to predict future stock prices based on the past 60 days of Close prices.
FastApi_StocksTesla (Request).png shows a sample request
response_sample shows a sample response

## Running the Project
Clone the repository:
```
git clone https://github.com/CrimsonRed89/Tesla-Stocks-Prediction-LSTM
cd <repository-directory>
```

Set up your virtual environment and install requirements:
```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Run the FastAPI application:
```
uvicorn app:app --reload
```


Access the API documentation at http://127.0.0.1:8000/docs to explore the available endpoints.

## Conclusion
This project provides a framework for predicting stock prices using LSTM, a well-suited model for sequential data such as stock prices. The model's performance can be improved by incorporating more features, fine-tuning the architecture, or experimenting with different hyperparameters. The integration of FastAPI allows for real-time predictions through a scalable RESTful service.



