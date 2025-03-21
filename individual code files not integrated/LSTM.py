from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = Flask(__name__)

def get_historical_stock_data(ticker: str, years: int = 5, interval: str = "1d"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    return df

def create_stock_graph(df, y_column, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[y_column], mode='lines', name=title, line=dict(color=color), hoverinfo='x+y'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=f'{title} (USD)', template='plotly_dark', hovermode='x unified')
    return fig.to_html(full_html=False)

def plot_moving_averages(df, ticker, short_window=50, long_window=200):
    df['SMA'] = df['Close'].rolling(window=short_window).mean()
    df['LMA'] = df['Close'].rolling(window=long_window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price', line=dict(color='#FF5733'), hoverinfo='x+y'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA'], mode='lines', name=f'{short_window}-Day SMA', line=dict(color='#33FFCE', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['LMA'], mode='lines', name=f'{long_window}-Day LMA', line=dict(color='#FFA500', dash='dot')))
    fig.update_layout(title=f'{ticker} Moving Averages Trend', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark', hovermode='x unified')
    return fig.to_html(full_html=False)

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    seq_length = 60
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(100, return_sequences=False, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, X_test, scaler, prediction_days):
    future_input = X_test[-1].reshape(1, X_test.shape[1], 1)
    future_predictions = []
    for _ in range(prediction_days):
        pred = model.predict(future_input)
        future_predictions.append(pred[0, 0])
        pred_reshaped = np.reshape(pred, (1, 1, 1))
        future_input = np.append(future_input[:, 1:, :], pred_reshaped, axis=1)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        prediction_days = int(request.form['prediction_days'])
        df = get_historical_stock_data(ticker)
        closing_graph = create_stock_graph(df, 'Close', 'Closing Price Trend', '#FF5733')
        opening_graph = create_stock_graph(df, 'Open', 'Opening Price Trend', '#33FFCE')
        high_graph = create_stock_graph(df, 'High', 'High Price Trend', '#FFA500')
        volume_graph = create_stock_graph(df, 'Volume', 'Volume Trend', '#FF00FF')
        moving_avg_graph = plot_moving_averages(df, ticker)

        X, y, scaler = prepare_lstm_data(df)
        X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
        y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
        model = build_lstm_model(X.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])
        
        future_predictions = predict_future_prices(model, X_test, scaler, prediction_days)
        dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=future_predictions.flatten(), mode='lines', name='Forecasted Price', line=dict(color='#FF5733')))
        fig.update_layout(title=f'{ticker} {prediction_days}-Day Stock Price Forecast', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
        forecast_graph = fig.to_html(full_html=False)
        
        forecast_df = pd.DataFrame({'Date': dates, 'Predicted Price': future_predictions.flatten()})
        return render_template('LSTM.html', closing_graph=closing_graph, opening_graph=opening_graph, high_graph=high_graph,
                               volume_graph=volume_graph, moving_avg_graph=moving_avg_graph, forecast_graph=forecast_graph,
                               forecast_table=forecast_df.to_html(classes='table table-dark'))
    return render_template('LSTM.html', closing_graph=None, opening_graph=None, high_graph=None, volume_graph=None,
                           moving_avg_graph=None, forecast_graph=None, forecast_table=None)

if __name__ == '__main__':
    app.run(debug=True)



