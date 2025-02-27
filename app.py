from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    return df

def calculate_ema(df):
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Buy/Sell signals
    df['Signal'] = np.where(df['EMA_100'] > df['EMA_200'], 1, 0)
    df['Buy_Signal'] = (df['Signal'].shift(1) == 0) & (df['Signal'] == 1)
    df['Sell_Signal'] = (df['Signal'].shift(1) == 1) & (df['Signal'] == 0)

    return df

def plot_stock(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label="Actual Price", color='blue')
    plt.plot(df.index, df['EMA_100'], label="100 EMA", color='orange')
    plt.plot(df.index, df['EMA_200'], label="200 EMA", color='red')

    # Plot Buy/Sell signals
    plt.scatter(df.index[df['Buy_Signal']], df['Close'][df['Buy_Signal']], marker='^', color='green', label='Buy Signal', alpha=1)
    plt.scatter(df.index[df['Sell_Signal']], df['Close'][df['Sell_Signal']], marker='v', color='red', label='Sell Signal', alpha=1)

    plt.title(f"{ticker} Stock Price Prediction with EMA Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    
    # Save the figure in memory and encode it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        try:
            df = fetch_stock_data(ticker)
            df = calculate_ema(df)
            plot_url = plot_stock(df, ticker)
            return render_template("result.html", plot_url=plot_url, ticker=ticker)
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
