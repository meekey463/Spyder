#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 00:12:48 2020

@author: meekey
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from flask import Flask, render_template
from flask import Flask, request, make_response, send_file
import io
import csv
import pandas_datareader.data as pdr
from scipy import stats
from flasgger import Swagger
import os


app = Flask(__name__)
swagger = Swagger(app)

@app.route('/price', methods = ['POST'])

def stock_price():
    """Endpoint returning stock price
    ---
    parameters:
      - name: ticker
        in: query
        type: string
        required: true
    responses:
        500:
            description: Error
        200:
            description: Successful
        
    """
    stocks = request.args.get("ticker")
    print(stocks)
#    output_file('tech_ind.html', title = 'Stock Technical Indicators')
    data = pd.DataFrame()
#    stocks = pd.read_csv(request.files.get("input_file"))
#    stocks = stocks.Symbol.tolist()
    start_date = "2019-01-01"
    # train_end_dt = dt.datetime.today()- dt.timedelta(days=1)
    end_date = dt.datetime.today()
    # test_date = dt.datetime.today()
    ohlcv = pdr.get_data_yahoo(stocks, start_date, end_date)
    return str(ohlcv['Adj Close'])

#if __name__ == "__main__":
#        app.run()
#
#    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
