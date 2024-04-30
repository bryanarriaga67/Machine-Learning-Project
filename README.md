*** Dataset information ***
    NasDaq_100.csv
This dataset provides the daily Open, High, Low, Close, and Volume (OHLCV) data for the NASDAQ 100 index, which consists of the top 100 non-financial companies listed on the NASDAQ stock exchange.

    QQQ_raw.csv
The dataset encapsulates the daily OHLCV + Additional Metrics data for the QQQ ETF, which tracks the performance of the NASDAQ 100 Index. This raw data offers a non-adjusted view.

    QQQ_split_adj.csv
This dataset offers the split-adjusted daily OHLCV + Additional Metrics data for the QQQ ETF. Adjustments are made to ensure that stock splits don't distort the historical view of the ETF's performance.

    SP500.csv
Contained within is the daily OHLCV + Additional Metrics data for the S&P 500 index. This index represents the performance of 500 of the largest U.S. publicly traded companies.

    SPY.csv
This dataset presents the daily OHLCV + Additional Metrics data for the SPY ETF, which closely tracks the performance of the S&P 500 index.

***Our Approach***
Given the OHLCV of a stock we plan to predict the Closing price for the next day. This script can then be used to predict the next 'n' days. 

We began using linear regression but after seeing the state of our data we found that we would either have to do extensive data cleaning.

We then attemped gradient boosting regressor 
Gradient boosting regressor



Notes from Piplai:
Try:
Random forest regressor OR Time series model
Attempt one neural network- can be from online,, For this the data must be 


DO:
Use TWO datasets for the training and try the feature important scored to see what shows up and if it is consisten

Normalization strategies for ALL our data before and then re-normalize the data
-Min max scaler
-mean scaler

feature important scored- tells you if the features that you added are better than the raw data

***Plotting the data with the actual predicions, then compare to the true Y***