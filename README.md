***To Run This Code***
the file main.ipynb should be simple to run. Ensure you have the necessary software downloaded onto your IDE to use pandas, numpy, matplotlib, and sklearn, then run each cell in order.

For file RNN.ipynb: We were unable to make this code run in our Visual Studio Code libraries, so we opted to used Google Colab. Downloading the file and uploading it to Google Colab should allow you to run the code smoothly, cell by cell.

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

The code starts by reading the original S&P 500 dataset and creating additional columns to capture the information from the previous day, week, and month. These new features include open, high, low, close, volume, change percentage, trend percentage, and volatility. The target variable is the next day's stock price. After calculating these additional features, the code performs data cleaning by dropping rows with missing values to ensure the dataset has no empty features. The modified dataset with the expanded feature set is then saved as a new CSV file.

Before training the machine learning models, the code applies feature scaling using the MinMaxScaler from scikit-learn. This normalization step transforms all the feature values to a common range, typically between 0 and 1. Scaling the features is a crucial preprocessing step for many machine learning algorithms, as it ensures that features with different scales do not unduly influence the model's predictions. The scaled feature matrix is then split into training and testing sets for model evaluation.

The Random Forest Regressor model demonstrated the best overall performance among the three models evaluated. It achieved the lowest Mean Squared Error (MSE) of 223.75, indicating that its predictions had the smallest residual errors compared to the other models. Furthermore, it attained a reasonably high R-squared value of 0.96, suggesting that it could explain 96% of the variance in the target variable (next day's stock price).

The Ridge Regression (Linear Regression) model performed better than the HistGradientBoostingRegressor, with an MSE of 254.62, which is lower than the 298.42 MSE of the HistGradientBoostingRegressor. However, both models exhibited relatively higher errors compared to the Random Forest Regressor. Linear models often struggle to capture complex non-linear relationships, which could explain the inferior performance of the Ridge Regression model in this case.

The HistGradientBoostingRegressor model had the highest MSE of 298.42, indicating larger deviations between its predictions and the actual values. Despite being an ensemble model capable of handling non-linear relationships, its performance was the weakest among the three models for this particular dataset and problem.

We also employed a Recurrent Neural Network (RNN) architecture for predicting the next day's stock prices. This model consisted of two SimpleRNN layers put between a dropout layer for regularization, with the output produced by a dense layer. The close prices were preprocessed, normalized, and transformed into input sequences before training. The RNN model was compiled with the Adam optimizer and mean squared error loss, then trained on the data for 5 epochs. We evaluated its performance by calculating the root mean squared error on both the training and test sets, and the code provided visualizations to compare actual and predicted values. By using an RNN, we aimed to capture temporal dependencies and non-linear relationships in the stock data to potentially improve prediction accuracy over the linear and ensemble models.
