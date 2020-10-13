# DataViz | Machine Learning | Deep Learning with stock prices

Here's some scripts who aim to predict stock price. This kind of work is really useful in term of business value. Stock prices are really volatile, hence very hard to predict so it's very appreciated for companies if you know the tricks. Moreover it's a very powerful way to train yourself because like I said previously, stock prices are strongly volatile and it's quite impossible to predict perfectly.

In those script you can specify any company you want (if she's in the dataset, they are 501 so you could try some if you want) and after that, just run the script and you'll obtain graph in output to see how well our models perform !


Here's some examples :

## Basic predictions with Facebook Prophet (stock_price_prophet.py)

![prophet_pred](https://user-images.githubusercontent.com/49553009/91452495-ee508880-e87e-11ea-94c9-c6a419a155f3.png)

## Prediction more complexe using xgboost with Moving Average, relative strength index and datetime feature (stock_price_xgboost.py)

![stock_price_MA_pred](https://user-images.githubusercontent.com/49553009/91452515-f27ca600-e87e-11ea-86a6-721023350390.png)

**Metrics

> R² = 0.71

> MAE = 3.044

> MSE = 16.16

> RMSE = 4.02

## Prediction more complexe using AutoARIMAX with Moving Average, relative strength index and datetime feature (stock_price_AUTOARIMAX.py)

![autoarimax](https://user-images.githubusercontent.com/49553009/95889301-87315980-0d82-11eb-8e86-9c5409bcc69f.png)

**Metrics**

> R² = 0.80

> MAE = 2.61

> MSE = 11.17

> RMSE = 3.34

## Prediction with LSTM (stock_price_LSTM.py)

![lstm](https://user-images.githubusercontent.com/49553009/95762150-9db9b100-0cad-11eb-8265-01b5105ffe71.png)

**Metrics**

> R² = 0.99

> MAE = 2.42

> MSE = 9.62

> RMSE = 3.102


## Prediction with GRU (stock_price_gru.py)

![téléchargement](https://user-images.githubusercontent.com/49553009/95761014-043dcf80-0cac-11eb-9552-5a30cd57d9ad.png)

**Metrics**

> R² = 0.98

> MAE = 2.65

> MSE = 10.51

> RMSE = 3.24




I also implemented a script to visualise those stock price in Dash, mcuh easier to explain for POC or other stuff with this kind of interactive data visualization, here's some examples :

![dash_graph_stock](https://user-images.githubusercontent.com/49553009/91452458-e2fd5d00-e87e-11ea-8292-96399b69ef2f.png)

![dash_graph_stock_2](https://user-images.githubusercontent.com/49553009/91452467-e690e400-e87e-11ea-962e-e3697f017188.png)


You just need to specify the name of the company and here you go !


