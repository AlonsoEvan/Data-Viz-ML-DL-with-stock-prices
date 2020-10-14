# DataViz | Machine Learning | Deep Learning with stock prices

Here's some scripts who aim to predict stock price. This kind of work is really useful in term of business value. Stock prices are really volatile, hence very hard to predict so it's very appreciated for companies if you know the tricks. Moreover it's a very powerful way to train yourself because like I said previously, stock prices are strongly volatile and it's quite impossible to predict perfectly.

In those script you can specify any company you want (if she's in the dataset, they are 501 so you could try some if you want) and after that, just run the script and you'll obtain graph in output to see how well our models perform !


Here's some examples :

## Basic predictions with Facebook Prophet (stock_price_prophet.py)

![prophet_pred](https://user-images.githubusercontent.com/49553009/91452495-ee508880-e87e-11ea-94c9-c6a419a155f3.png)

## Prediction more complexe using xgboost with Moving Average, relative strength index and datetime feature (stock_price_xgboost.py)

![stock_price_MA_pred](https://user-images.githubusercontent.com/49553009/91452515-f27ca600-e87e-11ea-86a6-721023350390.png)




![impxgboost](https://user-images.githubusercontent.com/49553009/96010973-b0192380-0e42-11eb-9e9d-7d8cbbe117e4.png)

**Metrics**

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

> MAE = 1.97

> MSE = 6.57

> RMSE = 2.56


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


## Future Development

- Maybe Increasing the number of timesteps : the model remembered the stock prices from the x previous financial days to predict the stock price of the next day. That’s because we chose a number of x timesteps. 

- Adding some other indicators: Maybe stock price of some other companies might be correlated to the one that is used, we could add this other stock price as a new indicator in the training data.
- Adding more layers: I build my Neural Network with four layers but I could try with even more. 

- Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 200 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.

- Getting more training data
