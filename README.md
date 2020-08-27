# Data-Viz-ML-DL-with-stock-prices

Here's some scripts who aim to predict stock price. 

In those script you can specify any company you want (if she's in the dataset, they are 501 so you could try some if you want) and after that, just run the script and you'll obtain graph in output to see how well our models perform !


Here's some examples :

## Basic predictions with Facebook Prophet (stock_price_prophet.py)

![prophet_pred](https://user-images.githubusercontent.com/49553009/91452495-ee508880-e87e-11ea-94c9-c6a419a155f3.png)

## Prediction more complexe with Moving Average and relative strength index (stock_price.py)

![stock_price_MA_pred](https://user-images.githubusercontent.com/49553009/91452515-f27ca600-e87e-11ea-86a6-721023350390.png)

## Prediction with LSTM (stock_price_LSTM.py)

![lstm_pred](https://user-images.githubusercontent.com/49553009/91452488-ebee2e80-e87e-11ea-8983-d46358a3c767.png)

## Prediction with GRU (stock_price_gru.py)

![gru_pred](https://user-images.githubusercontent.com/49553009/91452478-e8f33e00-e87e-11ea-9405-cbfc837e902d.png)




I also implemented a script to visualise those stock price in Dash, mcuh easier to explain for POC or other stuff with this kind of interactive data visualization, here's some examples :

![dash_graph_stock](https://user-images.githubusercontent.com/49553009/91452458-e2fd5d00-e87e-11ea-8292-96399b69ef2f.png)

![dash_graph_stock_2](https://user-images.githubusercontent.com/49553009/91452467-e690e400-e87e-11ea-962e-e3697f017188.png)


You just need to specify the name of the company and here you go !


