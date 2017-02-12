#Series-prediction using LSTMs in Tensorflow

## ML Methods

I have chosen a lstm based model for the problem.

It network is basically 
stacked lstm cells
an optional dense layers
 (read lstm_predictor.py)

## Usage

The required codebase is present in user tensorflow’s ~/codes/repo directory.

There are only files:
lstm_for_vf.py : main file
lstm_predictor.py: contains heling utils for main


Commands:
python lstm_for_vf.py test : runs the script in test mode.
python lstm_for_vf.py init : initializes the machine learning models
python lstm_for_vf.py update : updates the models by adding new rows which came after last_timestamp  
python lstm_for_vf.py [categ] [symbol] :  returns the desired prediction for categ & symbol.




## Initial Data Analysis of Data 
I assumed your data will be some kind of related to stocks data. It basically can be modelled as series prediction task. But here we have additional columns to help the predictions. 

I have removed initially some columns of low/NIL variance. The indices array is thus used to select a subset of rows to be used in training.

I have also observed that the table v1 actually contains 15 (symbols-(0,14)) different series, not just one. Rows of a symbol does have no correlation with any other symbol’s row. Similarly for each query type different models have been trained. One for M1, one for M3, …..

## Models

For each model’s, model data folders are generated in models/ directory.

Query will be of type query(symbol,categ): categ specified how far in future are we looking for prediction.  

Currently query is being done differently from requested.

Query(1,’H1’): will predict the current ‘val’ for symbol 1 using only the rows which were available on or before 1 hr. This way is better for visualization and comparision. I will change it later. 


For machine learning part, I have mainly used learn.TensorFlowEstimator()  module of  tensorflow contrib.

The method initialize(update=False) initializes and trains all the models. [ The limit parameter decides the amount of data from v1 used in training. Currently for quick running, I have chosen a low limit for it] . While calling initialize(update=True) just adds the newly added data to learned models 

## Parameters

HISTORY = 1 # history used for lstm.
TIMESTEPS = (len(indices)-3)*HISTORY 
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 1000
BATCH_SIZE = 1
PRINT_STEPS = TRAINING_STEPS / 10

Making HISTORY large improves the accuracy but affects the running time.
Similarly choosing more number of dense layers is helpful when used with large data(limit)


-----------------------------------------------------------------------------

To learn more about LSTMs, go to
http://www.slideshare.net/TaegyunJeon1/electricity-price-forecasting-with-recurrent-neural-networks. Its very helpful.




References

https://github.com/googledatalab/notebooks/blob/master/samples/TensorFlow%20Machine%20Learning%20with%20Financial%20Data%20on%20Google%20Cloud%20Platform.ipynb
http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
https://cloudplatform.googleblog.com/2016/03/TensorFlow-machine-learning-with-financial-data-on-Google-Cloud-Platform.html
https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series
https://github.com/tencia/stocks_rnn
https://github.com/mouradmourafiq/tensorflow-lstm-regression
https://github.com/aymericdamien/TensorFlow-Examples
