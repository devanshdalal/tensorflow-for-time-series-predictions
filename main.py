import sys
import math,os
import datetime
from datetime import timedelta
from threading import Thread
from copy import deepcopy

import pymysql as mariadb

import numpy as np
import pandas as pd
from pandas.tseries.offsets import *

import tensorflow as tf
import tflearn
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tf_utils import prepare_data,parse_config,network
import tf_utils

indices=tf_utils.indices
query_types=tf_utils.query_types


################################## configurable model_params #################################################

options = parse_config('config.txt')
LIMIT = int(options['LIMIT']) if 'LIMIT' in options else 1000
HISTORY = int(options['HISTORY']) if 'HISTORY' in options else 1
DENSE_LAYERS = int(options['DENSE_LAYERS']) if 'DENSE_LAYERS' in options else 1
TRAINING_STEPS = int(options['TRAINING_STEPS']) if 'TRAINING_STEPS' in options else 1000
BATCH_SIZE = int(options['BATCH_SIZE']) if 'BATCH_SIZE' in options else 1
USER = options['USER_PASS']
USER_PASS = options['USER_PASS']

# non-configurable params

DIMENSION = 2*len(indices)+8
TIMESTEPS = DIMENSION*HISTORY 
RNN_LAYERS = [{'steps': TIMESTEPS}]
PRINT_STEPS = TRAINING_STEPS / 10
MODEL_FOLDER = 'model/'


# symbols are in range(0,15)
last_timestamp=''

def record_timestamp():
	conn = mariadb.connect(user=USER, passwd=USER_PASS, db='modeldata')
	df = pd.read_sql_query('select datetime from v1 order by datetime desc LIMIT 1', conn)
	last_timestamp =  str(df.ix[0][0])
	print(last_timestamp,file=open("timestamp.txt","w"))

def restore_timestamp():
	return open("timestamp.txt").read().strip()


def generate_query(update_str):
	return 'SELECT * FROM ( '\
			'SELECT * from v1 '\
			 ''+update_str + ' ORDER BY datetime DESC ) sub '\
		'ORDER BY datetime ASC'

def initialize(incremental_learning=False):

	def process_categ(categ):
		print('process_categ running with args:', categ)
		train_model(categ=categ, incremental_learning=incremental_learning)

	threads_list={}
	for categ in query_types:
		process_categ(categ)
		# t = Thread(None,process_categ,None,(categ,))
		# t.start()
		# threads_list[t]=True		

	for th in threads_list:
		th.join()
	record_timestamp()

def train_model(categ,incremental_learning=False):

	MODEL_DIR = MODEL_FOLDER + categ

	update_str = ''	

	conn = mariadb.connect(user='freelancer', passwd='freelancer16', db='modeldata')
	if incremental_learning:
		last = datetime.datetime.strptime(restore_timestamp(), "%Y-%m-%d %H:%M:%S")
		last = str(last - HISTORY - query_types[categ].seconds//60 )
		print(last)
		update_str=' where datetime>\''+last+ '\' ' 
	else:
		t = pd.read_sql_query('SELECT * from (select DISTINCT datetime from v1 ORDER BY datetime DESC LIMIT '
			+str(LIMIT+1)+') sub ORDER BY datetime limit 1', conn).values[0,0]
		print('t',t,str(t))
		# t=datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f000")
		update_str = ' where datetime>\''+ str(t - np.timedelta64(minutes=LIMIT)) + '\' '
		print(update_str) 
		os.system('rm -rf '+ MODEL_DIR)
	
	query=generate_query(update_str)
	print('query',query)

	df = pd.read_sql_query(query, conn)

	print('Preparing data')

	X,y = prepare_data(df, history=HISTORY, query_type=query_types[categ] )

	print('X[0].shape[0]',X[0].shape[0])

	regressor = network(X[0].shape[0],SYMBOL_ALL,MODEL_DIR)

	regressor.fit(X['train'], y['train'] )

	# Printing the outcome for last 5 data points. 

	X['test']=X['train'][-5:]
	y['test']=y['train'][-5:]
	predicted = regressor.predict(X['test'])
	print('actual', 'predicted')
	for i,yi in enumerate(predicted):
		yi=np.array(yi).reshape(-1)
		y['test'][i]=np.array(y['test'][i]).reshape(-1)
		print(y['test'][i],' ',yi)

	mse = mean_squared_error(y['test'], predicted)
	print ("mean_squared_error : %f" % mse)
	###############################
	regressor.save( MODEL_DIR+'mdl.pkl' )

def predict(categ='M1',visualization=True):
	"""
	Creates a prediction:
	    * categ
	:param visualization: if true, the predicted 'val' are compared with existing 'val'
	                     if false, the actual query will be answered for future time(categ minutes later).
	:return: predicted 'val'

	:description: this does not train any model, it restores the directory after prediction is made.
	            : the fit() is just called to pass the mode_dir to regressor.
	"""
	MODEL_DIR = MODEL_FOLDER +categ

	conn = mariadb.connect(user='freelancer', passwd='freelancer16', db='modeldata')
	minutes = 0
	if visualization:
		minutes = ( query_types[categ].seconds // 60)
	limit=minutes+HISTORY+1

	t = pd.read_sql_query('SELECT * from v1 ORDER BY datetime DESC LIMIT 1', conn).values[0,0]
	update_str = ' where datetime>\''+ str(t - np.timedelta64(minutes=limit)) + '\' '
	print(update_str) 
	query=generate_query(update_str)

	df = pd.read_sql_query(query, conn)

	X, y = prepare_data(df, history=HISTORY, query_type=query_types[categ]  )

	print('XX ', X['train'].shape)
	
	regressor = network(X[0].shape[0],SYMBOL_ALL,MODEL_DIR)
	regressor.save( MODEL_DIR+'mdl.pkl' )

	X['test']=X['train'][-1:]
	y['test']=np.arrat(y['train'][-1]).reshape(-1);

	predicted = regressor.predict(X['test'])
	predicted = np.array(predicted[0]).reshape(-1)

	print('actual', 'predicted')
	if visualization:
		print('actual', 'predicted')
		for i,yi in enumerate(predicted):
			print(y['test'][i],' ',yi)

		mse = mean_squared_error(y['test'], price)
		print ("mean_absolute_error : %f" % mse)
	
	return predicted

###############################################################################################

assert( len(sys.argv)>=1 )

if sys.argv[1]=='test':
# testing mode: it will not change any model
	MODEL_FOLDER='tmp_model/'
	train_model(categ='M3')
	predict(categ='M3')
	os.system('rm -rf '+ MODEL_FOLDER + "*")

# if this is an initialization.
elif sys.argv[1]=='init':
	initialize( incremental_learning=False )

# if this is an incremental update.
elif sys.argv[1]=='update':

	initialize( incremental_learning=True )

# if this is a query
elif len(sys.argv)>=2 and (sys.argv[1] in query_types) and int(sys.argv[2]) in range(0,15) :
	symbol=int(sys.argv[2])
	category=sys.argv[1]
	result = predict(symbol=symbol,categ=category,visualization=False)
	print('Prediction',result)

else:
	raise ValueError('Bad command line input')


