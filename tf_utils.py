import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from datetime import timedelta
from sklearn.cross_validation import train_test_split
import sys
from math import log

indices=['datetime','symbol','categ','val','ind58','ind59','ind60','ind61','ind63','ind64','ind66','ind67','ind68','ind69','ind70','ind71','ind72','ind73','ind74','ind75','ind76','ind77','ind78','ind79','ind80','ind81','ind82','ind83','ind84','ind85','ind87','ind88','ind89','ind93','ind94','ind96','ind97','ind98','ind99','ind100','ind102','ind106','ind108','ind110','ind111','ind112','ind113','ind114','ind115','ind116','ind117','ind118','ind119','ind120','ind121','ind122','ind123','ind125','ind126','ind127','ind128','ind130','ind131','ind132','ind133','ind134','ind135','ind136']

SYMBOL_ALL = 15
keys = ['M1','M3','M5','H1','H4','D1']
query_types={
'M1': timedelta(minutes=1),
'M3': timedelta(minutes=3),
'M5': timedelta(minutes=5),
'H1': timedelta(hours=1),
'H4': timedelta(hours=4),
'D1': timedelta(days=1)
}

def structure_data(data, history, query_type, predict=False ):
    rnn_dfx = []
    rnn_dfy = []

    minutes= 0 if predict else ( query_type.seconds // 60)
    i=history+minutes-1
    print('query_type',minutes,'i',i,'len(data)',len(data))
    while i<len(data):
        j = i - minutes
        # while(data[i][0]-data[j][0]>timedelta(minutes=minutes)):
        #     j+=1
        print('i',i,'j',j)
        args = []

        for y in range(j - history+1,j+1):
            # Adding other symbols as features
            args+=data[ y , 1 ]
        # Adding indices as features
        for z in range(SYMBOL_ALL):
            args+=data[ j , 2][z][1:]
        print('args',len(args))
        rnn_dfx.append( [[i] for i in args] )
        rnn_dfy.append( [ [ data[i][1][s] for s in range(SYMBOL_ALL)] ] )
        i+=1

    print( 'rnn_dfx.shape', np.array(rnn_dfx).shape )
    print( 'rnn_dfy.shape', np.array(rnn_dfy).shape )
    return np.array(rnn_dfx).astype(np.float32, copy=False),np.array(rnn_dfy).astype(np.float32, copy=False)

def get_data_point(last_time):
    return [last_time,[0]*SYMBOL_ALL,[[0]*(len(indices)-4) for x in range(SYMBOL_ALL)]]

def convert(data, history, query_type, predict=False):
    res=[]
    i=0
    last_time=data[0][0]
    # format:
    # old: [timestamp,[15 vals for each symbol], {'M1': [indi]+ , ... }  ]
    # new: [timestamp,[15 vals for each symbol], [[indi for 'M1']*15] ]
    data_point= get_data_point(last_time)
    while i<len(data):
        if(data[i][0]!=last_time):
            # print('data_point',data_point)
            res.append(data_point)
            last_time=data[i][0]
            data_point=get_data_point(last_time)
        s = data[i][1]
        c = data[i][2]
        if c=='M1':
            data_point[1][s]=data[i][3]
            data_point[2][s]=data[i][4:].tolist()
        # print('cccc',s,symbol,c)
        for x in data_point[2]:
            # print('data_point[2][x]',data_point[2][x])
            if len(res)>0 and sum(data_point[2][x])==0:
                data_point[2][x]=res[-1][2][x]
        i+=1
    # for b in res:
    #     print(b)
    return np.array(res)


def prepare_data(df, history=0 , query_type=timedelta(minutes=1), predict=False ):
    df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
    df=df[indices]
    print(df.columns)
    print(len(df.columns))
    data = df.values
    print( len(data) )
    data = convert(data,history,query_type,predict=predict)
    train_x, train_y = structure_data(data, history, query_type, predict=predict )
    return dict(train=train_x), dict(train=train_y)

 
def parse_config(filename):
    options = {}
    COMMENT_CHAR = '#'
    OPTION_CHAR =  '='
    f = open(filename)
    for line in f:
        # First, remove comments:
        if COMMENT_CHAR in line:
            # split on comment char, keep only the part before
            line, comment = line.split(COMMENT_CHAR, 1)
        # Second, find lines with an option=value:
        if OPTION_CHAR in line:
            # split on option char:
            option, value = line.split(OPTION_CHAR, 1)
            # strip spaces:
            option = option.strip()
            value = value.strip()
            # store in dictionary:
            options[option] = value
    f.close()
    return options

def network(intput_nodes,output_nodes,checkpt_dir):
    net = tflearn.input_data([None, intput_nodes])
    net = tflearn.fully_connected(net, 64, activation='linear',
                                     regularizer='L2', weight_decay=0.0005)
    # net = tflearn.embedding(net, input_dim=100, output_dim=128)
    # net = tflearn.lstm(net, 128, dropout=0.1)
    net = tflearn.fully_connected(net, output_nodes, activation='linear')
    net = tflearn.regression(net, optimizer=
        tflearn.optimizers.AdaGrad(learning_rate=0.01, initial_accumulator_value=0.01), 
        loss='mean_square', learning_rate=0.05)

    model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path=checkpt_dir);

    return model
