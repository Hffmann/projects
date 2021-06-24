import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model, Sequential, load_model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from DARNN.attention_layers import *

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

import argparse
import time

from DARNN.layer_definition import My_Transpose,My_Dot,Expand

from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def parse_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=128, type=int, help="batch size")
    parser.add_argument("--epochs", default=100, type=int, help="epochs")
    parser.add_argument("--test-split", default=0.2, type=float, help="test split")
    parser.add_argument("--seq-len", default=10, type=int, help="sequence length")
    parser.add_argument("--multivariate-len", default=81, type=int, help="multivariate input length")
    parser.add_argument("--m-len", default=64, type=int, help="m hidden state length")
    parser.add_argument("--p-len", default=64, type=int, help="p hidden state length")
    parser.add_argument("--random-state", default=0, type=int, help="random state")
    parser.add_argument("--shuffle", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether to shuffle the training data before each epoch")
    parser.add_argument("--n-outputs", default=1, type=int, help="number of outputs")
    parser.add_argument("--model-type", default='DA-RNN', type=str,
                        help="which model to instantiate")
    parser.add_argument("--window-size", default=0, type=int,
                        help="whether or not to train the model")
    parser.add_argument("--num-units", default=0, type=int,
                        help="whether or not to train the model")
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not to train the model")
    parser.add_argument("--do-eval", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    parser.add_argument("--do-test", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    parser.add_argument("--do-genetic-training", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not performing genetic training optimization")
    parser.add_argument("--do-specific-training", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not performing specific training optimization")
    parser.add_argument("--dataset_type", default='multivariate', type=str, help="dataset type")
    parser.add_argument("--data_type", default='NASDAQ100', type=str, help="data type")
    parser.add_argument("--data_path", default='nasdaq100/small/nasdaq100_padding.csv', type=str, help="data path")
    parser.add_argument("--model-path", default='weights/DA-RNN_0_0_128xrelu.h5', type=str, help="model path")

    return parser.parse_args()

args = vars(parse_args())

def get_data(data_type=args['data_type'], data_path=args['data_path']):

    if data_type == 'wind power':

        data = pd.read_csv(data_path)
        data = np.reshape(np.array(data['wp1']),(len(data['wp1']),1))

        # Use first 17,257 points as training/validation and rest of the 1500 points as test set.
        train_data = data[0:17257]
        test_data = data[17257:]

        return train_data, test_data

def prepare_dataset(data=None, data_path=args['data_path'], data_type=args['data_type'], model_type=args['model_type'], window_size=args['window_size'], test_split=args['test_split'], random_state=args['random_state'], seq_len=args['seq_len'], m_len=args['m_len'], p_len=args['p_len']):

    if data_type == 'wind power':

        X, Y = np.empty((0,window_size)), np.empty((0))
        for i in range(len(data)-window_size-1):
            X = np.vstack([X,data[i:(i + window_size),0]])
            Y = np.append(Y,data[i + window_size,0])
        X = np.reshape(X,(len(X),window_size,1))
        Y = np.reshape(Y,(len(Y),1))

        X_train, X_val, Y_train, Y_val = split(X, Y, test_size=test_split, random_state=random_state)

        return X, Y, X_train, X_val, Y_train, Y_val

    if data_type == 'NASDAQ100':

        if model_type == 'DA-RNN':

            T = seq_len
            n = args['multivariate_len']
            m =n_h = n_s = m_len  #length of hidden state m
            p = n_hde0 = n_sde0 = p_len #p

            input_X = []
            input_Y = []
            label_Y = []

            n_outputs = args['n_outputs']

            data = pd.read_csv(data_path)
            row_length = len(data)
            column_length = data.columns.size
            for i in range(row_length-T+1-n_outputs):
                # X_data = data.iloc[i:i+T, 0:column_length-1]
                # Y_data = data.iloc[i:i+T-n_outputs,column_length-1]
                # label_data = data.iloc[i+T-n_outputs:i+T,column_length-1]
                X_data = data.iloc[i:i+T, 0:column_length-1]
                Y_data = data.iloc[i:i+T,column_length-1]
                label_data = data.iloc[i+T:i+T+n_outputs,column_length-1]
                input_X.append(np.array(X_data))
                input_Y.append(np.array(Y_data))
                label_Y.append(np.array(label_data))
            input_X = np.array(input_X).reshape(-1,T,n)
            input_Y = np.array(input_Y).reshape(-1,T,1)
            label_Y = np.array(label_Y).reshape(-1, n_outputs)

            print(input_Y[0],label_Y[0])

            if args['do_test']:
                shuffle = False
            else:
                shuffle = True

            input_X_train, input_X_test, input_Y_train, input_Y_test, label_Y_train, label_Y_test = split(input_X, input_Y, label_Y, test_size=test_split, random_state=random_state, shuffle=shuffle)
            print('input_X_train shape:',input_X_train.shape)
            print('input_X_test shape:', input_X_test.shape)
            print('input_Y_train shape:', input_Y_train.shape)
            print('input_Y_test shape:',input_Y_test.shape)
            print('label_Y_train shape:', label_Y_train.shape)
            print('label_Y_test shape:', label_Y_test.shape)

            s0_train = h0_train = np.zeros((input_X_train.shape[0], m))
            h_de0_train = s_de0_train =np.zeros((input_X_train.shape[0], p))
            s0_test = h0_test = np.zeros((input_X_test.shape[0],m))
            h_de0_test = s_de0_test =np.zeros((input_X_test.shape[0],p))

            return [input_X_train, input_Y_train, s0_train, h0_train, s_de0_train, h_de0_train], [input_X_test,input_Y_test,s0_test,h0_test,s_de0_test,h_de0_test], label_Y_train, label_Y_test

        # elif model_type == 'VAE':
        #
        #     T = args['seq_len']
        #     n = args['multivariate_len']
        #
        #     data = pd.read_csv(data_path)
        #
        #     preprocess(df)
        #     df = df.reshape(-1,T,n)
        #
        #     return df

def genetic_training(population_size='4', num_generations='4', gene_length='10'):

    best_values = []
    # As we are trying to minimize the RMSE score, that's why using -1.0.
    # In case, when you want to maximize accuracy for instance, use 1.0
    creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
    creator.create('Individual', list , fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary,
    n = gene_length)
    toolbox.register('population', tools.initRepeat, list , toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', train_evaluate)

    population = toolbox.population(n = population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)

    # Print top N solutions - (1st only, for now)
    best_individuals = tools.selBest(population,k = 1)
    best_seq_len = None
    best_m_len = None
    best_p_len = None

    for bi in best_individuals:
        seq_len_bits = BitArray(bi[0:6])
        m_len_bits = BitArray(bi[6:12])
        p_len_bits = BitArray(bi[12:])
        best_seq_len = seq_len_bits.uint
        best_m_len = m_len_bits.uint
        best_p_len = p_len_bits.uint

        best_seq_len = best_seq_len + args['n_outputs']
        best_m_len = best_m_len + 1
        best_p_len = best_p_len + 1

        print('\nSequence Length: ', best_seq_len, ', m length: ', best_m_len, ', p length: ', best_p_len)

    best_values.append(best_seq_len)
    best_values.append(best_m_len)
    best_values.append(best_p_len)

    return best_values

def get_model(num_units=None, input_X_train=None, window_size=None, relu=False, dataset_type=args['dataset_type'], state_type='stateful', model_type=args['model_type'], seq_len=args['seq_len'], m_len=args['m_len'], p_len=args['p_len']):

    if dataset_type is 'univariate':

        print(model_type, type(model_type))

        if model_type == 'LSTM_1':
            model = Sequential()
            model.add(LSTM(num_units, input_shape=(window_size,1)))
            model.add(Dense(1))
            model.compile(optimizer='adam',loss='mean_squared_error')

        elif model_type == 'LSTM_2':
            model = Sequential()
            model.add(LSTM(num_units, activation='relu', input_shape=(window_size,1)))
            model.add(Dense(1))
            model.compile(optimizer='adam',loss='mean_squared_error')

        elif model_type == 'LSTM_stacked':
            model = Sequential()
            model.add(LSTM(num_units, activation='relu', input_shape=(window_size,1)))
            model.add(LSTM(num_units, activation='relu', input_shape=(window_size,1)))
            model.add(Dense(1))
            model.compile(optimizer='adam',loss='mean_squared_error')

        elif model_type == 'LSTM_denseStacked':
            model = Sequential()
            model.add(LSTM(num_units, activation='relu', input_shape=(window_size,1)))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(6, activation='relu'))
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam',loss='mean_squared_error')

    if dataset_type is 'multivariate':

        if model_type == 'DA-RNN':

            T = seq_len
            n = args['multivariate_len']
            m =n_h = n_s = m_len  #length of hidden state m
            p = n_hde0 = n_sde0 = p_len #p
            n_outputs = args['n_outputs']

            # en_densor_We = Dense(T)
            # en_LSTM_cell = LSTM(n_h,return_state=True)
            # de_LSTM_cell = LSTM(p,return_state=True)
            # de_densor_We = Dense(m)
            # LSTM_cell = LSTM(p,return_state=True)

            X = Input(shape=(T,n))   #输入时间序列数据
            s0 = Input(shape=(n_s,))  #initialize the first cell state
            h0 = Input(shape=(n_h,))   #initialize the first hidden state
            h_de0 = Input(shape=(n_hde0,))
            s_de0 = Input(shape=(n_sde0,))
            Y = Input(shape=(T,1))

            X_ = encoder_attention(T,n,X,s0,h0,n_h,p,m,relu)
            print('X_:',X_)
            X_ = Reshape((T,n))(X_)
            print('X_:',X_)
            h_en_all = LSTM(m,return_sequences=True)(X_)
            h_en_all = Reshape((T,-1))(h_en_all)
            print('h_en_all:',h_en_all)

            h,context = decoder_attention(T,h_en_all,Y,s_de0,h_de0,p,m,n_outputs)
            h = Reshape((1,p))(h)
            concat = Concatenate(axis=2)([h,context])
            concat = Reshape((-1,))(concat)
            print('concat:',concat)
            result = Dense(p)(concat)
            print('result:',result)
            output = Dense(n_outputs)(result)

            s0_train = h0_train = np.zeros((input_X_train.shape[0],m))
            h_de0_train = s_de0_train =np.zeros((input_X_train.shape[0],p))
            model = Model(inputs=[X,Y,s0,h0,s_de0,h_de0],outputs=output)
            model.compile(loss='mse',optimizer='adam',metrics=['mse'])

        # if model_type == 'VAE':
        #
        #     T = args['seq_len']
        #     n = args['multivariate_len']
        #
        #     model = LSTM_Var_Autoencoder(intermediate_dim = 15,z_dim = 3, n_dim=n, stateful = True) #default stateful = False

    print(model.summary())

    return model

def custom_learning_rate(epoch, lrate):
    if epoch < (10000 * int(math.ceil(epoch/10000))):
        return lrate
    else:
        return lrate * 0.1

def fit_model(model=None, X_train=None, Y_train=None, train_data=None, epochs=args['epochs'], batch_size=args['batch_size'], shuffle=args['shuffle'], validation_split=0):

    # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    # validation_split=0.0, validation_data=None, shuffle=True,
    # lass_weight=None, sample_weight=None, initial_epoch=0,
    # steps_per_epoch=None, validation_steps=None, validation_freq=1,
    # ax_queue_size=10, workers=1, use_multiprocessing=False)

    # early_stopping = EarlyStopping(monitor='val_mean_squared_error',patience=20,mode='min')
    # model_checkpoint = ModelCheckpoint(fname_model,monitor='val_mean_squared_error',verbose=0)
    # lrs = LearningRateScheduler(custom_learning_rate)
    # rlrop = ReduceLROnPlateau(monitor='mse', factor=0.1, patience=100)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_split=validation_split)
    # else:
    #     history = model.fit(df, learning_rate=0.001, batch_size = 100, num_epochs = 200, opt = tf.train.AdamOptimizer, REG_LAMBDA = 0.01,
    #         grad_clip_norm=10, optimizer_params=None, verbose = True)

    return history


    # # list all data in history
    # print(history.history.keys())
    #
    # print(type(history))
    # # summarize history for accuracy
    # if 'mse' in history.history.keys():
    #     print('True')
    #     plt.plot(history.history['mse'][1:])
    # if 'val_mse' in history.history.keys():
    #     plt.plot(history.history['val_mse'][1:])
    # plt.title('model mse')
    # plt.ylabel('mse')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # if 'loss' in history.history.keys():
    #     print('True')
    #     plt.plot(history.history['loss'][1:])
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'][1:])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

def evaluate_model(model, X_test, Y_test, save_name='', model_type=args['model_type'], num_units=args['num_units'], window_size=args['window_size'], mode='Test', save=False, seq_len=args['seq_len'], m_len=args['m_len'], p_len=args['p_len']):

    score = model.evaluate(X_test,Y_test,batch_size=X_test[0].shape[0],verbose=1)

    if save and mode is 'Test':
        model.save("weights/{}_{}_{}_{}_{}_{}proto_less1000.h5".format(model_type, seq_len, m_len, p_len, args['n_outputs'], save_name))
    if save and mode is 'Validation':
        model.save("temp_weights/{}_{}_{}_{}_{}_{}proto_less1000.h5".format(model_type, seq_len, m_len, p_len, args['n_outputs'], save_name))

    return score

def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units
    print(len(ga_individual_solution))

    m_len_bits = BitArray(ga_individual_solution[0:6])
    p_len_bits = BitArray(ga_individual_solution[6:12])
    seq_len_bits = BitArray(ga_individual_solution[12:])

    seq_len = seq_len_bits.uint
    m_len = m_len_bits.uint
    p_len = p_len_bits.uint

    seq_len = seq_len + args['n_outputs']
    m_len = m_len + 1
    p_len = p_len + 1

    print('\nSequence Length: ', seq_len, ', m length: ', m_len, ', p length: ', p_len)

    # # Return fitness score of 100 if window_size or num_unit is zero
    # if seq_len == 0 or m_len == 0 or p_len == 0:
    #     return 100,

    X_train, X_val, Y_train, Y_val = prepare_dataset(seq_len=seq_len, m_len=m_len, p_len=p_len)

    model = get_model(input_X_train=X_train[0], seq_len=seq_len, m_len=m_len, p_len=p_len)

    fit_model(model=model, X_train=X_train, Y_train=Y_train, validation_split=0.2)

    score = evaluate_model(model, X_val, Y_val, mode='Validation', save=True, seq_len=seq_len, m_len=m_len, p_len=p_len)
    return score

if args['do_train']:

    np.random.seed(args['random_state'])

    # train_data, test_data = get_data()

    if args['do_genetic_training']:

        population_size = 6
        num_generations = 6
        gene_length = 16

        best_values = genetic_training(population_size, num_generations, gene_length)
        best_seq_len = best_values[0]
        best_m_len = best_values[1]
        best_p_len = best_values[2]

        X_train, X_val, Y_train, Y_val = prepare_dataset(seq_len=best_seq_len, m_len=best_m_len, p_len=best_p_len)

        model = get_model(input_X_train=X_train[0], seq_len=best_seq_len, m_len=best_m_len, p_len=best_p_len)

        fit_model(model=model, X_train=X_train, Y_train=Y_train)
        # model.fit(X_train, Y_train, epochs=5, batch_size=10,shuffle=True)

        rmse = evaluate_model(model, X_val, Y_val, mode='Validation', save=True)

    if args['do_specific_training']:

        if args['dataset_type'] == 'univariate':
            window_size = args['window_size']
            num_units = args['num_units']
            print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)

        # if args['model_type'] != 'VAE':

            if args['data_type'] == 'wind power':
                X_train, Y_train, _, _, _, _ = prepare_dataset(train_data, window_size=window_size)
                X_test, Y_test, _, _, _, _ = prepare_dataset(test_data, window_size=window_size)
                model = get_model(num_units, window_size)
            # elif args['data_type'] == 'NASDAQ100':
            #     X_train, X_test, Y_train, Y_test = prepare_dataset()
            #     model = get_model(input_X_train=X_train[0])
            #
            # fit_model(model=model, X_train=X_train, Y_train=Y_train)

        # else:

        if args['data_type'] == 'NASDAQ100':

            if not args['do_genetic_training']:
                X_train, X_test, Y_train, Y_test = prepare_dataset()
                model = get_model(input_X_train=X_train[0])

            else:
                X_train, X_test, Y_train, Y_test = prepare_dataset(seq_len=best_seq_len, m_len=best_m_len, p_len=best_p_len)
                model = get_model(input_X_train=X_train[0], seq_len=best_seq_len, m_len=best_m_len, p_len=best_p_len)

        history = fit_model(model=model, X_train=X_train, Y_train=Y_train)
        score = evaluate_model(model, X_test, Y_test, mode='Test', save=True)

        print('loss:',score[0])
        print('mse:',score[1])
        print('rmse:',np.sqrt(score[1]))

        # list all data in history
        print(history.history.keys())

        # summarize history for accuracy
        if 'mse' in history.history.keys():
            plt.plot(history.history['mse'][20:])

        # X_train, X_test, Y_train, Y_test = prepare_dataset(seq_len=31, m_len=46, p_len=10)
        # model = get_model(input_X_train=X_train[0], seq_len=31, m_len=46, p_len=10)
        #
        # history = fit_model(model=model, X_train=X_train, Y_train=Y_train)
        # score = evaluate_model(model, X_test, Y_test, mode='Test', save=True)
        #
        # print('loss:',score[0])
        # print('mse:',score[1])
        # print('rmse:',np.sqrt(score[1]))
        #
        # # list all data in history
        # print(history.history.keys())
        #
        # # summarize history for accuracy
        # if 'mse' in history.history.keys():
        #     plt.plot(history.history['mse'][20:])
        #
        # X_train, X_test, Y_train, Y_test = prepare_dataset(seq_len=10, m_len=64, p_len=64)
        # model = get_model(input_X_train=X_train[0], seq_len=10, m_len=64, p_len=64)
        #
        # history = fit_model(model=model, X_train=X_train, Y_train=Y_train)
        # score = evaluate_model(model, X_test, Y_test, mode='Test', save=True)
        #
        # print('loss:',score[0])
        # print('mse:',score[1])
        # print('rmse:',np.sqrt(score[1]))
        #
        # # list all data in history
        # print(history.history.keys())
        #
        # # summarize history for accuracy
        # if 'mse' in history.history.keys():
        #     plt.plot(history.history['mse'][20:])

        plt.title('model mse')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['64'], loc='upper left')
        plt.show()


        # #TEST
        # model = get_model(input_X_train=X_train[0], relu=True)
        #
        # fit_model(model=model, X_train=X_train, Y_train=Y_train)
        # score = evaluate_model(model, X_test, Y_test, save_name="2", mode='Test', save=True)
        #
        # print('loss:',score[0])
        # print('mse:',score[1])
        # print('rmse:',np.sqrt(score[1]))
        # #TEST

if args['do_eval'] and not args['do_train']:

    if args['data_type'] == 'wind power':
        X_train, Y_train, _, _, _, _ = prepare_dataset(train_data, window_size=window_size)
        X_test, Y_test, _, _, _, _ = prepare_dataset(test_data, window_size=window_size)
        # model = get_model(num_units, window_size)
    elif args['data_type'] == 'NASDAQ100':
        X_train, X_test, Y_train, Y_test = prepare_dataset()
        # model = get_model(input_X_train=X_train[0])

    model = load_model(args['model_path'], custom_objects={'Expand': Expand, 'My_Dot': My_Dot, 'My_Transpose': My_Transpose})
    print(model.summary())

    start = time.perf_counter()
    # Y_pred = model.predict(np.array([X_test[3],]))
    score = evaluate_model(model, X_test, Y_test, mode='Test', save=False)
    end = time.perf_counter()

    print('loss:',score[0])
    print('mse:',score[1])
    print('rmse:',np.sqrt(score[1]))

    print(f"Prediction took {end - start:0.4f} seconds")

if args['do_test']:

    if not args['do_train'] and not args['do_eval']:

        if args['data_type'] == 'wind power':
            X_train, Y_train, _, _, _, _ = prepare_dataset(train_data, window_size=window_size)
            X_test, Y_test, _, _, _, _ = prepare_dataset(test_data, window_size=window_size)
        elif args['data_type'] == 'NASDAQ100':
            _, X_test, _, Y_test = prepare_dataset()

        model = load_model(args['model_path'], custom_objects={'Expand': Expand, 'My_Dot': My_Dot, 'My_Transpose': My_Transpose})
        print(model.summary())

    # Y_pred = model.predict(X_test)
    #
    # X = np.array([X_test[0][0],X_test[0][1],X_test[0][2],X_test[0][3],X_test[0][4],X_test[0][5]]).reshape(6,10,81).tolist()
    # X.reshape((1, X.shape[2], X.shape[1]))
    # X = [np.array([[X_test[0][i],],]) for i in range(6)]
    # print(X_test.shape)

    seq_len = args['seq_len']
    test_range = 500
    test_values = X_test[1][args['seq_len']:500]
    pred_values = []
    test_values = []
    n_values = []

    for i in range(0, test_range, seq_len):

        if i != 0:

            X = [np.array([[*X_test[j][i-seq_len][int(len(X_test[j][i-seq_len])/2):].tolist(),
             *X_test[j][i][:int(len(X_test[j][i])/2)].tolist()]]) for j in range(6) ]

            Y_test2 = X_test[1][i][int(len(X_test[1][i])/2):]

        else:
            X = [np.array([X_test[j][0]]) for j in range(6)]
            Y_test2 = Y_test[0]

        start = time.perf_counter()
        Y_pred = model.predict(X)[0]
        end = time.perf_counter()
        # print(type(X_test), type(X_test[0]), type(X_test[0][0]), type(X_test[0][0][0]), X[0].shape)

        pred_values.extend(Y_pred)
        test_values.extend(Y_test2)

        # print(X_test[1][i], Y_pred, Y_test2)

        print(f"Prediction took {end - start:0.4f} seconds")

    plt.figure()
    plt.title('Predicted values x Test values')
    plt.xlabel('t')
    plt.ylabel('NDX')
    plt.plot(pred_values)
    plt.plot(test_values)
    plt.legend(['pred', 'test'], loc='upper right')
    plt.show()





    # k = 150
    # n_values = []
    #
    # X = [np.array([[*X_test[j][k-seq_len][int(len(X_test[j][k-seq_len])/2):].tolist(),
    #  *X_test[j][k][:int(len(X_test[j][k])/2)].tolist()]]) for j in range(6) ]
    #
    # Y_test2 = X_test[1][k][int(len(X_test[1][k])/2):]
    #
    # start = time.perf_counter()
    # Y_pred = model.predict(X)[0]
    # end = time.perf_counter()
    #
    # n_values = [[X[0][0][i][j] for i in range(len(X[0][0]))] for j in range(len(X[0][0][0]))]
    #
    # print(f"Prediction took {end - start:0.4f} seconds")
    #
    # plt.figure()
    # plt.title('Multivariate values')
    # plt.xlabel('t')
    # plt.ylabel('class')
    # for i in range(len(X_test[0][150][0])):
    #     plt.plot(n_values[i])
    # plt.show()
