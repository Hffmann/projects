import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model, Sequential, load_model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

import argparse
import time

def parse_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", default=0, type=int,
                        help="whether or not to train the model")
    parser.add_argument("--num_units", default=0, type=int,
                        help="whether or not to train the model")
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not to train the model")
    parser.add_argument("--do-eval", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    parser.add_argument("--do-test", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    return parser.parse_args()

args = vars(parse_args())

def prepare_dataset(data, window_size):
    X, Y = np.empty((0,window_size)), np.empty((0))
    for i in range(len(data)-window_size-1):
        X = np.vstack([X,data[i:(i + window_size),0]])
        Y = np.append(Y,data[i + window_size,0])
    X = np.reshape(X,(len(X),window_size,1))
    Y = np.reshape(Y,(len(Y),1))
    return X, Y

def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:6])
    num_units_bits = BitArray(ga_individual_solution[6:])
    window_size = window_size_bits.uint
    num_units = num_units_bits.uint
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)

    # Return fitness score of 100 if window_size or num_unit is zero
    if window_size == 0 or num_units == 0:
        return 100,

    # Segment the train_data based on new window_size; split into train and validation (80/20)
    X,Y = prepare_dataset(train_data,window_size)
    X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)


    ''' Model 1
    model = Sequential()
    model.add(LSTM(num_units, input_shape=(window_size,1)))
    model.add(Dense(1, activation='linear'))
    '''

    # ''' Model 2
    model = Sequential()
    model.add(LSTM(num_units, activation='relu', input_shape=(window_size,1)))
    model.add(Dense(1, activation='linear'))
    # '''

    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=10,shuffle=True)
    y_pred = model.predict(X_val)

    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print('Validation RMSE: ', rmse,'\n')

    model.save("weights_lstmrelu/model_{}_{}.h5".format(num_units, window_size))

    return rmse,

if args['do_train']:

    np.random.seed(1120)

    data = pd.read_csv('train.csv')
    data = np.reshape(np.array(data['wp1']),(len(data['wp1']),1))

    # Use first 17,257 points as training/validation and rest of the 1500 points as test set.
    train_data = data[0:17257]
    test_data = data[17257:]

    if args['window_size'] is 0 and args['num_units'] is 0:

        population_size = 4
        num_generations = 4
        gene_length = 10

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
        r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1,
        ngen = num_generations, verbose = False)

        # Print top N solutions - (1st only, for now)
        best_individuals = tools.selBest(population,k = 1)
        best_window_size = None
        best_num_units = None

        for bi in best_individuals:
            window_size_bits = BitArray(bi[0:6])
            num_units_bits = BitArray(bi[6:])
            best_window_size = window_size_bits.uint
            best_num_units = num_units_bits.uint
            print('\nWindow Size: ', best_window_size, ', Num of Units: ', best_num_units)

        # Train the model using best configuration on complete training set
        #and make predictions on the test set
        X_train,y_train = prepare_dataset(train_data,best_window_size)
        X_test, y_test = prepare_dataset(test_data,best_window_size)

        model = Sequential()
        model.add(LSTM(best_num_units, activation='relu', input_shape=(best_window_size,1)))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=10,shuffle=True)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print('Test RMSE: ', rmse)

        # save model and architecture to single file
        model.save("weights_lstmrelu/model.h5")
        print("Saved model to disk")

    else:

        window_size = args['window_size']
        num_units = args['num_units']
        print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)

        X_train, y_train = prepare_dataset(train_data, window_size)
        X_test, y_test = prepare_dataset(test_data, window_size)

        model = Sequential()
        model.add(LSTM(num_units, activation='relu', input_shape=(window_size,1)))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=10,shuffle=True)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print('Test RMSE: ', rmse)

        # save model and architecture to single file
        model.save("weights/model_specificnotlin_{}_{}.h5".format(num_units, window_size))
        print("Saved model to disk")

if args['do_test']:

    if not args['do_train']:
        data = pd.read_csv('train.csv')
        data = np.reshape(np.array(data['wp1']),(len(data['wp1']),1))
        X_test, y_test = prepare_dataset(data[17257:],49)

    model = load_model("weights/model.h5")
    start = time.perf_counter()
    y_pred = model.predict(np.array([X_test[3],]))
    end = time.perf_counter()
    print(X_test[3])
    print(y_pred, y_test[3])
    print(f"Prediction took {end - start:0.4f} seconds")
