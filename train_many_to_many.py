#!/usr/bin/env python
# coding: utf-8

# ### Training the MANY TO MANY MODEL and save models to directory

# In[2]:


import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime
import os
import glob
import matplotlib as mpl

base_directory = os.getcwd().replace('\\', '/') + '/'

def getDirectory(directory):
    directory = directory.replace(" / ", "_")
    os.chdir(base_directory)
    if not os.path.exists(directory):
            os.makedirs(directory)
    os.chdir('./' + directory)
    
def groupDataframe(data, col='', directory=''):
    getDirectory(directory)
    for index, df in data.groupby(col):
        df['Date'] =  [parse(week) for week in df.Week]
        file_key = 'df_' + str(index).replace(" / ", "_")
        df.name = file_key
        df.to_hdf(file_key + '.h5', key = file_key, mode='w')

def loadDataframe(df_name='', directory=''):
    getDirectory(directory)
    df = pd.read_csv(df_name)
    return df

def plot_comparison(df, col1, col2):
    plt.figure(figsize=(12,5))
    plt.xlabel('Days')

    ax1 = df[col1].plot(marker = 'o', markersize = 4, color = 'blue', grid=True, label=col1)
    ax2 = df[col2].plot(marker = 'x',  markersize=12, color = 'blue', grid=True, label=col2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.legend(h1+h2, l1+l2, loc=2)
    plt.savefig(col1 + '_' + col2 + '.png')
    plt.show()
        
def saveDataframe(df, name_dir='New Folder', name_file='Untitled'):
    getDirectory(name_dir)
    df.to_csv(name_file + '.csv')

def get_file_list(path, ext='csv'):
    result = glob.glob('*.{}'.format(ext))
    return result


# In[3]:


def get_normalization_factor(array):
    try:
        c = len(array)
        s = sum(array)
        normalization_factor = 1/(1+(s/c))
    except ZeroDivisionError:
        normalization_factor = 0
    return normalization_factor
    
def mean_scale_normalize(array):
    normalization_factor = get_normalization_factor(array)
    return np.array(array) * normalization_factor

def inverse_mean_scale_normalize(array, normalization_factor):
    return np.array(array) / normalization_factor

def inverse_mean_scale_normalize_cluster(array, normalization_factor):
    normalization_factor = np.array(normalization_factor)
    normalization_factor.reshape((1,len(normalization_factor)))
    return np.array(array) / normalization_factor

def inverse_normalize_0_1_cluster(array_true, array_pred):
    cols = array_true.shape[1] 
    rows = array_true.shape[0]
    array_pred_rescaled = np.zeros((rows, cols))
    for index in range(cols):
        array_pred_rescaled[:, index] = MinMaxScaler().fit(array_true[:, index].reshape(1, -1)).inverse_transform(array_pred[:, index].reshape(1, -1))
    return array_pred_rescaled


# In[4]:


def moving_window_batch_generator(X_train, Y_train, batch_size=256, input_sequence_length=2, output_sequence_length=2):
    num_X_features = X_train.shape[1]
    num_Y_features = Y_train.shape[1]
    train_size = Y_train.shape[0]
    
    while True:
        X_shape = (batch_size, input_sequence_length, num_X_features)
        X_train_batch = np.zeros(shape=X_shape, dtype=np.float16)
        
        Y_shape = (batch_size, output_sequence_length, num_Y_features)
        Y_tain_batch = np.zeros(shape=Y_shape, dtype=np.float16)

        for i in range(batch_size):
            id = np.random.randint(train_size - input_sequence_length)
            X_train_batch[i] = X_train[id:id+input_sequence_length]
            Y_tain_batch[i] = Y_train[id:id+output_sequence_length]
        
        yield (X_train_batch, Y_tain_batch)


# In[5]:


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

def get_callbacks(patient):

    path_checkpoint ='checkpoint_keras_'+ patient
    log_dir='logs_' +  patient
    
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, 
                                          monitor='val_loss', 
                                          verbose=1, 
                                          save_weights_only=False, 
                                          save_best_only=True, 
                                          mode='auto', 
                                          period=1)
    callback_early_stopping = EarlyStopping(monitor='val_loss', 
                                            patience=50, 
                                            verbose=1)
    callback_tensorboard = TensorBoard(log_dir=log_dir, 
                                       histogram_freq=0, 
                                       write_graph=False)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                           factor=0.1, 
                                           min_lr=1e-4, 
                                           patience=3, 
                                           verbose=1)

    callbacks = [callback_checkpoint, callback_tensorboard, callback_reduce_lr]

    return callbacks



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, GaussianNoise, BatchNormalization, CuDNNLSTM, LeakyReLU, Reshape
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

def get_model(hyperparameters, predictors, targets):

    # Initialising the RNN
    model = Sequential()
    regularizer = l2(0.01)
    optimizer = Adam(lr=hyperparameters['learning_rate'])

    model.add(
        CuDNNLSTM(units = 30,  
                  input_shape=(hyperparameters['input_sequence_length'], len(predictors)),
                  return_sequences = True,
                  kernel_regularizer=regularizer))
    model.add(GaussianNoise(1e-4))
    model.add(BatchNormalization())
    
    model.add(
        CuDNNLSTM(units = 20, 
             return_sequences = True,
             kernel_regularizer=regularizer))
    model.add(GaussianNoise(1e-4))
    model.add(BatchNormalization())
    
    model.add(
        CuDNNLSTM(units = 10,
             kernel_regularizer=regularizer, return_sequences = False))
    model.add(GaussianNoise(1e-4))
    model.add(BatchNormalization())
    
    model.add(Dense(hyperparameters['output_sequence_length'] * len(targets), activation='relu'))
    
    model.add(Reshape((hyperparameters['output_sequence_length'], len(targets))))
    
    model.compile(optimizer = optimizer, loss = 'mean_absolute_error')
   
    return model


# In[6]:


def get_prediction(X, Y, model, normalization_factor, start=0, length=100):
    Y_true = Y
    end = start + length
    X = X[start:end]
    Y_true = Y_true[start:end]
    X = np.expand_dims(X, axis=0)
    Y_pred = model.predict(X)
    Y_pred_rescaled = inverse_mean_scale_normalize(Y_pred.flatten(), normalization_factor)
    return Y_pred_rescaled

def get_prediction_clustered(X, model, normalization_factor=[], start=0, length=100):
    Y_pred = model.predict(X)
    print('--------------------------------------')
    print('Y_pred.shape:', Y_pred.shape)
    print('--------------------------------------')
    Y_pred_rescaled = inverse_mean_scale_normalize_cluster(Y_pred[0], normalization_factor)
    return Y_pred_rescaled
   
def plot_comparison(df, acct, msku):
    plt.figure(figsize=(12,5))
    plt.xlabel('Dates')

    ax1 = df.Y_true.plot(color='blue', grid=True, label='True')
    ax2 = df.Y_pred.plot(color='red', grid=True, label='Predicted')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.legend(h1+h2, l1+l2, loc=2)
    
    title = 'Account: ' + acct + '\n' + 'MSKU: ' + msku.split('.')[0]
    plt.title(title)
    
    plt.savefig(msku.split('.')[0] + '.png')
    plt.show()


# In[7]:


def get_keras_shape(array, time_steps = 1, num_features=1):
    batch_size = array.shape[0]
    return array.reshape((batch_size, time_steps, num_features))

def get_train_pred_set(X_train, hyperparameters, predictors, length = 1):
    X_train_pred_sets = []
    no_of_train_sets = X_train.shape[0]
    window_size = hyperparameters['output_sequence_length']
    start_pred_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(weeks = hyperparameters['output_sequence_length']*length)
    
    while no_of_train_sets < ((length*hyperparameters['output_sequence_length']) + hyperparameters['input_sequence_length']):
        length -= 1
        start_pred_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(weeks = hyperparameters['output_sequence_length']*length)
        
    print('X_train.shape', X_train.shape)

    for index in range(length):
        X_train_pred_sets.append(get_keras_shape(np.array([X_train[- 1 - (index*window_size)]]), 
                                                 time_steps=hyperparameters['input_sequence_length'], 
                                                num_features = len(predictors)))
    return X_train_pred_sets, start_pred_date


# In[8]:


counter = 0


# In[9]:


from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import MinMaxScaler

def addExogenousVariables(df):
    holidays = USFederalHolidayCalendar().holidays(start=df.index.min(), end=df.index.max())
    df['holiday'] = df.index.isin(holidays).astype(int)
    df['week_number'] = df.index.week
    df['day_of_week'] = df.index.dayofweek
    return df

def get_train_val_test(df, predictors, targets, train_size):
    X = df[predictors]
    Y = df[targets]
    
    X_train = X.iloc[0:train_size,:], X.iloc[train_size:,:]
    Y_train = Y.iloc[0:train_size,:], Y.iloc[train_size:,:]
    return X_train, Y_train

def plot_predictions(Y_true, Y_pred, df, acct, targets):
    df_plot = pd.DataFrame(index=df.index)
    cols = Y_true.shape[1]
    
    for index in range(cols):
        df_plot['Y_true'] = Y_true[:, index]
        df_plot['Y_pred'] = Y_pred[:, index]
        plot_comparison(df_plot, acct, targets[index].split('_')[-1] + '.file')

def moving_window_data_formatter_train(X, Y, train_size, hyperparameters):

    X_train, Y_train = [], []
    for i in range(train_size):
        X_start_index = i
        X_end_index = i + hyperparameters['input_sequence_length']
        Y_start_index = i + hyperparameters['input_sequence_length']
        Y_end_index = i + hyperparameters['input_sequence_length'] + hyperparameters['output_sequence_length']
        if Y_end_index <= train_size:
            X_train.append(X[X_start_index:X_end_index, :])
            Y_train.append(Y[Y_start_index:Y_end_index, :])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    return X_train, Y_train


def fit_all_dataframe(patient,counter):

    hyperparameters = {
        'cell_dimension': 30,
        'batch_size': 128,
        'input_sequence_length': 5,
        'output_sequence_length': 1,
        'learning_rate': 1e-2,
        'epochs': 300,
        'gaussian_noise': 1e-4,
        'L2_regularization': 1e-4
    }


    df = pd.DataFrame()
    df_cols = pd.DataFrame(columns = cols)
    
    
    df = loadDataframe(patient, directory='train_data/processed')
    df_cols = df[cols]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])
    df_normalize = pd.DataFrame(data = scaled, columns = cols)
    df_normalize = df_normalize.fillna(value = -1)

    train_size = len(df.index)
    X_train = Y_train = df_normalize.values
    
    X_train, Y_train = moving_window_data_formatter_train(X_train, Y_train, train_size, hyperparameters)

    X_train = get_keras_shape(X_train,
                              time_steps=hyperparameters['input_sequence_length'], 
                              num_features = len(predictors))


    num_X_features = len(predictors)
    num_Y_features = len(targets)
    model = get_model(hyperparameters, predictors, targets)

    getDirectory('models/many_to_many/' + 'keras_'+ 'patient')
    
    if counter==0:
            
        try:
            path_checkpoint = 'checkpoint_keras_' + 'patient'
            model.load_weights(path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)
    
    
    counter = 1
    
    model.fit(X_train, Y_train,
              validation_split=0.25,
              epochs = hyperparameters['epochs'],
              batch_size = hyperparameters['batch_size'],
              callbacks=get_callbacks('patient'))


# In[ ]:


import json
getDirectory('train_data/processed')
predictors = targets = cols = ['X' + str(i) + '_missing_data' for i in range(1, 14)]
patients = glob.glob("*")
for patient in patients:
    print(patient + "\n \n ")
    fit_all_dataframe(patient,counter)

