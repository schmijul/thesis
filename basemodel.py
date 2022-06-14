import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd


def create_callback(trainedModelPath, EarlyStopping=False,verbose=False):
    
        """
        quick function to create callbacks and or overwrite existing callbacks
        
        """
        
        callbacks=  [
            tf.keras.callbacks.ModelCheckpoint(
                trainedModelPath+"/best_model.h5", 
                save_best_only=True, 
                monitor="val_loss", 
                verbose=verbose
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, patience=400, 
                in_lr=0.0001
            )]
        
        if EarlyStopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                              patience=150, 
                                                              verbose=verbose))
            
        return callbacks
    
    
def build_model(verbose=False):

    """

    lazy Code to build Sequential model
    x_train : array of training data to get right input shape
    
    """
    model = Sequential()
    model.add(Dense(1000, input_dim=2,  kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1000, activation='relu'))
                #model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','accuracy'])
    if verbose:
            print(model.summary())


    return model



def train(x_train, y_train, x_val, y_val,length,model, epochs,sampling_distance_x, sampling_distance_y, save_hist=True,  verbose=False, batch_size=100):
    """
        train model 

        x_train : known points ( coordinates)
        y_train : known points ( values)

        x_val : unknown points ( coordinates)
        y_val : unknown points ( values)

        model : Kera Model
        epochs : number of epochs
        batch_size : batch size

        
    """

    trainedModelPath = f'trainedModels/NonRandomResapling/samplerate_x{sampling_distance_x}_y{sampling_distance_y}_length{length}/'
         

    if not os.path.exists(trainedModelPath):
        os.makedirs(trainedModelPath)
                

    
    x_train = x_train.to_numpy()
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

    y_train = y_train.to_numpy()
    y_train = y_train.reshape(y_train.shape[0],1)

    x_val= x_val.to_numpy()
    x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
    y_val= y_val.to_numpy()
    y_val = y_val.reshape(y_val.shape[0],1)

    history = model.fit(x_train
                            , y_train
                            , epochs=epochs
                            , validation_data = (x_val, y_val)
                            , callbacks=create_callback(trainedModelPath, verbose=verbose)
                            , batch_size=batch_size
                            , verbose=verbose
                                )

    if save_hist:
             pd.DataFrame(history.history).to_csv(f'{trainedModelPath}/history_x{sampling_distance_x}_y{sampling_distance_y}.csv')

    return model, trainedModelPath
    
def predict(trainedModelPath, unknown_points, x_val):
    """

        Predict the output of the model on the validation set.
        model : pretrained Kera Model
        x_val : validation set input

    """
    model = tf.keras.models.load_model(trainedModelPath+'/best_model.h5')
    prediction = model.predict(x_val)

    
    pred_stacked = pd.DataFrame([prediction[:,0],unknown_points['x'].to_numpy(),unknown_points['y'].to_numpy()]).transpose()
    pred_stacked.columns = ['z','x','y']

    pred_stacked = pred_stacked[['x','y','z']]
    pred_stacked.index = unknown_points.index
    
    return pred_stacked

