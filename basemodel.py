import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import pandas as pd



def create_callbacks(trainedmodelpath, earlystopping=False,verbose=True):

    """

        _summary_

                Args:
                    trainedmodelpath (str): path where trained model are saved
                    earlystopping (bool): if True, early stopping is used
                    verbose (bool): if True, print information about the training process

        _description_

                    quick function to create callbacks
    """

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
                trainedmodelpath+"/best_model.h5",
                save_best_only=True,
                monitor="val_loss",
                verbose=verbose
            ),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.25, patience=20,
                in_lr=0.01
            )]

    if earlystopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=100,
                                                          verbose=verbose))

    return callbacks


def build_model(num_units, verbose=False):

    """

    lazy Code to build Sequential model
    This model uses coordinates (x,y) as input and z ( value at coordinates) as output

    """
    model = Sequential()
    model.add(Dense(num_units, input_dim=2,  kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_units, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_units, activation='relu'))
                #model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    if verbose:
        print(model.summary())


    return model



def train(x_train, y_train, x_val, y_val,
          model, epochs, scenario,
          save_hist=False,  verbose=False, batch_size=100):
    """
    _summary_
        Args:
                x_train (pandas DataFrame) : known points ( coordinates)
                y_train (pandas DataFrame): known points ( values)

                x_val (pandas DataFrame) : unknown points ( coordinates)
                y_val (pandas DataFrame) : unknown points ( values)

                model (keras model) : precompiled model
                epochs (int) : number of epochs
                scenario (str) : name of the scenario (for trainedmodelpath)

                save_hist (bool): if True, save history of training
                verbose (bool): if True, print information about the training process
    """

    trainedmodelpath = f'trainedModels/baseModel/{scenario}'

    if not os.path.exists(trainedmodelpath):
        os.makedirs(trainedmodelpath)


    x_train = x_train.to_numpy()
    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1],
                              1)

    y_train = y_train.to_numpy()
    y_train = y_train.reshape(y_train.shape[0],1)

    x_val= x_val.to_numpy()
    x_val = x_val.reshape(x_val.shape[0],
                          x_val.shape[1],
                          1)

    y_val= y_val.to_numpy()
    y_val = y_val.reshape(y_val.shape[0],1)

    history = model.fit(x_train
                            , y_train
                            , epochs=epochs
                            , validation_data = (x_val, y_val)
                            , callbacks=create_callbacks(trainedmodelpath, verbose=verbose)
                            , batch_size=batch_size
                            , verbose=verbose
                                )

    if save_hist:
        pd.DataFrame(history.history).to_csv(f'{trainedmodelpath}/history.csv')

    return model, trainedmodelpath

def predict(trainedmodelpath, x_val):

    """
    _summary_
            Args:
                trainedmodelpath (str): path where trained model are saved
                x_val (pandas DataFrame) : validation points ( coordinates)

    _description_
                This Fct uses the trained model to predict the values at the validation points
                And then also uses the validation points ( coordinates)
                to assign each predicted value to it's coordinate pair
    """
    model = tf.keras.models.load_model(trainedmodelpath+'/best_model.h5')
    prediction = model.predict(x_val)


    pred_stacked = pd.DataFrame([prediction[:,0],
                                 x_val['x'].to_numpy(),
                                 x_val['y'].to_numpy()]).transpose()
    pred_stacked.columns = ['z','x','y']

    pred_stacked = pred_stacked[['x','y','z']]
    pred_stacked.index = x_val.index

    return pred_stacked



if __name__ == '__main__':

    #test if model build is working
    UNITS = 100
    TESTMODEL = build_model(UNITS,verbose=True)
    