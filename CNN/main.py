
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import keras

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

# This is a sample Python script.

main_map = np.load("map_data\\full_map_main.npy")
main_map_1cm = main_map[::12, :]

#######################################
# Config
#######################################
equivalent_uniform_distance_cm = 4
portion_size = 46

training_examples = 50
validation_examples = 20
test_examples = 20

number_filters_layer_1 = 256
number_filters_layer_2 = 128
number_filters_layer_3 = 1

kernel_size_layer_1 = 9
kernel_size_layer_2 = 1
kernel_size_layer_3 = 5

number_of_epochs = 400

########################################
# Scaling the map to 0 and 1
########################################

main_map_max = np.max(main_map_1cm)
main_map_min = np.min(main_map_1cm)

main_map_1cm_scaled = (main_map_1cm - main_map_min) / (main_map_max - main_map_min)


#########################################
# Random sample selection from map
#########################################

total_samples = main_map_1cm_scaled.shape[0] * main_map_1cm_scaled.shape[1]
number_random_samples = int(np.ceil(total_samples / equivalent_uniform_distance_cm**2))

random_sample_counter = 0
random_samples = np.empty((number_random_samples, 3))
random_samples[:] = np.nan

while random_sample_counter < number_random_samples:

    if random_sample_counter == 0:
        random_x = 0
        random_y = 0
    elif random_sample_counter == 1:
        random_x = 0
        random_y = main_map_1cm_scaled.shape[0]-1
    elif random_sample_counter == 2:
        random_x = main_map_1cm_scaled.shape[1]-1
        random_y = 0
    elif random_sample_counter == 3:
        random_x = main_map_1cm_scaled.shape[1]-1
        random_y = main_map_1cm_scaled.shape[0]-1

    else:

        random_x = np.random.randint(0, main_map_1cm_scaled.shape[1])
        random_y = np.random.randint(0, main_map_1cm_scaled.shape[0])

    # find entries with same y
    portion = random_samples[random_samples[:, 1] == random_y, :]
    # check if x is already in there
    if sum(portion[:, 0] == random_x) == 0:

        random_samples[random_sample_counter, 0] = random_x
        random_samples[random_sample_counter, 1] = random_y
        random_samples[random_sample_counter, 2] = main_map_1cm_scaled[random_y, random_x]
        random_sample_counter += 1


###########################################
# Nearest-Neighbor Interpolation for input
###########################################

grid_x, grid_y = np.mgrid[0:main_map_1cm_scaled.shape[1], 0:main_map_1cm_scaled.shape[0]]
points = random_samples[:, 0:2]
values = random_samples[:, 2]

main_map_random_NN = griddata(points, values, (grid_x, grid_y), method='nearest').T

'''
plt.imshow(main_map_random_NN.T, extent=(0, 1, 0, 1), origin='lower')
plt.title('Nearest')
plt.show()
'''

##########################################
# Cut map into equally sized parts
##########################################

map_examples = np.empty((int(np.floor(main_map_1cm_scaled.shape[1] / portion_size) *
                             np.floor(main_map_1cm_scaled.shape[0] / portion_size)),
                         portion_size, portion_size))

map_examples_true = np.empty((int(np.floor(main_map_1cm_scaled.shape[1] / portion_size) *
                                  np.floor(main_map_1cm_scaled.shape[0] / portion_size)),
                              portion_size, portion_size))

sort_counter = 0
for col_index in range(int(np.floor(main_map_1cm_scaled.shape[1] / portion_size))):

    xs_to_read = np.arange(col_index*portion_size, (col_index+1)*portion_size)

    for row_index in range(int(np.floor(main_map_1cm_scaled.shape[0] / portion_size))):
        ys_to_read = np.arange(row_index * portion_size, (row_index + 1) * portion_size)

        extraction_NN = main_map_random_NN[ys_to_read[0]:ys_to_read[-1]+1, xs_to_read[0]:xs_to_read[-1]+1]
        extraction_true = main_map_1cm_scaled[ys_to_read[0]:ys_to_read[-1]+1, xs_to_read[0]:xs_to_read[-1]+1]
        map_examples[sort_counter, :, :] = extraction_NN
        map_examples_true[sort_counter, :, :] = extraction_true

        sort_counter += 1


######################################################
# Shuffle the data set and get train, val and test set
######################################################

shuffled_indices = np.random.permutation(np.arange(0, map_examples.shape[0]))

train_set_x = map_examples[shuffled_indices[0:training_examples], :, :]
val_set_x = map_examples[shuffled_indices[training_examples:training_examples + validation_examples], :, :]
test_set_x = map_examples[shuffled_indices[training_examples + validation_examples:
                                           training_examples + validation_examples + test_examples], :, :]

train_set_y = map_examples_true[shuffled_indices[0:training_examples], :, :]
val_set_y = map_examples_true[shuffled_indices[training_examples:training_examples + validation_examples], :, :]
test_set_y = map_examples_true[shuffled_indices[training_examples + validation_examples:
                                                training_examples + validation_examples + test_examples], :, :]

train_set_x = train_set_x.reshape(train_set_x.shape[0], portion_size, portion_size, 1)
val_set_x = val_set_x.reshape(val_set_x.shape[0], portion_size, portion_size, 1)
test_set_x = test_set_x.reshape(test_set_x.shape[0], portion_size, portion_size, 1)

train_set_y = train_set_y.reshape(train_set_y.shape[0], portion_size, portion_size, 1)
val_set_y = val_set_y.reshape(val_set_y.shape[0], portion_size, portion_size, 1)
test_set_y = test_set_y.reshape(test_set_y.shape[0], portion_size, portion_size, 1)

input_shape = (portion_size, portion_size, 1)

#################################################
# Defining the Neural Network
#################################################


model = Sequential()
model.add(Conv2D(number_filters_layer_1, kernel_size=(kernel_size_layer_1, kernel_size_layer_1), strides=(1, 1),
                 activation='relu',
                 padding='same',
                 input_shape=input_shape))
model.add(Conv2D(number_filters_layer_2, (kernel_size_layer_2, kernel_size_layer_2), strides=(1, 1), activation='relu',
                 padding='same'))
model.add(Conv2D(number_filters_layer_3, (kernel_size_layer_3, kernel_size_layer_3), strides=(1, 1), activation='relu',
                 padding='same'))

model.summary()

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['mae', 'mse'])

trainedModelPath = "C:\\Users\\friedrich.burmeister\\PyCharmProjects\\CNN_REM_Interpolation\\TrainedModels\\"
my_callbacks = [keras.callbacks.ModelCheckpoint(
    filepath=trainedModelPath + 'best_model.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True),
    keras.callbacks.TensorBoard(log_dir='./logs'),
    #keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    #keras.callbacks.ModelCheckpoint(filepath=trainedModelPath + 'model.{epoch:02d}-{val_loss:.2f}.h5'),

]

history = model.fit(train_set_x, train_set_y,
                    batch_size=training_examples,
                    epochs=number_of_epochs,
                    verbose=1,
                    validation_data=(val_set_x, val_set_y),
                    validation_batch_size=validation_examples,
                    callbacks=my_callbacks)

score = model.evaluate(test_set_x, test_set_y, verbose=1)

print('Test loss:', score[0])

# Accuracy is only important for classification tasks and has no value for regressions
# --> Consider loss as important metric

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

###############################################################################
# Make predictions on test set, convert to original value range and compute MAE
###############################################################################
# load best model for this iteration
best_model = keras.models.load_model("TrainedModels/best_model.h5")

scaled_errors = np.empty((portion_size**2, test_set_x.shape[0]))
rescaled_errors = np.empty((portion_size**2, test_set_x.shape[0]))

for test_example_index in range(test_set_x.shape[0]):

    test_x = test_set_x[test_example_index, :, :, :]
    test_x = test_x.reshape(1, test_x.shape[0], test_x.shape[1], test_x.shape[2])
    true = test_set_y[test_example_index, :, :, :]
    prediction = best_model.predict(test_x)

    # get rid of unneeded array dimensions
    prediction = np.squeeze(prediction)
    true = np.squeeze(true)

    scaled_error_array = prediction - true
    scaled_errors[:, test_example_index] = np.reshape(scaled_error_array, (portion_size**2, ))

    prediction_back_scaled = prediction * (main_map_max - main_map_min) + main_map_min
    true_back_scaled = true * (main_map_max - main_map_min) + main_map_min

    rescaled_error_array = prediction_back_scaled - true_back_scaled
    rescaled_errors[:, test_example_index] = np.reshape(rescaled_error_array, (portion_size ** 2, ))


# MAE of scaled error
test_mae_scaled = np.mean(abs(scaled_errors))
test_mae_rescaled = np.mean(abs(rescaled_errors))

print('Manual Test MAE (scaled):', test_mae_scaled)
print('Manual Test MAE:', test_mae_rescaled)

test = 0