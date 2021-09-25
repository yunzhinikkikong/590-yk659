###################################################
#####Code ANN regression using KERAS, train on the Boston housing datase
########################################################################


# Load Boston hoursing dataset
import numpy as np
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# Normalize the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# Build the model, use a very small network with two hidden layers, each with 64 units

from keras import models
from keras import layers
from keras import regularizers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l1(0.001),activation='tanh',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l1(0.001),activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Validating the approach using K-fold validation

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []


for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [train_data[:i * num_val_samples],
    train_data[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]],
    axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


print(f'all_scores : {all_scores}')
print(f'mean all scores : {np.mean(all_scores)}')

num_epochs = 500
all_mae_histories = []
train_loss_histories = []
val_loss_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
         [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]],axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
              validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    all_mae_histories.append(mae_history)
    train_loss_histories.append(loss_train)
    val_loss_histories.append(loss_val)


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

train_loss_history = [np.mean([x[i] for x in train_loss_histories]) for i in range(num_epochs)]
val_loss_history = [np.mean([x[i] for x in val_loss_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

model = build_model()
model.fit(train_data, train_targets,
epochs=60, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

test_mae_score



fig, ax = plt.subplots()
ax.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'o', label='Training loss')
ax.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'o', label='Validation loss')
plt.xlabel('epochs', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.legend()
plt.show()