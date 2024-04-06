import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
import matplotlib
from keras.layers import Dropout
import matplotlib.pyplot as plt

# load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# preproc
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# Our vectorized data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = train_labels
y_test = test_labels

# Split train data into train and validation
split_at_idx = 20000
x_train, x_val = x_train[:split_at_idx], x_train[split_at_idx:]
y_train, y_val = y_train[:split_at_idx], y_train[split_at_idx:]
assert len(x_train) == split_at_idx and len(x_val) == len(train_data) - split_at_idx
assert len(y_train) == split_at_idx and len(y_val) == len(train_labels) - split_at_idx

# modeling
model = models.Sequential()
model.add(layers.Dense(32, activation='elu', input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adamax(learning_rate=0.000075), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=39, batch_size=256, validation_data=(x_val, y_val))

# evaluation
results = model.evaluate(x_test, y_test)
print(results)

history_dict = history.history

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('image.png')
# plt.show()
