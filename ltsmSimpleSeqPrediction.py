import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from numSeqPredictor import get_each_num

final_predicted_nums = []
for number in range(0,7):
    # data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    data = get_each_num()[i]
    # print(data)


    def splitSequence(seq, n_steps):
        # Declare X and y as empty list
        X = []
        y = []

        for i in range(len(seq)):
            # get the last index
            lastIndex = i + n_steps

            # if lastIndex is greater than length of sequence then break
            if lastIndex > len(seq) - 1:
                break

            # Create input and output sequence
            seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]

            # append seq_X, seq_y in X and y list
            X.append(seq_X)
            y.append(seq_y)
            pass
        # Convert X and y into numpy array
        X = np.array(X)
        y = np.array(y)

        return X, y

        pass


    # for count in range(0, 7):
    # integer_sequence = get_each_num()[1]
    n_steps = 5
    X, y = splitSequence(data, n_steps=n_steps)
    print(X)
    print(y)

    for i in range(len(X)):
        print(X[i], y[i])

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    print(X.shape)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    print(X[:2])

    model = tf.keras.Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(layers.Dense(1))
    print(model.layers)
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.fit(X, y, epochs=1000, verbose=1)

    test_data = np.array(data[-5:])
    test_data = test_data.reshape((1, n_steps, n_features))
    print(test_data)

    predictNextNumber = model.predict(test_data, verbose=1)
    print(predictNextNumber)
    final_predicted_nums.append(predictNextNumber)

print("----------------------------------")
print("final numbers:")
print(final_predicted_nums)
