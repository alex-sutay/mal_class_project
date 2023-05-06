import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input

from get_data import get_data, data_to_arrays, get_mwr 


def build_classifier(input_shape, inner_shapes, output_shape):
    model = Sequential()
    model.add(Input(input_shape))
    for shape in inner_shapes:
        model.add(Dense(shape, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    return model


def tf_count(t, val):
        elements_equal_to_value = tf.equal(t, val)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        return count


if __name__ == '__main__':
    ben, mal = data_to_arrays(get_data('drebin-215-dataset-5560malware-9476-benign.csv'))
    ben_train, mal_train, ben_test, mal_test = get_mwr(ben, mal, 0.4, 0.8)
    model = build_classifier(ben_train.shape[1], [200, 200], 1)
    model.summary()
    x_train = np.vstack((ben_train, mal_train))
    y_train = np.array([[0]] * len(ben_train) + [[1]] * len(mal_train))
    x_test = np.vstack((ben_test, mal_test))
    y_test = np.array([[0]] * len(ben_test) + [[1]] * len(mal_test))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc', results)
    print(np.squeeze(model(np.expand_dims(x_test[0], 0))), np.squeeze(y_test[0]))

