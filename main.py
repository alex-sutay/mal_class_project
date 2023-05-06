import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

from get_data import get_data, data_to_arrays, get_mwr 
from build_classifier import build_classifier


def evaluate(model, x, y):
    rates = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    res = tf.math.round(model(x))
    for i in range(len(res)):
        # is negative
        if y[i][0] == 0:
            # said negative
            if res[i][0] == 0:
                rates['TN'] += 1
            # said positive
            else:
                rates['FN'] += 1
        # is positive
        elif y[i][0] == 1:
            # said positive
            if res[i][0] == 1:
                rates['TP'] += 1
            # said negative
            else:
                rates['FP'] += 1
    return rates


def train():
    ben, mal = data_to_arrays(get_data('/scratch/sutay/malware_data/drebin-215-dataset-5560malware-9476-benign.csv'))
    results = {}
    with open('models.json', 'r') as f:
        models = json.load(f)
    for m in models:
        this_name = f'model_{"-".join(str(l) for l in m["shape"])}_mwr_{m["mwr"]}'
        ben_train, mal_train, ben_test, mal_test = get_mwr(ben, mal, m['mwr'], 0.8)
        model = build_classifier(ben_train.shape[1], m['shape'], 1)
        x_train = np.vstack((ben_train, mal_train))
        y_train = np.array([[0]] * len(ben_train) + [[1]] * len(mal_train))
        x_test = np.vstack((ben_test, mal_test))
        y_test = np.array([[0]] * len(ben_test) + [[1]] * len(mal_test))
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)
        rates = evaluate(model, x_test, y_test)
        model.save('models/'+this_name)
        results[this_name] = rates
    with open('results.json', 'w') as f:
        json.dump(results, f)


def gen_adv_ex(x, y, F, k, I):
    adv_x = tf.Variable(np.expand_dims(x, 0), dtype=float)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    changes = 0
    for changes in range(k):
        with tf.GradientTape() as tape:
            tape.watch(adv_x)
            pred = F(adv_x)
            loss = loss_object(tf.convert_to_tensor([[1.]]), pred)  # assumes the label is supposed to be malware: 1

        if pred < 0.5:  # predicts benign, exit
            break

        grad = tape.gradient(loss, adv_x)
        i_max = -1
        # get the i_max
        for i in np.argsort(-grad)[0]:
            if i in I and adv_x[0][i] == 0:
                i_max = i
                break

        if i_max == -1:  # found nothing valuable to add
            break

        add_array = np.zeros(adv_x.shape)
        add_array[0, i_max] = 1
        adv_x.assign_add(tf.convert_to_tensor(add_array, dtype=float))

    return changes, adv_x


def test_gen_adv():
    # load the data
    ben, mal = data_to_arrays(get_data('drebin-215-dataset-5560malware-9476-benign.csv'))
    # get the valid features to change I
    with open('dataset-features-categories.csv') as f:
        feat_data = f.read()
        features = [feat.split(',') for feat in feat_data.split('\n')][:-1]  # cuts off the last line, which is empty
        allowed = {'Hardware Components', 'Manifest Permission', 'Components', 'Intent'}
        I = [i for i in range(len(features)) if features[i][1] in allowed]
    results = {}
    with open('models.json', 'r') as f:
        models = json.load(f)
    for m in models:
        this_name = f'model_{"-".join(str(l) for l in m["shape"])}_mwr_{m["mwr"]}'
        # load and test the models
        model = tf.keras.models.load_model('models/' + this_name)
        new_mal = None
        len_mal = len(mal)
        total_dis = 0
        for i, ex in enumerate(mal):
            dis, adv_ex = gen_adv_ex(ex, 0, model, 20, I)
            total_dis += dis
            if i % 50 == 0:
                print(f'\r{this_name}: {i}/{len_mal} ({i/len_mal:.2%})', end='    ')

            if new_mal is None:
                new_mal = adv_ex
            else:
                new_mal = np.vstack((new_mal, adv_ex))

        x_test = np.vstack((ben, new_mal))
        y_test = np.array([[0]] * len(ben) + [[1]] * len(new_mal))
        rates = evaluate(model, x_test, y_test)
        results[this_name] = [rates, total_dis / len_mal]
        print(results[this_name])
    with open('adv_results.json', 'w') as f:
        json.dump(results, f)


def main():
    train()
    test_gen_adv()


if __name__ == '__main__':
    main()

