import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def audio_data():
    feat = np.load('feat.npy')
    rs = []
    for i in feat:
        for j in range(30):
            rs.append(list(i))
    rs = np.array(rs)
    return rs


def test_data_gen():
    audio_batch = audio_data()
    x = os.listdir('features_X')
    li = []
    for i in range(len(x)):
        tem = []
        x_f = './features_X/' + x[i]
        y_f = './features_y/' + x[i]
        tem.append(x_f)
        tem.append(y_f)
        li.append(tem)
    li.sort()
    test_len = len(li)
    X_test = np.zeros((test_len, 30, 25, 2048))
    y_test = np.zeros((test_len*30,))
    for i in range(test_len):
        X_test[i, :, :, :] = np.load(li[i][0])
        for j in range(30):
            y_test[i*30+j] = float(np.load(li[i][1]))
    X_test = np.reshape(X_test, (test_len * 30, 25 * 2048))
    X1 = X_test[:, :-3]
    mix_data = np.concatenate((audio_batch, X1), axis=1)
    mix_data = StandardScaler().fit_transform(mix_data)
    pca = PCA(3)
    mix_data = pca.fit_transform(mix_data)
    X_test = np.concatenate((X1, mix_data), axis=1)

    np.save('./features/x_test_audio.npy', X_test)
    np.save('./features/y_test_audio.npy', y_test)

test_data_gen()









