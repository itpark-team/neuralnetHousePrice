import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

min_max_scaler = preprocessing.MinMaxScaler()

houses_data = pd.read_csv('housepricedata.csv')
load_dataset = houses_data.values

# train models
# X = load_dataset[:, 0:10]
# Y = load_dataset[:, 10]
#
# X_scale = min_max_scaler.fit_transform(X)
#
# X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
#
# X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
#
# model = Sequential([
#     Dense(32, activation='relu', input_shape=(10,)),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid'),
# ])
#
# model.compile(optimizer='sgd',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# hist = model.fit(X_train, Y_train,
#                  batch_size=32, epochs=100,
#                  validation_data=(X_val, Y_val))
#
# accuracy_result = model.evaluate(X_test, Y_test)[1]
# print(accuracy_result)
#
# model.save("training4.h5")

# prediction
# model = keras.models.load_model("training4.h5")
#
# predict_dataset = np.concatenate((load_dataset, [[8712, 5, 7, 992, 1, 0, 2, 5, 0, 756, 0]]), axis=0)
# predict_X = predict_dataset[:, 0:10]
# predict_X_scale = min_max_scaler.fit_transform(predict_X)
#
# predict_vals = [[]]
# for curVal in predict_X_scale[-1]:
#     predict_vals[0].append(curVal)
#
# print(predict_vals)
#
# predict_result = model.predict(predict_vals)
# print(predict_result)
#
# if predict_result >= 0.5:
#     print(1)
# else:
#     print(0)
