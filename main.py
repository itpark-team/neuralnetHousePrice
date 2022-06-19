import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# houses_data = pd.read_csv('housepricedata.csv')
# dataset = houses_data.values
#
# X = dataset[:, 0:10]
# Y = dataset[:, 10]
#
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scale = min_max_scaler.fit_transform(X)
#
# X_scale = preprocessing.normalize(X)
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
#                  batch_size=32, epochs=150,
#                  validation_data=(X_val, Y_val))
#
# accuracy_result = model.evaluate(X_test, Y_test)[1]
# print(accuracy_result)
#
# model.save("training3.h5")


# my_array_1 = [[0.04159948, 0.44444444, 0.75, 0.17774141, 0.33333333, 0., 0.375, 0.33333333, 0., 0.34555712]]  # 0
# my_array_2 = [[0.03879502, 0.55555556, 0.5, 0.23175123, 0.33333333, 0., 0.375, 0.41666667, 0.66666667, 0.29478138]]  # 1

model = keras.models.load_model("training3.h5")

my_array = [[9550, 7, 5, 756, 1, 0, 3, 7, 1, 642]]
my_array_scale = preprocessing.normalize(my_array)

predict_result = model.predict(my_array_scale)
print(predict_result)

if predict_result >= 0.5:
    print(1)
else:
    print(0)
