import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import os
from keras.callbacks import EarlyStopping

def set_seed(seed_value):
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    tf.random.set_seed(seed_value)  # Tensorflow module.
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Environment variable.

seed_value = 666
set_seed(seed_value)

# 載入訓練資料
train_data = pd.read_csv("D:/NTUT/deep_learning_venv/DLcourse/firstexam/Data/secondReport/regr_trn.csv", header=None)

# 分離特徵和class
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# 縮放特徵數據
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)



# 載入測試資料
test_data = pd.read_csv("D:/NTUT/deep_learning_venv/DLcourse/firstexam/Data/secondReport/regr_tst.csv", header=None)
X_test = test_data.values
X_test_scaled = scaler.transform(X_test)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=666)

model = Sequential()

model.add(Dense(15, activation='relu', input_shape=(7,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(
    monitor='val_loss',  # 監控驗證集的損失值
    patience=10,  # 在10個epoch後驗證損失沒有改善，則提前終止訓練
    verbose=1,
    restore_best_weights=True  # 恢復到最佳模型權重
)

trainmodel = model.fit(X_train_scaled, y_train, epochs=200, batch_size=8,
                        validation_data=(X_val_split, y_val_split),
                          verbose=1, callbacks=[early_stopping])
# 訓練過程中的最後一個loss和accuracy
#last_loss = trainmodel.history['loss'][-1]
#last_mae = trainmodel.history['mae'][-1]



# 儲存neural network model
model.save("report2.h5")





# 預測測試資料的目標值
y_test_pred = model.predict(X_test_scaled)

# 將預測結果添加到測試數據並儲存
test_data[''] = y_test_pred
test_data.to_csv("regr_ans.csv", index=False, header=None)

best_val_loss_index = np.argmin(trainmodel.history['val_loss'])
best_val_loss = trainmodel.history['val_loss'][best_val_loss_index]
best_val_mae = trainmodel.history['val_mae'][best_val_loss_index]

print(f'Best Validation MSE: {best_val_loss}')
print(f'Best Validation MAE: {best_val_mae}')

import matplotlib.pyplot as plt

def plot_trainmodel(history):
    # 繪製MSE趨勢圖
    plt.figure()
    plt.plot(trainmodel.history['loss'])
    plt.plot(trainmodel.history['val_loss'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('mse.png') # 儲存MSE趨勢圖

    # 繪製MAE趨勢圖
    plt.figure()
    plt.plot(trainmodel.history['mae'])
    plt.plot(trainmodel.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('mae.png') # 儲存MSE趨勢圖

    plt.show()



plot_trainmodel(trainmodel)