from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #編號0~9999的單字載入

print(train_data[0]) #數字組成的list
print(train_labels[0]) #0:負面評論，1:正面評論

print(max([max(sequence) for sequence in train_data])) #max([各評論內最大之數])

word_index = imdb.get_word_index() #取得word: number的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #反轉為number: word的字典
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]]) #因為load_data()時會自動將所有數字+3，0~2有其他作用，所以-3才是真正的編號
                #每隔一格加入單字             #若無此key則輸出'?'
print(decoded_review)

#數據向量化: 將list轉成張量，才能輸入神經網路
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences): #enumerate()為每個子串列編號，i: 編號，sequence: 子串列
        results[i, sequence] = 1.
    return results

#向量化
#2D
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0])
#1D
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#設定驗證資料集
x_val = x_train[:10000] #val
partial_x_train = x_train[10000:] #train

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#輸入資料: 向量，標籤: 純量
#用具有relu啟動函數(activation)的全連接層(密集層)堆疊架構: Dense(16, activation='relu')
#output = relu(dot(W, input) + b)
#16個神經單元(neuron)表示權重矩陣W.shape = (input_dimension, 16)
#將input資料與W做點積後，將input資料映射至16維的表示空間中
#表示空間的維度 = 學習內部資料轉換時允許神經網路有多少自由度
#擁有更多神經單元(更多維表現空間)
#優點: 可讓神經網路學習更複雜的資料表示法
#缺點: 神經網路的計算成本提高，導致學習到不想要的資料樣態(pattern)
#--> 提高訓練資料的成效，但不見得增進測試資料的成效 --> overfitting
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10000, )), #輸入層&隱藏層
    layers.Dense(16, activation='relu'), #隱藏層
    layers.Dense(1, activation='sigmoid') #輸出層
])
#若無relu(非線性函數)的啟動函數，Dense層只會是: output = dot(W, input) + b
#該層只能學習輸入資料的線性變換(affine仿射變換): 該層的假設空間是輸入資料映射到16維空間的所有可能的線性變換集合
#這樣的假設空間太過侷限且不利多層的轉換表示，因為線性層的深層堆疊仍只會做線性運算，添加再多層也不會擴展其假設空間
#因此為了獲得更豐富多樣的假設空間，以利深度轉換的表現，需要非線性函數或啟動函數

#選擇損失函數和優化器:
#處理二元分類問題，且神經網路輸出的是機率值 --> binary_crossentropy損失函數
#輸出為機率值的模型 --> crossentropy(交叉商)
#衡量機率分布間的距離(差異)，這裡用來測量真實分布(標準答案)與預測分布間的距離
model.compile(optimizer='rmsprop', #指定優化器: 神經網路根據其輸入資料及損失函數值而自行更新的機制
        loss='binary_crossentropy', #指定損失函數: 衡量神經網路在訓練資料上的表現，並引導網路朝正確的方向修正
        metrics=['accuracy']) #指定評量準則: 圖片是否分類至正確類別
'''
from tensorflow.keras import optimizers, losses, metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001), #調整優化器參數lr的值
        loss=losses.binary_crossentropy, #自行指定其他的損失函數
        metrics=[metrics.binary_accuracy]) #自行指定其他的metric函數
'''

#訓練模型
history = model.fit(partial_x_train, partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

#繪製訓練與驗證的損失分數
import matplotlib.pyplot as plt

loss_values = history_dict['loss'] #每次訓練的loss訓練損失分數
val_loss_values = history_dict['val_loss'] #每次驗證的val_loss驗證損失分數

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss') #'bo': 藍點
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') #'b': 藍線
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend() #每個輸出圖表的圖像名稱

plt.show()

#繪製訓練和驗證精準度
plt.clf() #清除圖表
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend

plt.show()

#重新開始訓練模型，因為在epochs = 5後出現overfitting(過度配適)的現象
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10000, )),
    layers.Dense(16, activation='relu', input_shape=(10000, )),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train,
        epochs=5,
        batch_size=512)

results = model.evaluate(x_test, y_test)
print(results)

print(model.predict(x_test))