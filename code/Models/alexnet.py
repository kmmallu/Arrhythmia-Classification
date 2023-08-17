from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
X, y = mit_train_data.iloc[: , :-1], mit_train_data.iloc[: , -1]
X, valX, y, valy= train_test_split(X,y,test_size=0.2)
testX, testy = mit_test_data.iloc[: , :-1], mit_test_data.iloc[: , -1]
y = to_categorical(y)
testy = to_categorical(testy)
valy=to_categorical(valy)

print("X shape=" +str(X.shape))
print("y shape=" +str(y.shape))
print("valX shape=" +str(valX.shape))
print("valy shape=" +str(valy.shape))
print("testX shape=" +str(testX.shape))
print("testy shape=" +str(testy.shape))

from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
alexNet_model=Sequential()

alexNet_model.add(Conv1D(filters=96, activation='relu', kernel_size=11, strides=4, input_shape=(187,1)))
alexNet_model.add(BatchNormalization())
alexNet_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

alexNet_model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
alexNet_model.add(BatchNormalization())
alexNet_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

alexNet_model.add(Conv1D(filters=384, padding='same', kernel_size=3, activation='relu'))
alexNet_model.add(Conv1D(filters=384, kernel_size=3, activation='relu'))
alexNet_model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
alexNet_model.add(BatchNormalization())
alexNet_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

alexNet_model.add(Flatten())
alexNet_model.add(Dense(4096, activation='relu'))
alexNet_model.add(Dropout(0.4))
alexNet_model.add(Dense(4096, activation='relu'))
alexNet_model.add(Dropout(0.4))
alexNet_model.add(Dense(5, activation='softmax'))

alexNet_model.summary()

from tensorflow.keras.utils import plot_model
plot_model(alexNet_model)

alexNet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

alexNet_model_history = alexNet_model.fit(valX, valy, epochs = 5, batch_size = 100, validation_data = (testX, testy))

plt.plot(alexNet_model_history.history['accuracy'])
plt.plot(alexNet_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


plt.plot(alexNet_model_history.history['loss'])
plt.plot(alexNet_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')

y_true=[]
for element in testy:
    y_true.append(np.argmax(element))
prediction_proba=alexNet_model.predict(testX)
prediction=np.argmax(prediction_proba,axis=1)

from sklearn.metrics import confusion_matrix
alexNet_model_cf_matrix = confusion_matrix(y_true, prediction)
sns.heatmap(alexNet_model_cf_matrix/np.sum(alexNet_model_cf_matrix), annot=True,fmt='.3%', cmap='Blues')
