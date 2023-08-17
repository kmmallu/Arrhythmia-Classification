vgg_16_model=Sequential()

vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',  input_shape=(187,1)))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

vgg_16_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

vgg_16_model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=512, kernel_size=1, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(Conv1D(filters=512, kernel_size=1, activation='relu', padding='same'))
vgg_16_model.add(BatchNormalization())
vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

vgg_16_model.add(Flatten())
vgg_16_model.add(Dense(4096, activation='relu'))
vgg_16_model.add(Dropout(0.4))
vgg_16_model.add(Dense(4096, activation='relu'))
vgg_16_model.add(Dropout(0.4))
vgg_16_model.add(Dense(5, activation='softmax'))

vgg_16_model.summary()

plot_model(vgg_16_model)

vgg_16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train=valX
y_train=valy
X_test=testX
y_test=testy

vgg_16_model_history = vgg_16_model.fit(X_train, y_train, epochs = 5, batch_size = 100, validation_data = (X_test, y_test))


plt.plot(vgg_16_model_history.history['accuracy'])
plt.plot(vgg_16_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.plot(vgg_16_model_history.history['loss'])
plt.plot(vgg_16_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')

y_true=[]
for element in y_test:
    y_true.append(np.argmax(element))
prediction_proba=vgg_16_model.predict(X_test)
prediction=np.argmax(prediction_proba,axis=1)

vgg_16_model_cf_matrix = confusion_matrix(y_true, prediction)
sns.heatmap(vgg_16_model_cf_matrix/np.sum(vgg_16_model_cf_matrix), annot=True,fmt='.3%', cmap='Blues')
