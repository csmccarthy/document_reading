import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
from mnistHelper import mnistHelper
import pickle

model = Sequential()

model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_dir = '/tmp/new_cnn_augmented3'

with open("C:\\Users\\csmccarthy\\Documents\\pythonprojects\\augmented_mnist\\mnist_data3_full.pkl", 'rb') as f:
        train_data = pickle.load(f)
with open("C:\\Users\\csmccarthy\\Documents\\pythonprojects\\augmented_mnist\\mnist_lbls3_full.pkl", 'rb') as f:
        train_labels = pickle.load(f)
print("Loaded training data")

train_data = train_data.reshape([-1, 28, 28, 1])
lr_decay = LearningRateScheduler(lambda x: 1e-2 * (0.95 ** x))
log = model.fit(x=train_data, y=train_labels, batch_size=2048, epochs = 3,
                verbose=1, callbacks=[lr_decay], use_multiprocessing=True)

eval_data, eval_lbls = mnistHelper.load_data(train=False, blanks=True)

loss, acc = model.evaluate(x=eval_data, y=eval_lbls, use_multiprocessing=True, batch_size=2048)

print('Eval loss:', loss)
print('Eval accuracy:', acc)

def predict(model):
	img = cv2.imread("C:/Users/csmccarthy/Documents/pythonprojects/testnum2.png", 0)
	img = np.asarray(img, dtype=np.float64)
	img = img.reshape([1, 28, 28, 1])
	label = np.argmax(model.predict(img))
	return label
