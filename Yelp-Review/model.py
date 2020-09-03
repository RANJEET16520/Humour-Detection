import sys
print(sys.executable)

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_gpu_support())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



funny_data = []
nfunny_data = []

with open('Model_Input/funny_data.txt', 'rb') as file:
	funny_data = pickle.load(file)
print(len(funny_data))

with open('Model_Input/nfunny_data.txt', 'rb') as file:
	nfunny_data = pickle.load(file)
print(len(nfunny_data))



print("############################## Word2Vec-Model ##############################")
import gensim
w2v_model = gensim.models.Word2Vec.load("word2vec.model")
words = list(w2v_model.wv.vocab)
Embedding_index = w2v_model[words]
print(Embedding_index.shape)



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_joke= funny_data + nfunny_data
print(len(train_joke))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_joke)
encoded_joke = tokenizer.texts_to_sequences(train_joke)

print(len(encoded_joke[1]))
print(encoded_joke[1])
print(train_joke[1])

max_length = max([len(s.split()) for s in train_joke])
print(max_length)

X = pad_sequences(encoded_joke, maxlen=max_length)
print(len(X))
Y = np.array([1 for _ in range(46485)] + [0 for _ in range(46485)])
print(len(Y))

print(train_joke[1])
print(X[1])
print(Y[1])



print("############################## Deep Learning-Model ##############################")
from sklearn.model_selection import train_test_split
X_train, X_text, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_text.shape, Y_train.shape, Y_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

model_joke = Sequential()
model_joke.add(Embedding(Embedding_index.shape[0], Embedding_index.shape[1], weights=[Embedding_index], input_length=max_length, trainable=False))
# CNN Model
model_joke.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model_joke.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
model_joke.add(MaxPooling1D(pool_size=2))
model_joke.add(Dropout(rate = 0.6))
model_joke.add(Flatten())
model_joke.add(Dense(256, activation='relu'))
model_joke.add(Dense(1, activation='sigmoid'))
model_joke.summary()

# #LSTM Model
# model_joke.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model_joke.add(Dense(256, activation='relu'))
# model_joke.add(Dense(1, activation='sigmoid'))
# model_joke.summary()

model_joke.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_joke.fit(X_train, Y_train, validation_data=(X_text, Y_test), epochs=5, batch_size=32, verbose=2)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')	
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss, acc = model_joke.evaluate(X_text, Y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))
print('Test Loss: %f' % (np.exp(loss)))



from sklearn.metrics import accuracy_score, r2_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score

Y_pred = model_joke.predict_classes(X_text)
for i in range(20):
	print('Actual = {}:, Predicted = {}'.format(Y_test[i], Y_pred[i]))

acc = accuracy_score(Y_test, Y_pred)
print("Accuracy : {}".format(acc))

error = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error : {}".format(error))

error = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error : {}".format(error))

score = r2_score(Y_test, Y_pred)
print("R2 Score : {}".format(score))

score = precision_score(Y_test, Y_pred, average='macro')
print("Precision Score : {}".format(score))

score = recall_score(Y_test, Y_pred, average='macro')
print("Recall Score : {}".format(score))

score = f1_score(Y_test, Y_pred)
print("F1 Score : {}".format(score))

score = fbeta_score(Y_test, Y_pred, beta = 0.5)
print("F-Beta Score : {}".format(score))


jokes_roc = roc_auc_score(Y_test, Y_pred)
print('Jokes: ROC-Area Under Curve = {}'.format(jokes_roc))

J_fpr, J_tpr, _ = roc_curve(Y_test, Y_pred)
plt.plot(J_fpr, J_tpr, color = 'blue', label = 'Jokes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


