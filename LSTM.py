import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore

# This project uses the IMDB dataset, which consists of movie reviews,
#  to classify whether a review is positive or negative. We use an LSTM (Long Short-Term Memory) network,
#  a type of recurrent neural network (RNN), to handle the sequential nature of text data.

max_features = 10000  
maxlen = 500  

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)

model = models.Sequential()

model.add(layers.Embedding(max_features, 128, input_length=maxlen))
model.add(layers.LSTM(128, return_sequences=False))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=5, 
                    batch_size=64, 
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
