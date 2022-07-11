import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import keras
from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json',encoding='UTF8').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern.lower())
    words.extend(word_list)
    documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

#print(words)
#print(classes)
#print(documents)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)  # [0, 0, 0, 0, 0, 0, 0]

for document in documents:
  bag = []
  word_patterns = document[0]
  #print(word_patterns)
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
  #print(word_patterns)
  for word in words:
    if word in word_patterns:
      bag.append(1)
    else:
      bag.append(0)

  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1  # 원핫 벡터  [0, 0, 0, 0, 1, 0, 0]
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', history)
print("Done")


