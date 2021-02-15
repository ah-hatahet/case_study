import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from helpers import load_file, get_unique_chars, preprocess_texts
from typing import Tuple



def gen_model():

    n_chars = 29
    input_len = 29

    model = Sequential()
    model.add(Input(shape=(input_len, n_chars)))
    model.add(Bidirectional(LSTM(400, return_sequences=False)))
    model.add(RepeatVector(input_len))
    model.add(Bidirectional(LSTM(350, return_sequences=True)))
    model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
    return model



def main():

  # import data
  factors, expansions = load_file('train.txt')

  # split data
  factors = np.array(factors)
  expansions = np.array(expansions)
  X_train, X_test, y_train, y_test = train_test_split(factors, expansions, test_size=0.2, random_state=1)

  # get the unique elements in all factors/expansions and make dictionaries of (char,code_num) and (code_num,char)
  unique_chars = get_unique_chars(X_train,y_train)
  char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
  int_to_char = dict((i, c) for i, c in enumerate(unique_chars))
  n_chars = len(unique_chars)

  # preprocess data
  X_train, y_train = preprocess_texts(X_train, char_to_int), preprocess_texts(y_train, char_to_int)

  # create model
  model = gen_model()

  n_batch = 50
  n_epoch = 8

  callback =  EarlyStopping(monitor='accuracy')

  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  # get model summary and weights
  model_summary = model.summary()
  print(model_summary)
  with open('network.txt','w') as fh:
      model.summary(print_fn=lambda x: fh.write(x + '\n'))

  weights = model.get_weights()
  with open('weights.txt', 'w') as f:
      for item in weights:
          f.write("%s\n" % item)

  # train model
  model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch)

  model.save("model.hdf5")
  print("Done")

if __name__ == "__main__":
    main()