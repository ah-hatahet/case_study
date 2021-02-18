import sys
import numpy as np
from typing import Tuple
from tensorflow.keras.models import load_model
from tqdm import tqdm
from helpers import invert


def load_file(file_path):
    """ A helper functions that loads the file into a tuple of strings
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion, pred_expansion):
    """ the scoring function - this is how the model will be evaluated
    """
    return int(true_expansion == pred_expansion)


def preprocess_text(X, char_to_int):

    """ function to preprocess text before feeding into model
    """

    max_length = 29
    n_chars = len(char_to_int)
    # pad text
    X_padded = ''.join([' ' for _ in range(max_length-len(X))]) + X
    # integer encode strings
    X_encoded = [char_to_int[char] for char in X_padded]
    # one-hot encode
    X_onehot = list()
    for index in X_encoded:
        vector = [0 for _ in range(n_chars)]
        vector = np.asarray(vector)
        vector[index] = 1
        X_onehot.append(vector)
    X_onehot = np.asarray(X_onehot)

    X_onehot = np.array(X_onehot)

    return X_onehot



def predict(factor, model):

    # unique chars codes for preprocessing
    char_to_int = {' ': 0, '(': 1, ')': 2, '*': 3, '+': 4, '-': 5, '0': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '9': 15, 'a': 16, 'c': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 21, 'n': 22, 'o': 23, 's': 24, 't': 25, 'x': 26, 'y': 27, 'z': 28}
    int_to_char = {0: ' ', 1: '(', 2: ')', 3: '*', 4: '+', 5: '-', 6: '0', 7: '1', 8: '2', 9: '3', 10: '4', 11: '5', 12: '6', 13: '7', 14: '8', 15: '9', 16: 'a', 17: 'c', 18: 'h', 19: 'i', 20: 'j', 21: 'k', 22: 'n', 23: 'o', 24: 's', 25: 't', 26: 'x', 27: 'y', 28: 'z'}
    n_chars = len(int_to_char)

    # preprocess data
    factor_processed = preprocess_text(factor, char_to_int)

    # expand dimensions to fit value into model
    factor_processed = np.expand_dims(factor_processed, 0) 

    # make prediction
    pred = model.predict(factor_processed)

    # convert shape of pred
    pred = np.squeeze(pred,axis=0)

    # convert pred values to int to avoid very small numbers
    pred = np.rint(pred).astype(int)

    # decode pred
    expansion = invert(pred,int_to_char)
    expansion = expansion.replace(' ','')

    return expansion


def main(filepath: str):

    # load model
    model = load_model('model.hdf5')

    factors, expansions = load_file(filepath)

    factors_len = len(factors)

    pred = []
    for i in tqdm(range(factors_len)):
        prediction = predict(factors[i], model)
        pred.append(prediction)

    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))
    # 0.88



if __name__ == "__main__":
    main("valid.txt")
