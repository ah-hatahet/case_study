import numpy as np
from numpy import argmax
from typing import Tuple


# functions
def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def get_unique_chars(array1, array2):

	""" gets unique elements in two arrays """

	a1_unique_chars = {char for word in array1 for char in word}
	a2_unique_chars = {char for word in array2 for char in word}

	unique_chars = a1_unique_chars | a2_unique_chars
	unique_chars.add(' ')
	unique_chars = sorted(unique_chars)

	return unique_chars


def pad_data(texts):

	""" pads a vector of strings by adding zeroes """

	texts_padded = []
	max_length = 29

	for element in texts:
		strp =  ''.join([' ' for _ in range(max_length-len(element))]) + element
		texts_padded.append(strp)

	return texts_padded



def integer_encode(texts, char_to_int):

	''' encode each str in vector into a unique number '''

	texts_encoded = list()

	for pattern in texts:
		integer_encoded = [char_to_int[char] for char in pattern]
		texts_encoded.append(integer_encoded)

	return texts_encoded
  

def one_hot_encode(texts, max_int):

	''' one-hot encodes each code list in vector '''

	texts_onehot = list()

	for seq in texts:
		pattern = list()

		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector = np.asarray(vector)
			vector[index] = 1
			pattern.append(vector)
		texts_onehot.append(np.asarray(pattern))

	return texts_onehot



# invert encoding
def invert(seq, int_to_char):

	''' converts binary arrays into strings '''

	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
		
	return ''.join(strings)



def preprocess_text(X, char_to_int):

	''' preprocess a single string '''

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


def preprocess_texts(X,char_to_int):

	''' preprocess a vector of strings '''

	max_length = 29
	n_chars = len(char_to_int)

	# pad text
	X_padded = pad_data(X)
	# integer encode texts
	X_encoded = integer_encode(X_padded, char_to_int)
	# one hot encode texts
	X_onehot = one_hot_encode(X_encoded, max_length)

	# convert to np array
	X_onehot = np.array(X_onehot)

	return X_onehot

