import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for n in range(0, (len(series)-window_size)):
        # make the input window from the beginning index untill window_size elements after it
        X.append(series[n:n+window_size])
        y.append(series[n+window_size])
    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    model = Sequential()
    model.add(LSTM(5, input_shape= (window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.

def cleaned_text(text):
    import string
    punctuation = ['!', ',', '.', ':', ';', '?']
    for char in text:
        # replace the text where it is not part of the punctuation or the ascii letters
        if char not in punctuation and char not in string.ascii_letters:
            text = text.replace(char, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    index = 0
    while index < (len(text)-window_size):
        # make the input window from the beginning index untill window_size elements after it
        inputs.append(text[index:index+window_size])
        outputs.append(text[index+window_size])
        index += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
