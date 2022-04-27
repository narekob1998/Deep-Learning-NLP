import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.utils import np_utils
from keras.layers import Dropout
from tensorflow.keras.models import load_model

model_link = "https://drive.google.com/file/d/1nkOaZK5tiruGG4xzLltZsU4Q-zrLd5ju/view?usp=sharing"

def get_trained_model(path):
    model = load_model(path) 
    return model

def create_model(mlen):
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=np.zeros((28,)),
                                      input_shape=(mlen, 28)))
    model.add(LSTM(128, return_sequences=True,
                   input_shape=(mlen, 28)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(28, activation='softmax'))

    model.compile(optimizer = "adam", loss= "CategoricalCrossentropy")
    return model

def generate_name(N, mlen=16):
    model = get_trained_model('trained_model.h5')
    
    name = []
    start_char = np.zeros((28,))
    start_char[26] = 1
    
    x = np.zeros((1, mlen, 28))
    i = 0
    
    while i < N:
        probs = list(model.predict(x)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(28), p=probs)
        if index == 27:
            break
        character = list('abcdefghijklmnopqrstuvwxyz<>')[index]
        name.append(character)
        x[0, i+1, index] = 1
        i += 1
    return ''.join(name)
