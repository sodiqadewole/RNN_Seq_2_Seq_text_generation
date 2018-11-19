# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:46:11 2018

@author: SODIQ-PC
"""

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras.models import load_model
import numpy as np
import os

data = """The cat and her kittens
They put on their mittens,
To eat a Christmas pie.
The poor little kittens
They lost their mittens,
And then they began to cry.
O mother dear, we sadly fear
We cannot go to-day,
For we have lost our mittens."
"If it be so, ye shall not go,
For ye are naughty kittens."""

tokenizer = Tokenizer()

def dataset_preparation(data):
    '''This function helps prepare the data into 2_D array of sequences
    with each row padded to the maximum length in the entire sequence
    
    :params data: our entire dataset as a single string with each 
    sentence separated by new_line
    '''
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len, padding = 'pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    
    return predictors, label, max_sequence_len, total_words
    

def create_model(predictors, label, max_sequence_len, total_words):
    '''This function creates an LSTM network taking the X and 
    label as input
    
    :params predictors: design matrix of padded sequence of our encoded text
    :params label: immediate next word following a sequence of predictor
    :params max_sequence_len: length of the maximum sequence in the set of predictors
    :params total_words: the vocabulary set'''
    
    input_len = max_sequence_len - 1
    
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)
    
    return model
    

def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,
                                   padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    return seed_text

def save_model():
    """
    This function helps to create and save the model in 
    the current directory
    """
    X, Y, max_len, total_words = dataset_preparation(data)
    model = create_model(X, Y, max_len, total_words)
    model.save(os.path.join(os.getcwd(),"seq2seq_model.h5"))
    

    
def main(inputSpeech):
    """The main function loads the model from 
    the current directory and predict the next 
    sequence based on the input""" 
    inputSpeech = str(inputSpeech)
    model = load_model(os.path.join(os.getcwd(), "seq2seq_model.h5"))
    _, _, max_len, _ = dataset_preparation(data)
    text = generate_text(inputSpeech, 3, max_len, model)
    
    return text

