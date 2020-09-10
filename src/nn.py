import json
import os
import time
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation,RepeatVector
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History

from dataset import Dataset


class NNModel:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 200
        self.dataset = Dataset(self.batch_size)
        self.model, self.encoder_model, self.decoder_model = self.create_model(
            self.dataset.num_input_paragraph_tokens,
            self.dataset.input_paragraph_max_seq_length,
            self.dataset.num_input_question_tokens,
            self.dataset.input_question_max_seq_length,
            self.dataset.num_target_tokens)
        self.model.summary()

        checkpoint = ModelCheckpoint(os.path.join('..', 'model_%s.h5' % time.time()), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        reduce_alpha = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001)
        # es = Earpping(monitor='val_loss', mode='min', verbose=1, patience=2)
        self.callbacks = [checkpoint, reduce_alpha]


    def create_model(self, num_encoder_paragraph_tokens, max_encoder_paragraph_seq_length,
                     num_encoder_question_tokens, max_encoder_question_seq_length, num_decoder_tokens):
        hidden_units = 256
        embed_hidden_units = 100

        context_inputs = Input(shape=(None,), name='context_inputs')
        encoded_context = Embedding(input_dim=num_encoder_paragraph_tokens + 1, output_dim=embed_hidden_units,
                                    input_length=max_encoder_paragraph_seq_length,
                                    name='context_embedding')(context_inputs)
        encoded_context = Dropout(0.3)(encoded_context)

        question_inputs = Input(shape=(None,), name='question_inputs')
        encoded_question = Embedding(input_dim=num_encoder_question_tokens + 1, output_dim=embed_hidden_units,
                                     input_length=max_encoder_question_seq_length,
                                     name='question_embedding')(question_inputs)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = LSTM(units=embed_hidden_units, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(max_encoder_paragraph_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])

        encoder_lstm = LSTM(units=hidden_units, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(merged)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=hidden_units, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        encoder_model = Model([context_inputs, question_inputs], encoder_states)

        decoder_state_inputs = [Input(shape=(hidden_units,)), Input(shape=(hidden_units,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

    def train(self):
        history = self.model.fit_generator(
            generator=self.dataset.train_gen,
            steps_per_epoch=self.dataset.train_num_batches,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.dataset.test_gen,
            validation_steps=self.dataset.test_num_batches,
            callbacks=self.callbacks)

    def test(self, context, question):
        predicted_answer = self.reply(context, question)
        print('context: ', context)
        print('question: ', question)
        print('answer: ', predicted_answer)


    def reply(self, paragraph, question):
        input_paragraph_seq = []
        input_question_seq = []
        input_paragraph_wid_list = []
        input_question_wid_list = []
        input_paragraph_text = paragraph.lower()
        input_question_text = question.lower()
        for word in nltk.word_tokenize(input_paragraph_text):
            if not self.in_white_list(word):
                continue
            idx = 1  # default [UNK]
            if word in self.dataset.input_paragraph_word2idx:
                idx = self.dataset.input_paragraph_word2idx[word]
            input_paragraph_wid_list.append(idx)
        for word in nltk.word_tokenize(input_question_text):
            if not in_white_list(word):
                continue
                idx = 1  # default [UNK]
            if word in input_question_word2idx:
                idx = input_question_word2idx[word]
            input_question_wid_list.append(idx)
        input_paragraph_seq.append(input_paragraph_wid_list)
        input_question_seq.append(input_question_wid_list)

        input_paragraph_seq = pad_sequences(input_paragraph_seq, self.dataset.input_paragraph_max_seq_length)
        input_question_seq = pad_sequences(input_question_seq, self.dataset.input_question_max_seq_length)
        states_value = self.encoder_model.predict([input_paragraph_seq, input_question_seq])

        target_seq = np.zeros((1, 1, num_target_tokens))
        target_seq[0, 0, target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= target_max_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()
