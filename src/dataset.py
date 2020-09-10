import nltk
#nltk.download("punkt")
import os
import json
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.lookup = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
        self.data_path = '..\\data'
        with open(os.path.join(self.data_path, "train-v2.0.json"), mode="rt", encoding="utf-8") as file:
            self.qa_data = json.load(file)
        self.max_data_count = 10000
        self.max_context_seq_length = 300
        self.max_question_seq_length = 60
        self.max_target_seq_length = 50
        self.data = self.get_SQuAD_data(
            self.qa_data,
            self.max_data_count,
            self.max_context_seq_length,
            self.max_question_seq_length,
            self.max_target_seq_length)
        self.create_dictionary()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_encoded_data_samples,
            self.target_encoded_data_samples,
            test_size=0.2,
            random_state=42)
        self.train_gen = self.generate_batch(
            self.input_paragraph_max_seq_length,
            self.input_question_max_seq_length,
            self.target_max_seq_length,
            self.num_target_tokens,
            self.x_train,
            self.y_train,
            self.batch_size)
        self.test_gen = self.generate_batch(
            self.input_paragraph_max_seq_length,
            self.input_question_max_seq_length,
            self.target_max_seq_length,
            self.num_target_tokens,
            self.x_test,
            self.y_test,
            self.batch_size)
        self.train_num_batches = len(self.x_train) // self.batch_size
        self.test_num_batches = len(self.x_test) // self.batch_size


    def create_dictionary(self):
        max_input_vocab_size = 5000
        max_target_vocab_size = 5000

        input_data_samples = []
        output_data_samples = []

        self.input_paragraph_max_seq_length = 0
        self.input_question_max_seq_length = 0
        self.target_max_seq_length = 0

        input_paragraph_counter = Counter()
        input_question_counter = Counter()
        target_counter = Counter()
        # iterate over each paragraph, question and answer
        for sample in self.data:
            paragraph, question, answer = sample
            paragraph_word_list = [w.lower() for w in nltk.word_tokenize(paragraph) if self.in_white_list(w)]
            question_word_list = [w.lower() for w in nltk.word_tokenize(question) if self.in_white_list(w)]
            answer_word_list = [w.lower() for w in nltk.word_tokenize(answer) if self.in_white_list(w)]

            output_data = ['START'] + answer_word_list + ['END']

            input_data_samples.append([paragraph_word_list, question_word_list])
            output_data_samples.append(output_data)

            for w in paragraph_word_list:
                input_paragraph_counter[w] += 1
            for w in question_word_list:
                input_question_counter[w] += 1
            for w in output_data:
                target_counter[w] += 1

            self.input_paragraph_max_seq_length = max(self.input_paragraph_max_seq_length, len(paragraph_word_list))
            self.input_question_max_seq_length = max(self.input_question_max_seq_length, len(question_word_list))
            self.target_max_seq_length = max(self.target_max_seq_length, len(output_data))

        input_paragraph_word2idx = dict()
        input_question_word2idx = dict()
        target_word2idx = dict()

        # Mapping from word to index
        for idx, word in enumerate(input_paragraph_counter.most_common(max_input_vocab_size)):
            input_paragraph_word2idx[word[0]] = idx + 2
        for idx, word in enumerate(input_question_counter.most_common(max_input_vocab_size)):
            input_question_word2idx[word[0]] = idx + 2
        for idx, word in enumerate(target_counter.most_common(max_target_vocab_size)):
            target_word2idx[word[0]] = idx + 1

        target_word2idx['UNK'] = 0
        input_paragraph_word2idx['PAD'] = 0
        input_paragraph_word2idx['UNK'] = 1
        input_question_word2idx['PAD'] = 0
        input_paragraph_word2idx['UNK'] = 1

        # Mapping from index to word
        input_paragraph_idx2word = dict([(idx, word) for word, idx in input_paragraph_word2idx.items()])
        input_question_idx2word = dict([(idx, word) for word, idx in input_question_word2idx.items()])
        target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

        self.num_input_paragraph_tokens = len(input_paragraph_idx2word)
        self.num_input_question_tokens = len(input_question_idx2word)
        self.num_target_tokens = len(target_idx2word)

        self.input_encoded_data_samples = []
        self.target_encoded_data_samples = []
        # iterate over each text sample from paragraphs, questions and answers
        for input_data, output_data in zip(input_data_samples, output_data_samples):
            input_paragraph_encoded_data = []
            input_question_encoded_data = []
            target_encoded_data = []
            input_paragraph_data, input_question_data = input_data
            for word in input_paragraph_data:
                if word in input_paragraph_word2idx:
                    input_paragraph_encoded_data.append(input_paragraph_word2idx[word])
                else:
                    input_paragraph_encoded_data.append(1)
            for word in input_question_data:
                if word in input_question_word2idx:
                    input_question_encoded_data.append(input_question_word2idx[word])
                else:
                    input_question_encoded_data.append(1)
            for word in output_data:
                if word in target_word2idx:
                    target_encoded_data.append(target_word2idx[word])
                else:
                    target_encoded_data.append(0)
            self.input_encoded_data_samples.append([input_paragraph_encoded_data, input_question_encoded_data])
            self.target_encoded_data_samples.append(target_encoded_data)

    def generate_batch(self, input_paragraph_max_seq_length, input_question_max_seq_length, target_max_seq_length,
                       num_target_tokens, input_data, output_data, batch_size):
        num_batches = len(input_data) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_paragraph_data_batch = []
                encoder_input_question_data_batch = []
                for input_paragraph_data, input_question_data in input_data[start:end]:
                    encoder_input_paragraph_data_batch.append(input_paragraph_data)
                    encoder_input_question_data_batch.append(input_question_data)
                encoder_input_paragraph_data_batch = pad_sequences(encoder_input_paragraph_data_batch,
                                                                   input_paragraph_max_seq_length)
                encoder_input_question_data_batch = pad_sequences(encoder_input_question_data_batch,
                                                                  input_question_max_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, target_max_seq_length, num_target_tokens))
                decoder_input_data_batch = np.zeros(shape=(batch_size, target_max_seq_length, num_target_tokens))
                for lineIdx, target_wid_list in enumerate(output_data[start:end]):
                    for idx, wid in enumerate(target_wid_list):
                        if wid == 0:  # UNKNOWN
                            continue
                        decoder_input_data_batch[lineIdx, idx, wid] = 1
                        if idx > 0:
                            decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
                yield [encoder_input_paragraph_data_batch, encoder_input_question_data_batch,
                       decoder_input_data_batch], decoder_target_data_batch

    def in_white_list(self, _word):
        valid_word = False
        for char in _word:
            if char in self.lookup:
                valid_word = True
                break

        if valid_word is False:
            return False

        return True

    def get_SQuAD_data(self,qa_data,max_data_count,max_context_seq_length,max_question_seq_length,max_target_seq_length):
        data = list()
        for instance in self.qa_data['data']:
            for paragraph in instance['paragraphs']:
                context = paragraph['context']
                context_wid_list = [w.lower() for w in nltk.word_tokenize(context) if self.in_white_list(w)]
                if len(context_wid_list) > max_context_seq_length:
                    continue
                qas = paragraph['qas']
                for qas_instance in qas:
                    question = qas_instance['question']
                    question_wid_list = [w.lower() for w in nltk.word_tokenize(question) if self.in_white_list(w)]
                    if len(question_wid_list) > max_question_seq_length:
                        continue
                    answers = qas_instance['answers']
                    for answer in answers:
                        ans = answer['text']
                        answer_wid_list = [w.lower() for w in nltk.word_tokenize(ans) if self.in_white_list(w)]
                        if len(answer_wid_list) > max_target_seq_length:
                            continue
                        if len(data) < max_data_count:
                            data.append((context, question, ans))

                if len(data) >= max_data_count:
                    break

                break
        return data

