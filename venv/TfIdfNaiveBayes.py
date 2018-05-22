import math
import operator
import os
import re
import string
from collections import *
from datetime import datetime
from functools import *
from itertools import *
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# fumction to access the categories in training directory
def load_files(file_path):
    train_dir = os.listdir(file_path)
    train = [file_path + "/" + f for f in train_dir if f != '.DS_Store']
    return train


# Function that extracts words from a single file for testing purposes
def get_tokens_from_single_file(file_path):
    file = open(file_path, 'rt')
    text = file.read()
    file.close()
    # lemmatizer = WordNetLemmatizer()

    words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
    words = [word.lower() for word in words]  # lowercase
    words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

    stop_words = stopwords.words('english')  # filter out tokens that are stop words.
    words = [w for w in words if (w not in stop_words and len(w) != 1)]
    # words = [lemmatizer.lemmatize(w) for w in words]

    return words


class LazyDict(dict):
    def keylist(self, keys, value):
        for key in keys:
            self[key] = value


# gets vocabulary size of the training or test set
def get_vocab_size(train_dir):
    V = []
    # lemmatizer = WordNetLemmatizer()

    for category in train_dir:
        for filename in os.listdir(category):
            file = open(category + "/" + filename, 'rt')
            text = file.read()
            file.close()

            # table = str.maketrans('', '', string.punctuation)
            #  stripped = [w.translate(table) for w in words] # strip punctuation

            words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
            words = [word.lower() for word in words]  # lowercase
            words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

            stop_words = stopwords.words('english')  # filter out tokens that are stop words.
            words = [w for w in words if (w not in stop_words and len(w) != 1)]
            # words = [lemmatizer.lemmatize(w) for w in words]

            V.append(words)

    tokens = list(chain(*V))
    vocab = Counter(tokens)
    print("Vocabulary Size:")
    print(len(vocab))
    return len(vocab)


# Function returns the total number of documents
def get_number_of_documents(directories):
    counter = 0
    for category in directories:
        counter = counter + len(os.listdir(category))  # dir is your directory path

    return counter


# Function returns the total number of documents in a category
def number_of_documents_in_a_category(category):
    return (len(os.listdir(category)))


# Function that extracts words from a single category that are selected after applying a selection filters
def get_tokens_from_category(category):
    print("Extracting vocabulary from the category: " + os.path.basename(os.path.normpath((category))))

    V = []
    document_dictionaries = []
    lemmatizer = WordNetLemmatizer()

    for filename in os.listdir(category):
        file = open(category + "/" + filename, 'rt')
        text = file.read()
        file.close()

        words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
        words = [word.lower() for word in words]  # lowercase
        words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

        stop_words = stopwords.words('english')  # filter out tokens that are stop words.
        words = [w for w in words if (w not in stop_words and len(w) != 1)]
        # words = [lemmatizer.lemmatize(w) for w in words]

        document_words = list(set(words))

        d = LazyDict()
        d.keylist(document_words, 1)
        document_dictionaries.append(d)
        V.append(words)

    tokens = list(chain(*V))

    return [tokens, document_dictionaries]


# Function that maps each word in document category to its conditional probability given that category
def log_probabilities_for_tfidf_NB(smoothing,
                                   vocabulary_size, words,
                                   document_dictionaries,
                                   number_of_documents):  # Conditional Probabilities for multinomial NB

    number_of_words = 0

    word_counter = Counter(words)  # get each unique word

    for word in word_counter:
        counter = 0
        # for each word count the number of documents it is in
        for d in document_dictionaries:
            if word in d:
                counter = counter + 1

        idf = math.log10(number_of_documents / counter)
        tf = word_counter[word]
        tf_idf_value = idf * tf
        number_of_words = number_of_words + tf_idf_value
    # add a new token <UNK> for Unknown, that occurs nowhere in tokens.
    # This will map all unknown words in documents to be classified (test documents) to UNK

    log_prob_dictionary = {"<UNK>": math.log(
        smoothing / (number_of_words + smoothing * (vocabulary_size + 1)))}

    for word in word_counter:
        counter = 0
        # for each word count the number of documents it is in
        for d in document_dictionaries:
            if word in d:
                counter = counter + 1

        idf = math.log10(number_of_documents / counter)
        tf = word_counter[word]
        tf_idf_value = idf * tf
        log_prob_dictionary[word] = math.log(
            (tf_idf_value + smoothing) / (number_of_words + smoothing * (vocabulary_size + 1)))

    return log_prob_dictionary


class TfIdfNaiveBayesClassifier(object):

    def __init__(self, train_dir, test_dir, smoothing):
        self.train = load_files(train_dir)
        self.test = load_files(test_dir)
        self.smoothing = smoothing
        self.vocab_size = get_vocab_size(self.train)
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        self.number_of_documents = get_number_of_documents(self.train)
        self.document_dictionaries = {}

    def train_tfidf_nb(self):
        for category in self.train:
            label = os.path.basename(os.path.normpath((category)))

            # get the tokens and word dictionary representations of which documents includes which word
            tokens_and_doc_dictionary = get_tokens_from_category(category)

            # calculate conditional probability dictionary for each category
            self.conditional_probabilities[label] = log_probabilities_for_tfidf_NB(self.smoothing,
                                                                                   self.vocab_size,
                                                                                   tokens_and_doc_dictionary[
                                                                                       0],
                                                                                   tokens_and_doc_dictionary[
                                                                                       1],
                                                                                   number_of_documents_in_a_category(
                                                                                       category))
            # Map each category name to its prior probability
            self.prior_probabilities[label] = math.log(
                number_of_documents_in_a_category(category) / self.number_of_documents)

        # returns class with the highest probability

    def get_most_probable_class(self, word_counter):

        final_class_probabilities = {}

        for category in self.train:
            label = os.path.basename(os.path.normpath((category)))

            class_cond_prob_dictionary = self.conditional_probabilities[label]  # get conditional probability dict

            class_prior_probability = self.prior_probabilities[label]  # get prior probability of the label

            # calculate log (P(C) + sum of log (P (w|C)^count(w) where count(w): the number of times word w occurs in doc
            final_class_probability = reduce(
                lambda x, word: x + class_cond_prob_dictionary[word] * word_counter[word]
                if word in class_cond_prob_dictionary
                else x + class_cond_prob_dictionary["<UNK>"] * word_counter[word],
                word_counter, class_prior_probability)

            final_class_probabilities[label] = final_class_probability

        return max(final_class_probabilities.items(), key=operator.itemgetter(1))[0]

    def test_tfidf_nb(self):
        print("Testing tf-idf Naive Bayes Model....")

        correct = 0
        for category in self.test:
            for filename in os.listdir(category):

                tokens = get_tokens_from_single_file(category + "/" + filename)
                word_counter = Counter(tokens)

                predicted_class = self.get_most_probable_class(word_counter)
                if predicted_class == os.path.basename(os.path.normpath((category))):
                    correct = correct + 1

        print("Accuracy of tf-IDF Naive Bayes Classifier: {}".
              format((correct) / float(get_number_of_documents(self.test))))
        return correct / float(get_number_of_documents(self.test))


# Uncomment the following to test model on bydate data
NB = TfIdfNaiveBayesClassifier("20news-data/20news-bydate-rm-metadata/train",
                               "20news-data/20news-bydate-rm-metadata/test",
                               0.08)
NB.train_tfidf_nb()
result = NB.test_tfidf_nb()

# Uncomment the following to test model on random data
"""NB = TfIdfNaiveBayesClassifier("20news-data/20news-random-rm-metadata/train",
                               "20news-data/20news-random-rm-metadata/test",
                               0.08)
NB.train_tfidf_nb()
result = NB.test_tfidf_nb()"""
