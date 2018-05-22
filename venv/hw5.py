import math
import operator
import os
import re
import string
from collections import *
from datetime import datetime
from functools import *
from itertools import *

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# function to access the categories in training directory
def load_files(file_path):
    train_dir = os.listdir(file_path)
    train = [file_path + "/" + f for f in train_dir if f != '.DS_Store']
    return train


# Function that extracts words from a single file for testing purposes
def get_tokens_from_single_file(file_path):
    file = open(file_path, 'rt')
    text = file.read()
    file.close()
    words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
    words = [word.lower() for word in words]  # lowercase
    words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

    stop_words = stopwords.words('english')  # filter out tokens that are stop words.
    words = [w for w in words if (w not in stop_words and len(w) != 1)]

    return words


# Function that extracts words from a single category that are selected after applying a selection filters
def get_tokens_from_category(category):
    print("Extracting vocabulary from category: " + os.path.basename(os.path.normpath((category))))

    V = []
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
        wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
        # words = [lemmatizer.lemmatize(w, pos=wnpos(pos_tag(w)[0][1])) for w in words if
        #    (((pos_tag(w)[0][1])[0]) in ['J', 'N', 'R', 'V'])]

        V.append(words)

    tokens = list(chain(*V))

    return tokens


# This function gets vocabulary size of the training or test set
def get_vocab_size(train_dir):
    V = []
    for category in train_dir:
        for filename in os.listdir(category):
            file = open(category + "/" + filename, 'rt')
            text = file.read()
            file.close()

            words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
            words = [word.lower() for word in words]  # lowercase
            words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

            stop_words = stopwords.words('english')  # filter out tokens that are stop words.
            words = [w for w in words if (w not in stop_words and len(w) != 1)]

            wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
            # words = [lemmatizer.lemmatize(w, pos=wnpos(pos_tag(w)[0][1])) for w in words if
            #        (((pos_tag(w)[0][1])[0]) in ['J', 'N', 'R', 'V'])]

            V.append(words)

    tokens = list(chain(*V))
    vocab = Counter(tokens)
    print("Vocabulary Size:")
    print(len(vocab))
    return len(vocab)


# This function gets number of total documents (text files) in a single category.

def number_of_documents_in_a_category(category):
    return (len(os.listdir(category)))


# Function that calculates P (w_i | C) for a given class and maps each word to its conditional probability.
def log_probabilities_for_multinomial_NB(category, smoothing,
                                         vocabulary_size):  # Conditional Probabilities for multinomial NB

    tokens = get_tokens_from_category(category)  # extract words from all documents in that class
    number_of_words = len(tokens)

    word_counter = Counter(tokens)  # maps each word to its frequency

    # add a new token <UNK> for Unknown, that occurs nowhere in tokens.
    # This will map all unknown words in documents to be classified (test documents) to UNK

    log_prob_dictionary = {"<UNK>": math.log(
        smoothing / (number_of_words + smoothing * (vocabulary_size + 1)))}

    # Now we calculate each P(f[d, i]|y) in log space and add it to a dictionary that maps the word to its probability
    for word in word_counter:
        log_prob_dictionary[word] = math.log(
            (word_counter[word] + smoothing) / (number_of_words + smoothing * (vocabulary_size + 1)))

    return log_prob_dictionary


# Get total number of documents
def get_number_of_documents(directories):
    counter = 0
    for category in directories:
        counter = counter + len(os.listdir(category))  # dir is your directory path

    return counter


class MultinomialNaiveBayesClassifier(object):

    def __init__(self, train_dir, test_dir, smoothing):
        self.train = load_files(train_dir)
        self.test = load_files(test_dir)
        self.smoothing = smoothing
        self.vocab_size = get_vocab_size(self.train)
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        self.number_of_documents = get_number_of_documents(self.train)

    # Train C_BOW Model
    def train_multinomial_nb(self):
        print("Total number of training documents:")
        print(self.number_of_documents)

        for category in self.train:
            category_name = os.path.basename(os.path.normpath((category)))
            # Map category name to its conditional probability dictionary, which maps every word in it to the calculated probability
            self.conditional_probabilities[category_name] = log_probabilities_for_multinomial_NB(category,
                                                                                                 self.smoothing,
                                                                                                 self.vocab_size)
            # Map each category name to its prior probability

            self.prior_probabilities[category_name] = math.log(
                number_of_documents_in_a_category(category) / self.number_of_documents)

    # returns class with the highest probability
    def get_most_probable_class(self, word_counter):

        final_class_probabilities = {}

        for category in self.train:
            category_name = os.path.basename(os.path.normpath((category)))

            class_cond_prob_dictionary = self.conditional_probabilities[
                category_name]  # get conditional probability dict

            class_prior_probability = self.prior_probabilities[category_name]  # get prior probability of the class

            # calculate log (P(C) + sum of log (P (w|C)^count(w) where count(w): the number of times word w occurs in doc
            final_class_probability = reduce(lambda x, word: x + class_cond_prob_dictionary[word] * word_counter[word]
            if word in class_cond_prob_dictionary
            else x + class_cond_prob_dictionary["<UNK>"] * word_counter[word],
                                             word_counter, class_prior_probability)

            final_class_probabilities[category_name] = final_class_probability

        return max(final_class_probabilities.items(), key=operator.itemgetter(1))[0]

    def test_multinomial_nb(self):
        print("Testing c-Bow Naive Bayes Model....")

        correct = 0
        for category in self.test:
            for filename in os.listdir(category):

                tokens = get_tokens_from_single_file(category + "/" + filename)
                word_counter = Counter(tokens)

                predicted_class = self.get_most_probable_class(word_counter)
                if predicted_class == os.path.basename(os.path.normpath(category)):
                    correct = correct + 1
        print("Accuracy of C-Bow Naive Bayes Classifier on test data: {}".format(
            correct / float(get_number_of_documents(self.test))))
        return correct / float(get_number_of_documents(self.test))


# Uncomment following lines to train the model on bydate data


NB = MultinomialNaiveBayesClassifier("20news-data/20news-bydate-rm-metadata/train",
                                     "20news-data/20news-bydate-rm-metadata/test",
                                     0.08)

NB.train_multinomial_nb()
result = NB.test_multinomial_nb()

# Uncomment following lines to train the model on random data
"""
NB = MultinomialNaiveBayesClassifier("20news-data/20news-random-rm-metadata/train",
                                     "20news-data/20news-random-rm-metadata/test",
                                     0.08)

NB.train_multinomial_nb()
result = NB.test_multinomial_nb()
"""
