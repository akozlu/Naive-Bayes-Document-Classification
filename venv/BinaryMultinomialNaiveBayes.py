import heapq
import math
import operator
import os
import random
import re
import string
from collections import *
from datetime import datetime
from functools import *
from itertools import *

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
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


# Helper function to map every word in a document to 1 in single line.
class LazyDict(dict):
    def keylist(self, keys, value):
        for key in keys:
            self[key] = value


# gets vocabulary size of the training or test set
def get_vocab_size(train_dir):
    V = []
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

            V.append(words)

    tokens = list(chain(*V))
    vocab = Counter(tokens)
    print("The vocabulary size of current training set is {}".format(len(vocab)))
    return len(vocab)


def get_vocab_size_for_semi_supervisor_classifier(directory, train_dictionary):
    V = []

    for category in directory:

        for document in train_dictionary[category]:
            file = open(document, 'rt')
            text = file.read()
            file.close()

            words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
            words = [word.lower() for word in words]  # lowercase
            words = [word for word in words if
                     word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

            stop_words = stopwords.words('english')  # filter out tokens that are stop words.
            words = [w for w in words if (w not in stop_words and len(w) != 1)]
            V.append(words)

    tokens = list(chain(*V))
    vocab = Counter(tokens)
    print("The vocabulary size of current training set is {}".format(len(vocab)))
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

    for filename in os.listdir(category):
        file = open(category + "/" + filename, 'rt')
        text = file.read()
        file.close()

        words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
        words = [word.lower() for word in words]  # lowercase
        words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

        stop_words = stopwords.words('english')  # filter out tokens that are stop words.
        words = [w for w in words if (w not in stop_words and len(w) != 1)]
        document_words = list(set(words))

        d = LazyDict()
        d.keylist(document_words, 1)
        document_dictionaries.append(d)
        V.append(words)

    tokens = list(chain(*V))
    print("Extracted a total number of {} (non-unique) words".format(len(tokens)))

    return [tokens, document_dictionaries]


def get_tokens_from_category_for_semi_supervisor_classifier(category, train_dictionary):
    # This is a seperate function because we do not iterate over ALL documents in a file
    # print("Extracting vocabulary from the category: " + os.path.basename(os.path.normpath((category))))
    document_dictionaries = []
    V = []
    for document in train_dictionary:
        file = open(document, 'rt')
        text = file.read()
        file.close()

        words = re.split(r'\W+', text)  # first filter. Select for string of alphanumeric characters
        words = [word.lower() for word in words]  # lowercase
        words = [word for word in words if word.isalpha()]  # Filter out remaining tokens that are not alphabetic.

        stop_words = stopwords.words('english')  # filter out tokens that are stop words.
        words = [w for w in words if (w not in stop_words and len(w) != 1)]
        document_words = list(set(words))

        d = LazyDict()
        d.keylist(document_words, 1)  # map all the words in the document to 1.
        document_dictionaries.append(d)  # add it to a dictionary that will represent each document
        V.append(words)

    tokens = list(chain(*V))

    return [tokens, document_dictionaries]


# Function that maps each word in document category to its conditional probability given that category
def log_probabilities_for_binary_NB(smoothing,
                                    vocabulary_size, words,
                                    document_dictionaries):  # Conditional Probabilities for multinomial NB

    word_counter = Counter(words)  # get each unique word

    number_of_words = 0  # This will be the sum of f[D,j]
    for d in document_dictionaries:
        number_of_words = number_of_words + len(d.keys())
    # add a new token <UNK> for Unknown, that occurs nowhere in tokens.
    # This will map all unknown words in documents to be classified (test documents) to UNK

    log_prob_dictionary = {"<UNK>": math.log(
        float(smoothing) / float(number_of_words + smoothing * (vocabulary_size + 1)))}

    for word in word_counter:
        counter = 0
        # for each word count the number of documents it is in
        for d in document_dictionaries:
            if word in d:
                counter = counter + 1

        # calculate its conditional probability
        log_prob_dictionary[word] = math.log(
            float(counter + smoothing) / float(number_of_words + (smoothing * (vocabulary_size + 1))))

    return log_prob_dictionary


class BinaryMultinomialNaiveBayesClassifier(object):

    def __init__(self, train_dir, test_dir, smoothing, p, filter):
        self.train = load_files(train_dir)
        self.test = load_files(test_dir)
        self.smoothing = smoothing
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        self.number_of_documents = get_number_of_documents(self.train)
        self.p = p
        self.filter = filter
        self.semi_supervisor_training_dict = {}
        self.semi_supervisor_testing_dict = {}

    # Trains the Binary Multinomial Naive Bayes Model
    def train_binary_nb(self):
        vocab_size = get_vocab_size(self.train)
        for category in self.train:
            label = os.path.basename(os.path.normpath((category)))

            # get the tokens and word dictionary representations of which documents includes which word
            tokens_and_doc_dictionary = get_tokens_from_category(category)

            # calculate conditional probability dictionary for each category
            self.conditional_probabilities[label] = log_probabilities_for_binary_NB(self.smoothing,
                                                                                    vocab_size,
                                                                                    tokens_and_doc_dictionary[
                                                                                        0],
                                                                                    tokens_and_doc_dictionary[
                                                                                        1])
            # Map each category name to its prior probability
            self.prior_probabilities[label] = math.log(float(
                number_of_documents_in_a_category(category) / self.number_of_documents))

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

    # This function tests our model on testing data and reports the accuracy.
    def test_binary_nb(self):
        print("Testing the b-Bow Naive Bayes Model....")

        correct = 0
        for category in self.test:
            for filename in os.listdir(category):

                tokens = get_tokens_from_single_file(category + "/" + filename)
                word_counter = Counter(tokens)

                predicted_class = self.get_most_probable_class(word_counter)
                if predicted_class == os.path.basename(os.path.normpath((category))):
                    correct = correct + 1
        print(
            "Accuracy of B-bow Naive Bayes Classifier {}".format((correct) / float(get_number_of_documents(self.test))))

        return (correct) / float(get_number_of_documents(self.test))

    "The rest of the functions are dedicated to semi-supervised learning"

    # This function builds up a new training dictionary, taking % 5, % 10 or %50 of training data.
    # The rest of the documents are allocated to another test dictionary.
    def build_semi_supervisor_classifier(self):

        if self.p == 5:
            # We create a new training dictionary where each category is mapped to randomly selected %5 of documents names in the category.
            self.semi_supervisor_training_dict = {
                self.train[i]: [self.train[i] + "/" + s for s in (random.sample(os.listdir(self.train[i]), int((
                        number_of_documents_in_a_category(self.train[i]) * 0.05))))] for i in (range(len(self.train)))}

            # The test dictionary will map each label to every document in it that is not in the training dictionary.
            self.semi_supervisor_testing_dict = {self.train[i]:
                                                     [self.train[i] + "/" + os.listdir(self.train[i])[j] for j in
                                                      range(len(os.listdir
                                                                (self.train[i]))) if
                                                      self.train[i] + "/" + os.listdir(self.train[i])[j] not in
                                                      self.semi_supervisor_training_dict[
                                                          self.train[i]]] for i in
                                                 (range(len(self.train)))}

        if self.p == 10:
            self.semi_supervisor_training_dict = {
                self.train[i]: [self.train[i] + "/" + s for s in (random.sample(os.listdir(self.train[i]), int((
                        number_of_documents_in_a_category(self.train[i]) * 0.1))))] for i in (range(len(self.train)))}

            self.semi_supervisor_testing_dict = {self.train[i]:
                                                     [self.train[i] + "/" + os.listdir(self.train[i])[j] for j in
                                                      range(len(os.listdir
                                                                (self.train[i]))) if
                                                      self.train[i] + "/" + os.listdir(self.train[i])[j] not in
                                                      self.semi_supervisor_training_dict[
                                                          self.train[i]]] for i in
                                                 (range(len(self.train)))}
        if self.p == 50:
            self.semi_supervisor_training_dict = {
                self.train[i]: [self.train[i] + "/" + s for s in (random.sample(os.listdir(self.train[i]), int((
                        number_of_documents_in_a_category(self.train[i]) * 0.5))))] for i in (range(len(self.train)))}

            self.semi_supervisor_testing_dict = {self.train[i]:
                                                     [self.train[i] + "/" + os.listdir(self.train[i])[j] for j in
                                                      range(len(os.listdir
                                                                (self.train[i]))) if
                                                      self.train[i] + "/" + os.listdir(self.train[i])[j] not in
                                                      self.semi_supervisor_training_dict[
                                                          self.train[i]]] for i in
                                                 (range(len(self.train)))}

    # This function is responsible for training, testing and then updating the semi supervised Naive Bayes Classifier.
    def train_semi_supervisor_classifier(self):

        # get total number of documents currently in our training set
        number_of_training_documents = reduce(lambda x, label: x + len(self.semi_supervisor_training_dict[label]),
                                              self.train, 0)

        number_of_testing_documents = reduce(lambda x, label: x + len(self.semi_supervisor_testing_dict[label]),
                                             self.train, 0)
        print("Number of documents in training set: {}".format(number_of_training_documents))
        print("Number of documents in test set: {}".format(number_of_testing_documents))

        print("Extracting tokens from 20 categories...")
        vocab_size = get_vocab_size_for_semi_supervisor_classifier(self.train, self.semi_supervisor_training_dict)

        # Constructing our model
        for category in self.semi_supervisor_training_dict:
            # get the tokens from set of documents. Also get dictionary of dictionaries that represent which document contains which word.
            tokens_and_doc_dictionary = get_tokens_from_category_for_semi_supervisor_classifier(category,
                                                                                                self.semi_supervisor_training_dict[
                                                                                                    category])

            # calculate conditional probability table for each category. Map each word in the category to its conditional probability given that label.
            self.conditional_probabilities[category] = log_probabilities_for_binary_NB(self.smoothing,
                                                                                       vocab_size,
                                                                                       tokens_and_doc_dictionary[
                                                                                           0],
                                                                                       tokens_and_doc_dictionary[
                                                                                           1])
            # calculate prior probability for each category.
            self.prior_probabilities[category] = math.log(float
                                                          (len(self.semi_supervisor_training_dict[
                                                                   category]) / number_of_training_documents))

        # Now that our model is set up, we test our current model on test data and report the accuracy
        test_accuracy = self.test_semi_supervisor_classifier()

        U_Y_dictionary = {}  # This will map highest value (label,document name) object to its conditional probability
        Document = namedtuple("Document", ["predicted_label", "document_path"])

        # Now we can update the model. We get new predictions from our test set and add those predictions to a dictionary.
        for category in self.semi_supervisor_testing_dict:

            documents = self.semi_supervisor_testing_dict[category]

            # for each document, get its tokens and predict its label.

            for doc in documents:
                tokens = get_tokens_from_single_file(doc)
                word_counter = Counter(tokens)
                result = self.predict_label_for_semi_supervisor(word_counter, Document,
                                                                doc)  # get the label with highest probability
                U_Y_dictionary[result[0]] = result[
                    1]  # result[0] stores the matched category and the document name, result[1] stores the calculated log probability

        # Select Top-k probability documents and add them to the predicted label dictionary
        if self.filter == "top-k":
            k = 500

            if len(U_Y_dictionary) > k:

                selected_documents = heapq.nlargest(k, U_Y_dictionary, key=U_Y_dictionary.get)
            else:

                selected_documents = heapq.nlargest(len(U_Y_dictionary), U_Y_dictionary, key=U_Y_dictionary.get)

            # Check out how many of these new documents were actually correctly labelled.
            correct_labelings = 0
            for d in selected_documents:
                if (d.predicted_label == os.path.dirname(d.document_path)):
                    correct_labelings = correct_labelings + 1
            print(
                "Out of {} new documents added to the training set, {} were added according to their correct label".format(
                    k, correct_labelings))

            # Update training and test dictionaries with augmented instances
            for d in selected_documents:
                train_docs = self.semi_supervisor_training_dict[d.predicted_label]
                train_docs.append(d.document_path)
                self.semi_supervisor_training_dict[d.predicted_label] = train_docs
                test_docs = self.semi_supervisor_testing_dict[os.path.dirname(d.document_path)]
                test_docs.remove(d.document_path)
                self.semi_supervisor_testing_dict[os.path.dirname(d.document_path)] = test_docs

        if self.filter == 'threshold':
            threshold = - 500
            # Augment  those  instances  to  the  labeled  data  with confidence higher than this threshold
            selected_documents = {k: v for (k, v) in U_Y_dictionary.items() if v > threshold}

            # If none of the documents have confidence higher than the threshold, terminate.
            if len(selected_documents) == 0:
                return None
            else:
                print(len(selected_documents))
                # Check out how many of these new documents were actually correctly labelled.

                correct_labelings = 0
                for d in selected_documents:
                    if (d.predicted_label == os.path.dirname(d.document_path)):
                        correct_labelings = correct_labelings + 1
                print(
                    "Out of {} new documents added to the training set, {} were added according to their correct label".format(
                        len(selected_documents), correct_labelings))

                # Update training and test dictionaries with augmented instances
                for d in selected_documents:
                    train_docs = self.semi_supervisor_training_dict[d.predicted_label]
                    train_docs.append(d.document_path)
                    self.semi_supervisor_training_dict[d.predicted_label] = train_docs
                    test_docs = self.semi_supervisor_testing_dict[os.path.dirname(d.document_path)]
                    test_docs.remove(d.document_path)
                    self.semi_supervisor_testing_dict[os.path.dirname(d.document_path)] = test_docs

        return test_accuracy

    # Returns predicted label with its associated logarithmic conditional probability value.
    # The labels are returned as Document Object Tuples, where category is the predicted label and document_path is document file path.
    def predict_label_for_semi_supervisor(self, word_counter, Document, current_document):
        final_class_probabilities = {}

        for category in self.semi_supervisor_training_dict:
            class_cond_prob_dictionary = self.conditional_probabilities[category]  # get conditional probability dict

            class_prior_probability = self.prior_probabilities[category]  # get prior probability of the label

            # calculate log (P(C) + sum of log (P (w|C)^count(w) where count(w): the number of times word w occurs in doc
            final_class_probability = reduce(
                lambda x, word: x + class_cond_prob_dictionary[word] * word_counter[word]
                if word in class_cond_prob_dictionary
                else x + class_cond_prob_dictionary["<UNK>"] * word_counter[word],
                word_counter, class_prior_probability)

            curr_doc = Document(predicted_label=category, document_path=current_document)
            final_class_probabilities[curr_doc] = final_class_probability

        predicted_label = max(final_class_probabilities.items(), key=operator.itemgetter(1))[0]
        log_probability_of_label = final_class_probabilities[predicted_label]

        return [predicted_label, log_probability_of_label]

    # Tests Semi Supervisor Classifier on test data
    def test_semi_supervisor_classifier(self):
        print("Testing Semi-Supervised b-Bow Naive Bayes Model....")
        correct = 0
        for category in self.test:
            print("Testing documents in category: " + os.path.basename(os.path.normpath((category))))
            for filename in os.listdir(category):

                tokens = get_tokens_from_single_file(category + "/" + filename)
                word_counter = Counter(tokens)
                final_class_probabilities = {}
                for c in self.semi_supervisor_training_dict:
                    class_cond_prob_dictionary = self.conditional_probabilities[
                        c]  # get conditional probability dict

                    class_prior_probability = self.prior_probabilities[c]  # get prior probability of the label

                    # calculate log (P(C) + sum of log (P (w|C)^count(w) where count(w): the number of times word w occurs in doc
                    final_class_probability = reduce(
                        lambda x, word: x + class_cond_prob_dictionary[word] * word_counter[word]
                        if word in class_cond_prob_dictionary
                        else x + class_cond_prob_dictionary["<UNK>"] * word_counter[word],
                        word_counter, class_prior_probability)
                    final_class_probabilities[c] = final_class_probability
                # result = self.get_most_probable_class_for_semi_supervisor(word_counter, Document,
                #                                                       category + "/" + filename)
                predicted_label = max(final_class_probabilities.items(), key=operator.itemgetter(1))[0]

                if (os.path.basename(os.path.normpath((predicted_label)))) == (os.path.basename(
                        os.path.normpath((category)))):
                    correct = correct + 1
        print("Accuracy of Semi Supervisor B-bow Naive Bayes Classifier {}".format(
            (correct) / float(get_number_of_documents(self.test))))

        return correct / float(get_number_of_documents(self.test))

    # The main function were the semi supervised classifier learns through calling train_semi_supervisor_classifier()
    def iterate_semi_supervised_classifier(self, filter):

        if filter == 'top-k':

            # Build the training and unseen data sets

            self.build_semi_supervisor_classifier()

            test_dictionary_list = [d for d in self.semi_supervisor_testing_dict]

            iteration = 0

            accuracy_scores = []
            # While we still have documents on unseen data
            while (not all(len(self.semi_supervisor_testing_dict[d]) == 0 for d in test_dictionary_list)):
                acc = self.train_semi_supervisor_classifier()  # keep training the classifier and return the test accuracy
                iteration = iteration + 1
                accuracy_scores.append(acc)

            # Plot the test accuracy on each iteration
            xi = [i for i in range(1, iteration + 1)]
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy of Test Data in %')

            plt.plot(xi, accuracy_scores, marker='.', linestyle='-')
            plt.legend()
            plt.show()

        #Threshold filter
        else:

            accuracy_scores = []
            iteration = 1
            self.build_semi_supervisor_classifier()

            acc = self.train_semi_supervisor_classifier()
            accuracy_scores.append(acc)
            while(acc is not None):

                new_acc = self.train_semi_supervisor_classifier()

                acc = new_acc
                iteration = iteration + 1
                accuracy_scores.append(acc)

            # Plot the test accuracy on each iteration
            xi = [i for i in range(1, iteration + 1)]
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy of Test Data in %')

            plt.plot(xi, accuracy_scores, marker='.', linestyle='-')
            plt.legend()
            plt.show()


#RUNNING THE SCRIPT


# Specify top-k or threshold as the filter.
# Fourth argument refers to percentage of data set that will be allocated to training. Options are 5,10 or 50
# Uncommenting following lines will train and test the B-bow model using supervised learning and top-k as filter.

filter = 'top-k'
NB_top_k = BinaryMultinomialNaiveBayesClassifier("20news-data/20news-bydate-rm-metadata/train",
                                           "20news-data/20news-bydate-rm-metadata/test",
                                           0.08, 10, 'top-k')

NB_top_k.iterate_semi_supervised_classifier(filter)

# Uncommenting following lines will train and test the B-bow model using supervised learning and threshold as filter.
"""
filter = 'threshold'

NB_threshold = BinaryMultinomialNaiveBayesClassifier("20news-data/20news-bydate-rm-metadata/train",
                                                     "20news-data/20news-bydate-rm-metadata/test",
                                                     0.08, 10, "threshold")
NB_threshold.iterate_semi_supervised_classifier('threshold')
"""
# Uncommenting following lines will train and test the B-Bow Naive Bayes Model on bydate data

"""
NB1 = BinaryMultinomialNaiveBayesClassifier("20news-data/20news-bydate-rm-metadata/train",
                                            "20news-data/20news-bydate-rm-metadata/test",
                                            0.08, 50,'top-k')
NB1.train_binary_nb()
result = NB1.test_binary_nb()
"""

# Uncommenting following lines will train and test the B-Bow Naive Bayes Model on random data
"""
NB2 = BinaryMultinomialNaiveBayesClassifier("20news-data/20news-random-rm-metadata/train",
                                            "20news-data/20news-random-rm-metadata/test",
                                            0.08, 50,'top-k')
NB2.train_binary_nb()
result = NB2.test_binary_nb()
"""
