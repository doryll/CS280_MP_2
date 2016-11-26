from pprint import pprint
from time import time
import logging

import pandas as pd
import numpy as np
from numpy.random import permutation
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
debug = False
parameters = {
    'vect__binary': (True, False),
    'vect__strip_accents': ('ascii', 'unicode', None),
    'vect__max_df': np.arange(0.5, 1, 0.1),
    'vect__min_df': np.arange(0, 0.5, 0.1),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2', None),
    'clf__alpha': (0.001, 0.01, 0.1, 0.5, 1, 2)
}

# parameters = {'vect__strip_accents': ('ascii', 'unicode')}

step = 50
n_max = 5000

def most_informative_features(feature_names, coeff1, coeff2, n=200):
    negative_coefs = sorted(zip(coeff1, feature_names), reverse=True)[:n/2]
    positive_coefs = sorted(zip(coeff2, feature_names), reverse=True)[:n/2]

    if debug:
        top = zip(positive_coefs, negative_coefs)
        print "\tpositive\t\t\tnegative\n"
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%-15s\t\t%-15s" % (fn_1, fn_2)

    return set([token for (coeff, token) in negative_coefs] + [token for (coeff, token) in positive_coefs])

steps = 1
def printStep(str):
    global steps
    l = len(str) + len(str)/10 + 10
    print "=" * l
    print "\tstep", steps, ":", str
    print "=" * l
    steps += 1

# use preprocessed dataset with the following filters:
# 	@ tags removed from tweets
# 		to remove biases towards airlines
# 	neutral-labelled rows removed
#		only consider positive/negative sentiments
def get_dataset(file):
    dataset = pd.read_csv(filepath_or_buffer=file,
                          sep=',',
                          usecols=["airline_sentiment", "text"],
                          encoding='utf-8',
                          verbose=True
                          )

    #filter incorrectly labelled rows (malformed csv row)
    dataset = dataset[dataset['airline_sentiment'].isin(["negative", "positive"])]
    # print set(dataset['airline_sentiment'])
    # separate dataset into training and test sets
    split = 0.7
    size = int(round(split*len(dataset)))

    perm = permutation(len(dataset))
    perm_train= perm[:size]
    perm_test = perm[size:]

    # train_inputs = dataset['text'][perm_train].values.astype('U')
    # train_labels = dataset['airline_sentiment'][perm_train].values.astype('U')
    # test_inputs  = dataset['text'][perm_test].values.astype('U')
    # test_labels  = dataset['airline_sentiment'][perm_test].values.astype('U')
    train_inputs = dataset['text'][perm_train].values.astype('U')
    train_labels = dataset['airline_sentiment'][perm_train].values.astype('U')
    test_inputs  = dataset['text'][perm_test].values.astype('U')
    test_labels  = dataset['airline_sentiment'][perm_test].values.astype('U')

    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    printStep("Reading dataset")
    train_inputs, train_labels, test_inputs, test_labels = get_dataset("../data/Tweets_notag_noneutral.csv")

    print "\nDataset Counts"
    print "-" * 15
    print "Training data:\t\t", len(train_inputs) + len(test_inputs)
    print "Positive labels:\t", len(train_labels[train_labels == "positive"]) + len(test_labels[test_labels == "positive"])
    print "Negative labels:\t", len(train_labels[train_labels == "negative"]) + len(test_labels[test_labels == "negative"])

    printStep("Find optimal parameters with grid search")
    pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB())])
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1)

    print "Parameter options: ", parameters
    grid_search.fit(test_inputs, test_labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Optimal parameters found:")
    best_parameters = grid_search.best_estimator_.get_params(True)
    for param_name in sorted(parameters.keys()):
        print("\t%s: \t%r" % (param_name, best_parameters[param_name]))

    printStep("Creating multinomial NB model using optimal parameters")
    # fit using best parameters
    pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB())])
    pipeline.set_params(**best_parameters)
    pipeline.fit(train_inputs, train_labels)

    features = pipeline.get_params()['vect'].get_feature_names()
    coeff1 = pipeline.get_params()['clf'].coef_[1]
    coeff2 = pipeline.get_params()['clf'].coef_[2]

    best_predicted = pipeline.predict(test_inputs)

    n_max = len(features)
    n_range  = range(step, n_max, step)
    best_n   = 0
    best_acc = 0
    accuracies = np.zeros(len(n_range))

    printStep("Determining number of N most informative features to consider")
    for n in n_range:
        # get n most informative features
        vocab = most_informative_features(features, coeff1, coeff2, n)

        # train on most informative features only
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB())])
        best_parameters['vect__vocabulary'] = vocab
        text_clf.set_params(**best_parameters)

        # train model
        text_clf = text_clf.fit(train_inputs, train_labels)

        # test model
        predicted = text_clf.predict(test_inputs)
        accuracies[n/step - 1] = np.mean(predicted == test_labels)

        if debug:
            print "n = %d, accuracy = %1.3lf" % (n, accuracies[n/step - 1])
        elif (n % 500 == 0):
            print "n = %d, accuracy = %1.3lf" % (n, accuracies[n/step - 1])

        if accuracies[n/step - 1] >= best_acc:
            best_acc = accuracies[n/step - 1]
            best_predicted = predicted
            best_n = n

    printStep("Determine optimal value for N")
    print "Using N most informative features:"
    print("\tN: %d \tAccuracy: %1.3lf%%" % (best_n, best_acc * 100))
    print "Using all features:"
    print("\tN: %d \tAccuracy: %1.3lf%%" % (len(features), grid_search.best_score_ * 100))

    # remove nan results
    temp           = test_labels[test_labels != "nan"]
    test_labels    = test_labels[temp != "nan"]
    best_predicted = best_predicted[temp != "nan"]
    temp           = best_predicted[best_predicted != "nan"]
    test_labels    = test_labels[temp != "nan"]
    best_predicted = best_predicted[temp != "nan"]

    precision, recall, f_score, support = precision_recall_fscore_support(test_labels, best_predicted, labels=["positive", "negative"], average='micro')
    printStep("Plotting scores")
    plt.clf()

    print("precision\t%1.3lf%%"   % (precision * 100))
    print("recall\t\t%1.3lf%%"    % (recall * 100))
    print("f_score\t\t%1.3lf%%"   % (f_score * 100))

    #TODO: plot line for horizontal acc
    plt.plot(n_range, accuracies, color='navy',
             label='Vocabulary Size-Accuracy curve')
    # hline_data = np.array([best_acc for i in n_range])
    # plt.plot(n_range, hline_data, 'r--')

    plt.axhline(y=best_acc, xmin=0, xmax=n_max, hold=None)
    plt.axvline(x=best_n)

    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy')
    plt.ylim([0.7, 1.])
    plt.xlim([0.0, n_max])
    plt.legend(loc="lower left")
    plt.show()
