from pprint import pprint
from time import time
import logging

import pandas as pd
import numpy as np
from numpy.random import permutation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

debug = False

def most_informative_features(vectorizer, clf, n=200):
    feature_names = vectorizer.get_feature_names()
    negative_coefs = sorted(zip(clf.coef_[1], feature_names), reverse=True)[:n/2]
    positive_coefs = sorted(zip(clf.coef_[2], feature_names), reverse=True)[:n/2]

    if debug:
        top = zip(positive_coefs, negative_coefs)
        print "\tpositive\t\t\tnegative\n"
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%-15s\t\t%-15s" % (fn_1, fn_2)

    return set([token for (coeff, token) in negative_coefs] + [token for (coeff, token) in positive_coefs])

parameters = {
    'vect__binary': (True, False),
    'vect__strip_accents': ('ascii', 'unicode', None),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2', None),
    'clf__alpha': (0.001, 0.01, 0.1, 0.5, 1, 2)
}

steps = 1
def printStep(str):
    global steps
    print "=" * 40
    print "\tstep", steps, ":", str
    print "=" * 40
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
                          encoding='utf-8'
                          )

    #filter incorrectly labelled rows (malformed csv row)
    dataset = dataset[dataset['airline_sentiment'].isin(["negative", "positive"])]

    # separate dataset into training and test sets
    split = 0.7
    size = int(round(split*len(dataset)))

    perm = permutation(len(dataset))
    perm_train= perm[:size]
    perm_test = perm[size:]

    train_inputs = dataset['text'][perm_train].values.astype('U')
    train_labels = dataset['airline_sentiment'][perm_train].values.astype('U')
    test_inputs  = dataset['text'][perm_test].values.astype('U')
    test_labels  = dataset['airline_sentiment'][perm_test].values.astype('U')

    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    printStep("Reading dataset")
    train_inputs, train_labels, test_inputs, test_labels = get_dataset("../data/Tweets_notag_noneutral.csv")

    printStep("Find optimal parameters with grid search")
    vect  = CountVectorizer()
    tfidf = TfidfTransformer()
    clf   = MultinomialNB()
    pipeline = Pipeline([('vect', vect),
                        ('tfidf', tfidf),
                        ('clf', clf)])

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1)

    print "Parameter options: ", parameters
    t0 = time()
    grid_search.fit(test_inputs, test_labels)
    print("done in %0.3fs" % (time() - t0))
    print ""
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Optimal parameters found:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: \t%r" % (param_name, best_parameters[param_name]))

    printStep("Creating multinomial NB model using optimal parameters")
    # fit using best parameters
    vect  = CountVectorizer()
    tfidf = TfidfTransformer()
    clf   = MultinomialNB()
    pipeline = Pipeline([('vect', vect),
                        ('tfidf', tfidf),
                        ('clf', clf)])
    # pipeline.set_params(**best_parameters)
    pipeline.fit(train_inputs, train_labels)

    step     = 50
    max      = 5000

    n_range  = range(step, max, step)
    best_n   = 0
    best_acc = 0
    accuracies = np.zeros(len(n_range))

    printStep("Determining number of N most informative features to consider")
    for n in n_range:
        # get n most informative features
        vocab = most_informative_features(vect, clf, n)

        # train on most informative features only
        n_vect = CountVectorizer(vocabulary=vocab)
        n_clf  = MultinomialNB()
        text_clf = Pipeline([('vect', n_vect),
                            ('tfidf', TfidfTransformer()),
                            ('clf', n_clf)])

        # text_clf.set_params(**best_parameters)

        # train model
        text_clf = text_clf.fit(train_inputs, train_labels)

        # test model
        predicted = text_clf.predict(test_inputs)
        accuracies[n/step - 1] = np.mean(predicted == test_labels)
        print "n = %d, accuracy = %lf" % (n, accuracies[n/step - 1])

        if (accuracies[n/step - 1] >= best_acc):
            best_acc = accuracies[n/step - 1]
            best_n = n

    printStep("Optimal value for N")
    print "N:", best_n, "Accuracy:", best_acc

    # TODO: plot data and results
    print zip(n_range, accuracies)
