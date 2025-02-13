# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""

def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace, pos_prior)

    # Training
    pos_list = Counter()
    neg_list = Counter()
    pos_num, neg_num = 0, 0
    vocab = set()

    for i in range(len(train_labels)):
        vocab.update(train_set[i])
        if train_labels[i] == 1:
            pos_list.update(train_set[i])
            pos_num += len(train_set[i])
        else:
            neg_list.update(train_set[i])
            neg_num += len(train_set[i])

    vocab_num = len(vocab)

    #Prediction

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        log_pro_pos = math.log(pos_prior)
        log_pro_neg = math.log(1 - pos_prior)
        for word in doc:
            if word in vocab:
                prob_pos = (pos_list[word] + laplace) / (pos_num + laplace * (vocab_num + 1))
                prob_neg = (neg_list[word] + laplace) / (neg_num + laplace * (vocab_num + 1))
            else:
                prob_pos = laplace / (pos_num + laplace * (vocab_num + 1))
                prob_neg = laplace / (neg_num + laplace * (vocab_num + 1))
            log_pro_pos += math.log(prob_pos)
            log_pro_neg += math.log(prob_neg)
        yhats.append(1 if log_pro_pos > log_pro_neg else 0)

    return yhats