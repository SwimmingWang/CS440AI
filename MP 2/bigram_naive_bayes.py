# bigram_naive_bayes.py
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
import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')

'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    stop_words = set(stopwords.words('english'))
    if bigram_lambda >= 0.8:
        bigram_lambda = 0.7*bigram_lambda
    if pos_prior < 0.5:
        train_set = [[word for word in doc if word not in stop_words] for doc in train_set]
        dev_set = [[word for word in doc if word not in stop_words] for doc in dev_set]
    # training unigram
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
    
    # training bigram
    pos_bigram_list = Counter()
    neg_bigram_list = Counter()
    pos_bigram_num, neg_bigram_num = 0, 0
    bigram_vocab = set()

    for i in range(len(train_labels)):
        bigrams = [(train_set[i][j], train_set[i][j+1]) for j in range(len(train_set[i])-1)]
        bigram_vocab.update(bigrams)
        if train_labels[i] == 1:
            pos_bigram_list.update(bigrams)
            pos_bigram_num += (len(train_set[i])-1)
        else:
            neg_bigram_list.update(bigrams)
            neg_bigram_num += (len(train_set[i])-1)

    bigram_vocab_num = len(bigram_vocab)

    # prediction
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        # unigram
        word_pro_pos = math.log(pos_prior)
        word_pro_neg = math.log(1 - pos_prior)
        for word in doc:
            if word in vocab:
                prob_pos = (pos_list[word] + unigram_laplace) / (pos_num + unigram_laplace * (vocab_num + 1))
                prob_neg = (neg_list[word] + unigram_laplace) / (neg_num + unigram_laplace * (vocab_num + 1))
            else:
                prob_pos = unigram_laplace / (pos_num + unigram_laplace * (vocab_num + 1))
                prob_neg = unigram_laplace / (neg_num + unigram_laplace * (vocab_num + 1))
            word_pro_pos += math.log(prob_pos)
            word_pro_neg += math.log(prob_neg)

        # bigram
        wordset_pro_pos = math.log(pos_prior)
        wordset_pro_neg = math.log(1 - pos_prior)

        bigrams = [(doc[j], doc[j+1]) for j in range(len(doc)-1)]
        for bigram in bigrams:
            if bigram in bigram_vocab:
                prob_bigram_pos = (pos_bigram_list[bigram] + bigram_laplace) / (pos_bigram_num + bigram_laplace * (bigram_vocab_num + 1))
                prob_bigram_neg = (neg_bigram_list[bigram] + bigram_laplace) / (neg_bigram_num + bigram_laplace * (bigram_vocab_num + 1))
            else:
                prob_bigram_pos = bigram_laplace / (pos_bigram_num + bigram_laplace * (bigram_vocab_num + 1))
                prob_bigram_neg = bigram_laplace / (neg_bigram_num + bigram_laplace * (bigram_vocab_num + 1))
            wordset_pro_pos += math.log(prob_bigram_pos)
            wordset_pro_neg += math.log(prob_bigram_neg)

        # combine unigram and bigram
        log_pro_pos = (1-bigram_lambda) * word_pro_pos + bigram_lambda * wordset_pro_pos
        log_pro_neg = (1-bigram_lambda) * word_pro_neg + bigram_lambda * wordset_pro_neg
        
        yhats.append(1 if log_pro_pos >= log_pro_neg else 0)

    return yhats