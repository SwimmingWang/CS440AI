"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

import math
from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    tag_counts = defaultdict(int)
    total_words_for_tag = defaultdict(int)
    tag_tag_counts = defaultdict(lambda: defaultdict(int))
    word_count = defaultdict(int)
    hapax_tag_count = defaultdict(int)
    hapax_count = 0
    for sentence in sentences:
        prev_tag = None
        for i, (word, tag) in enumerate(sentence):
            word_count[word] += 1
            tag_counts[tag] += 1
            total_words_for_tag[tag] += 1
            if i == 0:
                if tag not in init_prob.keys():
                    init_prob[tag] = 0
                init_prob[tag] += 1

            if prev_tag is not None:
                tag_tag_counts[prev_tag][tag] += 1
            emit_prob[tag][word] = emit_prob[tag].get(word, 0) + 1
            prev_tag = tag
        if prev_tag:
            init_prob[prev_tag] += 1
    
    for sentence in sentences:
        for word, tag in sentence:
            if word_count[word] == 1:
                hapax_tag_count[tag] += 1
                hapax_count += 1

    number_sentence = len(sentences)
    len_tag_counts = len(tag_counts)
    V = len({word for sentence in sentences for word, _ in sentence})
    hapax_prob = {tag : count / hapax_count for tag, count in hapax_tag_count.items()}

    for tag, count in init_prob.items():
        init_prob[tag] = count/number_sentence

    for tag, words in emit_prob.items():
        for word, count in words.items():
            alpha = epsilon_for_pt * (hapax_prob.get(tag, emit_epsilon))
            emit_prob[tag][word] = (count + alpha) / (total_words_for_tag[tag] + alpha * (V + 1))
        emit_prob[tag]['UNKNOWN'] = alpha / (total_words_for_tag[tag] + alpha * (V + 1))

    for tag1, tags in tag_tag_counts.items():
        total_tag1 = sum(tags.values())
        for tag2, count in tags.items():
            trans_prob[tag1][tag2] = (count + epsilon_for_pt) / (total_tag1 + epsilon_for_pt * len_tag_counts)
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    for curr_tag in emit_prob:
        max_log_prob = -math.inf
        best_tag_seq = []

        for prev_tag in prev_prob:
            trans_p = trans_prob.get(prev_tag, {}).get(curr_tag, epsilon_for_pt)
            if word not in emit_prob[curr_tag].keys():
                emit_p = emit_prob[curr_tag].get('UNKNOWN', emit_epsilon)
            else:
                emit_p = emit_prob[curr_tag].get(word, emit_epsilon)

            current_log_prob = prev_prob[prev_tag] + log(trans_p) + log(emit_p)
            if current_log_prob > max_log_prob:
                max_log_prob = current_log_prob
                best_tag_seq = prev_predict_tag_seq[prev_tag] + [curr_tag]

        log_prob[curr_tag] = max_log_prob
        predict_tag_seq[curr_tag] = best_tag_seq
    return log_prob, predict_tag_seq

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction. 
        best_final_tag = max(log_prob, key=log_prob.get)
        best_tag_sequence = predict_tag_seq[best_final_tag]
        tagged_sentence = list(zip(sentence, best_tag_sequence))
        predicts.append(tagged_sentence)
    return predicts