import math
from collections import defaultdict, Counter
from math import log

# Smoothing constants
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

def classify_word(word):
    """
    Classifies a word into one of six types based on its form.
    
    Types:
    1. START_NUM_END_NUM
    2. VERY_SHORT
    3. SHORT
    4. END_S
    5. END_OTHER
    6. LONG
    
    :param word: The word to classify.
    :return: A string label representing the word type.
    """
    if word[0].isdigit() and word[-1].isdigit():
        return 'START_NUM_END_NUM'
    elif len(word) <= 3:
        return 'VERY_SHORT'
    elif len(word) >= 10:
        if word.endswith('s'):
            return "LONG_S"
        else:
            return "LONG_OTHER"
    elif len(word) >= 4 and len(word) <= 9:
        if word.endswith('s'):
            return "SHORT_S"
        else:
            return "SHORT_OTHER"
    else:
        return 'END_OTHER'  # Default case

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities,
    and prepares smoothing parameters for Viterbi_3.
    
    :param sentences: Training data.
    :return: initial tag probs, emission probs, transition probs, alpha for each tag-type pair
    """
    init_prob = defaultdict(lambda: 0)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))  # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))  # {tag0:{tag1: # }}
    word_counts = Counter()
    
    tag_counts = defaultdict(int)
    total_words_for_tag = defaultdict(int)
    tag_tag_counts = defaultdict(lambda: defaultdict(int))
    hapax_tag_count = defaultdict(int)

    # First pass to count all word occurrences
    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] += 1
    
    hapax_words = {word for word, count in word_counts.items() if count == 1}
    tag_type_hapax_counts = defaultdict(lambda: defaultdict(int))  # {tag: {type: count}}
    total_hapax = 0
    
    for sentence in sentences:
        prev_tag = None
        for i, (word, tag) in enumerate(sentence):
            tag_counts[tag] += 1
            total_words_for_tag[tag] += 1
            if i == 0:
                init_prob[tag] += 1
    
            if prev_tag is not None:
                tag_tag_counts[prev_tag][tag] += 1
            emit_prob[tag][word] += 1
            prev_tag = tag
            
            # If hapax, count tag-type pair
            if word in hapax_words:
                word_type = classify_word(word)
                tag_type_hapax_counts[tag][word_type] += 1
                hapax_tag_count[tag] += 1
                total_hapax += 1
    
    number_sentence = len(sentences)
    len_tag_counts = len(tag_counts)
    V = len(word_counts)
    hapax_prob = {tag : count / total_hapax for tag, count in hapax_tag_count.items()}

    for tag in init_prob:
        init_prob[tag] /= number_sentence
    
    # Calculate emission probabilities with smoothing
    for tag, words in emit_prob.items():
        for word, count in words.items():
            alpha = epsilon_for_pt 
            emit_prob[tag][word] = (count + alpha) / (total_words_for_tag[tag] + alpha * (V + 1))
        emit_prob[tag]['UNKNOWN'] = alpha / (total_words_for_tag[tag] + alpha * (V + 1))
    
    for tag1, tags in tag_tag_counts.items():
        total_tag1 = sum(tags.values())
        for tag2, count in tags.items():
            trans_prob[tag1][tag2] = (count + epsilon_for_pt) / (total_tag1 + epsilon_for_pt * len_tag_counts)
    
    alpha_tag_type = defaultdict(lambda: defaultdict(float))  # {tag: {type: alpha}}
    for tag in tag_counts:
        for word_type in ['START_NUM_END_NUM', 'VERY_SHORT', 'SHORT_OTHER', 'SHORT_S', 'LONG_OTHER', 'LONG_S']:
            count = tag_type_hapax_counts[tag].get(word_type, emit_epsilon)
            alpha = epsilon_for_pt * (count / total_hapax)
            alpha_tag_type[tag][word_type] = alpha / (total_words_for_tag[tag] + alpha * (V + 1))
    
    return init_prob, emit_prob, trans_prob, alpha_tag_type

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, alpha_tag_type):

    log_prob = {}
    predict_tag_seq = {}
    word_type = classify_word(word)
    
    for curr_tag in emit_prob:
        max_log_prob = -math.inf
        best_tag_seq = []
        
        for prev_tag in prev_prob:
            trans_p = trans_prob.get(prev_tag, {}).get(curr_tag, epsilon_for_pt)
            
            if word not in emit_prob[curr_tag].keys():
                alpha = alpha_tag_type[curr_tag].get(word_type, epsilon_for_pt)
                emit_p = alpha
            else:
                emit_p = emit_prob[curr_tag].get(word, epsilon_for_pt)
            
            current_log_prob = prev_prob[prev_tag] + log(trans_p) + log(emit_p)
            
            if current_log_prob > max_log_prob:
                max_log_prob = current_log_prob
                best_tag_seq = prev_predict_tag_seq[prev_tag] + [curr_tag]
        
        log_prob[curr_tag] = max_log_prob
        predict_tag_seq[curr_tag] = best_tag_seq
    
    return log_prob, predict_tag_seq

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob, alpha_tag_type = training(train)
    
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
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob, alpha_tag_type)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction. 
        best_final_tag = max(log_prob, key=log_prob.get)
        best_tag_sequence = predict_tag_seq[best_final_tag]
        tagged_sentence = list(zip(sentence, best_tag_sequence))
        predicts.append(tagged_sentence)
    return predicts