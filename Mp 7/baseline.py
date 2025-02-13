"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_set = {}
    tag_count_set = {}

    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_set:
                word_tag_set[word] = {}
            if tag not in word_tag_set[word]:
                word_tag_set[word][tag] = 1
            else:
                word_tag_set[word][tag] += 1

            if tag not in tag_count_set:
                tag_count_set[tag] = 1
            else:
                tag_count_set[tag] += 1
    
	
    most_appered_tag = max(tag_count_set, key = tag_count_set.get)
    word_tag_most_appear = {}
    for word in word_tag_set:
        word_tag_most_appear[word] = max(word_tag_set[word], key = word_tag_set[word].get)
    
    ans = []
    for sentence in test:
        ans_word = []
        for word in sentence:
            if word not in word_tag_most_appear:
                ans_word.append((word, most_appered_tag))
            else:
                ans_word.append((word, word_tag_most_appear[word]))
        ans.append(ans_word)
    
    return ans