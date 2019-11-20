from __future__ import division
from collections import Counter
from codecs import open
import numpy as np

NEG_WORD_COUNT = Counter()
POS_WORD_COUNT = Counter()

"""
Task 0: We first remove the document identifier, and also the topic label, which you don't need. 
Then,split the data into a training and an evaluation part. 
For instance, wemay use 80% for training and the remainder for evaluation.
"""
def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    
    return docs, labels
"""
Task 1: Estimating parameters for the Naive Bayes classifier
"""
def train_nb(documents, labels):
    neg_word_count = Counter()
    pos_word_count = Counter()
    neg_total_word = 0
    pos_total_word = 0
    # we now create our classification
    classifier = list(zip(labels, documents))
    for c in classifier:
        if c[0] == 'neg':
            neg_word_count.update(c[1])
        else:
            pos_word_count.update(c[1])

    neg_total_word = sum(neg_word_count.values())
    pos_total_word = sum(pos_word_count.values())
    total_word = sum(neg_total_word, pos_total_word)
    return neg_total_word, pos_total_word, total_word, neg_word_count, pos_word_count

"""
Task 2: Classifying new document
"""
def score_doc_label(document, label, neg_total_word, pos_total_word, total_word, neg_word_count, pos_word_count):
    
    if label == 'neg':
        labelProb = neg_total_word / total_word

    elif label == 'pos':
        labelProb = pos_total_word / total_word

    else:
        print("INVALID LABEL")
        return 0

    outProb = np.log10(labelProb)

    for word in document:

        if label == 'neg':
            wordProb = neg_word_count[word] / neg_total_word

        elif label == 'pos':
            wordProb = pos_word_count[word] / pos_total_word

        outProb += wordProb

    #print(outProb)

    return outProb

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

NEG_TOTAL_WORD, POS_TOTAL_WORD, TOTAL_WORD_SUM, NEG_WORD_COUNT, POS_WORD_COUNT = train_nb(train_docs, train_labels)