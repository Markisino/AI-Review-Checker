from __future__ import division
from collections import Counter
from codecs import open

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
    # we now create our classification
    classifier = list(zip(labels, documents))
    for c in classifier:
        if c[0] == 'neg':
            neg_word_count.update(c[1])
        else:
            pos_word_count.update(c[1])
    print(len(documents))
    print(len(neg_word_count) + len(pos_word_count))

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

train_nb(train_docs,train_labels)