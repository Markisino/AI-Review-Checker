from __future__ import division
from collections import Counter
from codecs import open
import numpy as np

NEG_WORD_COUNT = Counter()
POS_WORD_COUNT = Counter()

"""
Task 0: We first remove the document identifier, and also the topic label, which you don't need. 
Then,split the data into a training and an evaluation part. 
For instance, we may use 80% for training and the remainder for evaluation.
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

    neg_total_word = sum(neg_word_count.values())
    pos_total_word = sum(pos_word_count.values())
    total_word = neg_total_word + pos_total_word

    return neg_total_word, pos_total_word, total_word, neg_word_count, pos_word_count


"""
Task 2: Classifying new document
"""


def score_doc_label(document):

    negProb = np.log10(NEG_TOTAL_WORD / TOTAL_WORD_SUM)
    posProb = np.log10(POS_TOTAL_WORD / TOTAL_WORD_SUM)

    for word in document:

        # Infinity checker
        if NEG_WORD_COUNT[word] == 0:
            negProb = -np.Inf

        if POS_WORD_COUNT[word] == 0:
            posProb = -np.Inf

        # Add if log10 if not infinity
        if negProb != -np.inf:
            negProb += np.log10(NEG_WORD_COUNT[word] / NEG_TOTAL_WORD)

        if posProb != -np.inf:
            posProb += np.log10(POS_WORD_COUNT[word] / POS_TOTAL_WORD)

    return np.exp(negProb), np.exp(posProb)

def classify_nb(document):

    negProb, posProb = score_doc_label(document)

    out = str()

    if negProb > posProb:
        out = "neg"

    else:
        out = "pos"

    return out

"""
Task 3: Evaluating the classifier
"""
def classify_documents(docs):
    predictions = []
    for document in docs:
        result = classify_nb(document)
        predictions.append(result)
    return predictions
def compute_accuracy(predictions,labels):
    correct = 0
    for x in range(len(predictions)):
        if(predictions[x] == labels[x]):
            correct += 1
    return correct/len(predictions)
"""
MAIN
"""

### START INIT ###
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

NEG_TOTAL_WORD, POS_TOTAL_WORD, TOTAL_WORD_SUM, NEG_WORD_COUNT, POS_WORD_COUNT = train_nb(train_docs, train_labels)
### END INIT ###


print(classify_nb(eval_docs[2]))
print("Evaluate set accuracy : " + str(compute_accuracy(classify_documents(eval_docs),eval_labels)))
print("Training set accuracy : " + str(compute_accuracy(classify_documents(train_docs),train_labels)))