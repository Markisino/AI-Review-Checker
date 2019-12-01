from __future__ import division
from collections import Counter
from codecs import open
import numpy as np
import operator

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


def score_doc_label(document, smoothing=0.5):
    negProb = np.log10(NEG_TOTAL_WORD / TOTAL_WORD_SUM)
    posProb = np.log10(POS_TOTAL_WORD / TOTAL_WORD_SUM)

    smoothNegTotal = NEG_TOTAL_WORD + smoothing * (len(NEG_WORD_COUNT))
    smoothPosTotal = POS_TOTAL_WORD + smoothing * (len(POS_WORD_COUNT))

    for word in document:
        negProb += np.log10((NEG_WORD_COUNT[word] + smoothing) / smoothNegTotal)
        posProb += np.log10((POS_WORD_COUNT[word] + smoothing) / smoothPosTotal)

    return negProb, posProb


def classify_nb(document, smoothing=0.5):
    negProb, posProb = score_doc_label(document, smoothing)

    out = str()

    if negProb > posProb:
        out = "neg"

    else:
        out = "pos"

    return out


"""
Task 3: Evaluating the classifier
"""
def classify_documents(docs, smoothing=0.5):
    predictions = []
    for document in docs:
        result = classify_nb(document, smoothing)
        predictions.append(result)
    return predictions

def compute_accuracy(predictions, labels):
    correct = 0
    correct_pos = 0
    total_pos = 0
    correct_neg = 0
    total_neg = 0
    error_indexes = []
    for x in range(len(predictions)):
        if predictions[x] == 'pos':
            total_pos += 1
            if(predictions[x] == labels[x]):
                correct += 1
                correct_pos += 1
            else:
                error_indexes.append(x)
        elif predictions[x] == 'neg':
            total_neg += 1
            if(predictions[x] == labels[x]):
                correct += 1
                correct_neg += 1
            else:
                error_indexes.append(x)



    overall_accuracy = correct/len(predictions)
    pos_accuracy = correct_pos/total_pos
    neg_accuracy = correct_neg/total_neg
    print("correct pos: "+ str(correct_pos) + "/"+str(total_pos))
    print("correct neg: " +str(correct_neg) + "/"+str(total_neg))
    return overall_accuracy, pos_accuracy, neg_accuracy, error_indexes

def misclassified_document(evaluation_docs, evaluation_labels, error_indexes):
    for x in error_indexes:
        print('{}: {}'.format(evaluation_labels[x], evaluation_docs[x]), file=open('data2.txt', 'a'))
"""
MAIN
"""

def word_value(document, labels):
    #1 in data.txt is 9 in eval_docs array.
    #11 in data.txt is 50 in eval_docs array.
    #111 in data.txt is 547 in eval_docs array.
    index = [9, 50, 547]
    for word in document[index[2]]:
        print('Word "{}" in negative: {}, in positive: {}'.format(word, NEG_WORD_COUNT[word], POS_WORD_COUNT[word]), file=open('analysis2.txt', 'a'))
        
### START INIT ###
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]

to_classify = input("Enter filename to classify (leave blank if using default test data) :")
if(to_classify == ''):
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]
else:
    docs,labels = read_documents(to_classify)
    eval_docs = docs
    eval_labels = labels   
NEG_TOTAL_WORD, POS_TOTAL_WORD, TOTAL_WORD_SUM, NEG_WORD_COUNT, POS_WORD_COUNT = train_nb(train_docs, train_labels)
### END INIT ###

#print("Evaluate set accuracy (no smoothing) : " + str(compute_accuracy(classify_documents(eval_docs,smoothing=0),eval_labels)))
#print("Training set accuracy (no smoothing) : " + str(compute_accuracy(classify_documents(train_docs,smoothing=0),train_labels)))
#print("Evaluate set accuracy (0.5) : " + str(compute_accuracy(classify_documents(eval_docs, smoothing = 0.5),eval_labels)))
smoothing = 0.93
overall, pos, neg, err_index = compute_accuracy(classify_documents(eval_docs, smoothing=smoothing), eval_labels)
print("Prediction accuracy ("+str(smoothing)+" smoothing) : \n\t" + "Overall accuracy : "+str(overall) + "\n\tPos accuracy : " + str(pos) + "\n\tNeg accuracy : " + str(neg))

#misclassified_document(eval_docs, eval_labels, err_index)

#smoothing_old = 0
#eval_acc_old = compute_accuracy(classify_documents(eval_docs, smoothing=smoothing_old), eval_labels)[0]
#eval_acc_new = 1
#
#smoothing_values = {}

# word_value(eval_docs, eval_labels)
# while(smoothing_old < 1):
#     smoothing_new = smoothing_old + 0.01
#     eval_acc_new, new_pos, new_neg, err_index = compute_accuracy(classify_documents(eval_docs, smoothing = smoothing_new),eval_labels)
#     eval_acc_old = eval_acc_new
#     smoothing_old = smoothing_new
#     smoothing_values[(smoothing_new)] = eval_acc_new
#     print("Smoothing " + str(smoothing_new) + ": ")
#     print("\toverall \t" + str(eval_acc_new))
#     print("\tpos \t\t" + str(new_pos))
#     print("\tneg \t\t" + str(new_neg))
#     print('Smoothing ' + str(smoothing_new) + ": " + str(eval_acc_new) + ': ' + str(new_pos) + ': '+ str(new_neg), file=open('data.txt','a'))