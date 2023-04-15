from hazm import *
import re
from string import punctuation
import pandas as pd
import numpy as np
import copy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


def delete_stop_words(string):
    cleaned_string = re.sub(f'[{punctuation}؟،٪×÷»«]+', '', string)
    stop_words = utils.stopwords_list()
    tokenizer = WordTokenizer()
    lst = tokenizer.tokenize(cleaned_string)
    res = []
    for token in lst:
        if token not in stop_words:
            res.append(token)
    result = " ".join(res)
    return result


def prepare_lines(u, smat, vh):
    lines = []
    lines.append('u:')
    lines.append(newline)
    for i in range(u.shape[0]):
        tmp = u[i, :]
        res = " ".join([str(i) for i in u])
        lines.append(res)
        lines.append(newline)
    lines.append(newline)
    lines.append('s:')
    lines.append(newline)
    for i in range(smat.shape[0]):
        tmp = smat[i, :]
        res = " ".join([str(i) for i in smat])
        lines.append(res)
        lines.append(newline)
    lines.append(newline)
    lines.append('vh:')
    lines.append(newline)
    for i in range(vh.shape[0]):
        tmp = vh[i, :]
        res = " ".join([str(i) for i in vh])
        lines.append(res)
        lines.append(newline)
    return lines


if __name__ == "__main__":
    newline = '\n'

    persica = PersicaReader('persica.csv')
    persica = PersicaReader('persica.csv')
    i = 0
    docs = persica.docs()
    normalizer = Normalizer()
    cats = set()
    lst = []
    y = []
    i = 0

    try:
        while True:
            obj = next(docs)
            i += 1
            line = obj['text']
            y.append(obj['category2'])
            cats.add(obj['category2'])
            # line = normalizer.affix_spacing(line)
            # line = normalizer.character_refinement(line)
            # line = normalizer.punctuation_spacing(line)
            # line = delete_stop_words(line)
            lst.append(copy.deepcopy(line))
        # raise Exception('ii')
    except:
        print("number of docs =", i)
        print("end of docs")

        vectorizer = CountVectorizer()
        X1 = vectorizer.fit_transform(lst)

        vectorizer2 = TfidfVectorizer()
        X2 = vectorizer.fit_transform(lst)

        X1 = X1.toarray().T
        X2 = X2.toarray().T

        print("initial =", X1.shape)
        print("tfidf =", X2.shape)
        u, s, vh = randomized_svd(X1, n_components=250, random_state=None)
        smat = np.diag(s)

        # SVD = TruncatedSVD(n_components=50)
        # X_TSVD = SVD.fit_transform(X)
        # print(X_TSVD.shape)
        #
        # u, s, vh = np.linalg.svd(X_TSVD)
        print('u =', u.shape)
        print('smat =', smat.shape)
        print('vh =', vh.shape)
        corpus = np.dot(smat, vh)
        print('c =', corpus.shape)

        y = np.array(y)
        le = preprocessing.LabelEncoder()
        le.fit(list(cats))
        y_labeled = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(corpus.T, y_labeled, test_size=0.2)

        # SVM
        SVM_clf = SVC(kernel='rbf')
        SVM_clf.fit(X_train, y_train)

        y_hat = SVM_clf.predict(X_test)
        precision = np.round(precision_score(y_test, y_hat, average='macro') * 100, 4)
        recall = np.round(recall_score(y_test, y_hat, average='macro') * 100, 4)
        print("recall =", recall)
        print("precision =", precision)
        svm_score = SVM_clf.score(X_test, y_test)
        print("SVM for =", np.round(svm_score * 100, 4))

        # MLP
        hidden_layers = (20, 50, 30, 20, 10, 5)
        MLP_clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='tanh', max_iter=500)
        MLP_clf.fit(X_train, y_train)

        y_hat = MLP_clf.predict(X_test)
        precision = np.round(precision_score(y_test, y_hat, average='macro') * 100, 4)
        recall = np.round(recall_score(y_test, y_hat, average='macro') * 100, 4)
        print("recall =", recall)
        print("precision =", precision)
        mlp_score = MLP_clf.score(X_test, y_test)
        print("MLP =", np.round(mlp_score * 100, 4))

        # Random Forest
        RFclf = RandomForestClassifier(criterion='entropy')
        RFclf.fit(X_train, y_train)

        y_hat = RFclf.predict(X_test)
        precision = np.round(precision_score(y_test, y_hat, average='macro') * 100, 4)
        recall = np.round(recall_score(y_test, y_hat, average='macro') * 100, 4)
        print("recall =", recall)
        print("precision =", precision)
        rf_score = RFclf.score(X_test, y_test)
        print("Random Forest =", np.round(rf_score * 100, 4))
        # lines = prepare_lines(u, smat, vh)
        # with open('svd.txt', 'w') as f:
        #     f.writelines(lines)
    # print(len(lst))
# obj = next(docs)
# print(obj)

# print(obj.keys())
