import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from newdata_RF import run_one


def _unpickle(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def _pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

class RFCModel:

    def __init__(self):
        self.tfidf_matrix = _unpickle('data/tfidf_all.pkl')

    def load_data(self):
        self.y = self._get_y()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tfidf_matrix, self.y, test_size=0.20, stratify=self.y, random_state=42)

    def _get_y(self, filename='data/data.json'):
        df = pd.read_json(filename)
        df['fraud'] = (df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event')
        fraud = df['fraud'].values
        return fraud

    def fit(self):
        self.rf = RandomForestClassifier(bootstrap=True, \
                                        criterion = "gini", \
                                        class_weight = 'balanced_subsample', \
                                        n_estimators = 50, \
                                        max_depth = 100, \
                                        max_features= 30, \
                                        min_samples_leaf= 2, \
                                        min_samples_split = 2)
        self.rf.fit(self.X_train, self.y_train)

    def pickle(self):
        _pickle(self.rf, 'data/RFCModel.pkl')

    def unpickle(self):
        self.rf = _unpickle('data/RFCModel.pkl')

    def score(self):
        y_predict = self.rf.predict(self.X_test)
        print("score (accuracy):", self.rf.score(self.X_test, self.y_test))
        print("precision:", precision_score(self.y_test, y_predict))
        print("recall:", recall_score(self.y_test, y_predict))
        print(np.array([['TN','FP'],['FN', 'TP']]))
        print(confusion_matrix(self.y_test, y_predict))
        f = f1_score(self.y_test, y_predict)
        print('F1: ', f)

    def predict_proba_all(self):
        prob = self.rf.predict_proba(self.tfidf_matrix)
        return prob[:,1]

    def predict(self, X):
        return self.rf.predict(X)

    # def predict_proba(self, X):
    #     prob = self.rf.predict_proba(X)
    #     return prob[:,1]

    # Need to change this to read from a string, etc.
    def predict_proba(self, filename):
        prob = run_one(json_filename=filename, rf_filename='data/RFCModel.pkl')
        # prob = self.rf.predict_proba(X)
        return prob


if __name__ == '__main__':
    # model = RFCModel()
    # model.load_data()
    # model.fit()
    # model.pickle()
    # model.score()

    model = RFCModel()
    model.unpickle()
    prob = model.predict_proba('data/data_point.json')
    print(prob)
