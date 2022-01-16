import numpy as np
import pandas as pd
from sklearn import preprocessing
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


class UCIDataset:

    # The init method or constructor
    def __init__(self, file_path, sep, label, divide_dataset=True, header=None):
        # Read csv data file
        if (sep == ','):
            df = pd.read_csv(file_path, header=header, sep=',')
        if (sep == ';'):
            df = pd.read_csv(file_path, header=header, sep=';')
        if (sep == ' '):
            df = pd.read_csv(file_path, delim_whitespace=True, header=header)
        if (sep == 'df'):
            df = file_path
            file_path = 'dummy'

        # Set raw data attribute
        self.df = df
        self.df_sampled = df
        self.label = label

        self.name = file_path.split('/')[-1].split('.')[0]

        # Encode categorical features
        le = preprocessing.LabelEncoder()
        for col in self.df.columns:
            if (self.df[col].dtypes == 'object'):
                list_of_values = list(self.df[col].unique())
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
                le.fit(list_of_values)
                self.df[col] = le.transform(self.df[col])

        # Replcae missing values
        self.df = self.df.replace('?', np.NaN)
        self.df = self.df.fillna(self.df.median())
        # df = df.astype('int32')

        if (divide_dataset):
            self.divideDataset()

    def divideDataset(self, classifier, normalize=True, shuffle=True, all_features=True, all_instances=True,
                      evaluate=True, partial_sample=False, folds=10):

        # Set classifier
        self.clf = copy.copy(classifier)
        self.folds= folds

        # Shuffle dataset
        if (shuffle):
            self.df = self.df.sample(frac=1)
            self.df_sampled = self.df

        # Sample from dataset
        if (partial_sample):
            self.df_sampled = self.df.sample(n=partial_sample)

        # Divide datset into training/validation/testing
        if (self.label == -1):
            self.X = self.df_sampled.iloc[:, :-1].values
            self.y = self.df_sampled.iloc[:, -1].values
        else:
            selector = [x for x in range(self.df.shape[1]) if x != label]
            self.X = self.df_sampled.iloc[:, selector].values
            self.y = self.df_sampled.iloc[:, label].values

        X_train = self.X[0:int(0.6 * len(self.X)), :]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 0.0001
        X_train_normalized = (X_train - mean) / std
        y_train = self.y[0:int(0.6 * len(self.X))]

        X_val = self.X[int(0.6 * len(self.X)):int(0.8 * len(self.X)), :]
        y_val = self.y[int(0.6 * len(self.X)):int(0.8 * len(self.X))]
        X_val_normalized = (X_val - mean) / std

        X_test = self.X[int(0.8 * len(self.X)):, :]
        y_test = self.y[int(0.8 * len(self.X)):]
        X_test_normalized = (X_test - mean) / std

        # Set attribute values
        if (normalize):
            self.X_train = X_train_normalized
            self.y_train = y_train
            self.X_val = X_val_normalized
            self.y_val = y_val
            self.X_test = X_test_normalized
            self.y_test = y_test
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test

        # Confirm instances/features to be used for learning
        if (all_features):
            self.features = np.ones(X_train.shape[1])
        else:
            self.features = np.zeros(X_train.shape[1])
            while (np.sum(self.features) == 0):
                zero_p = random.uniform(0, 1)
                # zero_p = 0.5
                self.features = np.random.choice([0, 1], size=(X_train.shape[1],), p=[zero_p, (1 - zero_p)])
        self.features = list(np.where(self.features == 1)[0])

        if (all_instances):
            self.instances = np.ones(X_train.shape[0])
        else:
            self.instances = np.random.choice([0, 1], size=(X_train.shape[0],), p=[0.5, 0.5])
        self.instances = list(np.where(self.instances == 1)[0])
        # Train model and evaluate on validation/testing sets
        if (evaluate):
            self.fitClassifier()
            self.setValidationAccuracy()
            self.setTestAccuracy()

    def fitClassifier(self):
        self.clf = self.clf.fit(self.X_train[self.instances, :][:, self.features], self.y_train[self.instances])

    def setValidationAccuracy(self):
        y_pred = self.clf.predict(self.X_val[:, self.features])
        self.ValidationAccuracy = accuracy_score(self.y_val, y_pred)

    def setTestAccuracy(self):
        y_pred = self.clf.predict(self.X_test[:, self.features])
        self.TestAccuracy = accuracy_score(self.y_test, y_pred)

    def setCV(self):
        scores = cross_val_score(self.clf, self.X_train[:][:, self.features], self.y_train[:], cv=self.folds,
                                 scoring='accuracy')
        #print(scores)
        #print(np.mean(scores))
        #print()
        self.CV = np.mean(scores)

    def setTrainSet(self, selected_instances):
        self.X_train = self.X_train[selected_instances]
        self.y_train = self.y_train[selected_instances]

    def setFeatures(self, selected_features):
        self.features = selected_features

    def setInstances(self, selected_instances):
        self.instances = selected_instances

    def getValidationAccuracy(self):
        return self.ValidationAccuracy

    def getTestAccuracy(self):
        return self.TestAccuracy

    def getCV(self):
        return self.CV