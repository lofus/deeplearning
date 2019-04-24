from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from load_data_utils import *
import numpy as np
import pandas as pd
import h5py

def load_data_digits():
    digits = datasets.load_digits()
    Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)
    Ytrain = np.zeros((1, len(ytrain)))
    Ytrain[0] = ytrain
    Ytrain = Ytrain / 10
    Ytest = np.zeros((1, len(ytest)))
    Ytest[0] = ytest
    Ytest = Ytest / 10
    return Xtrain.T/50, Ytrain, Xtest.T/50, Ytest


def load_data_cat_pics():
    train_dataset = h5py.File('../data/h5/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('../data/h5/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Reshape the training and test examples
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    return train_x, train_set_y_orig, test_x, test_set_y_orig, classes

def load_data_jira():
    #dataset path
    train_data_files =  ['../data/jira_2/TrainingDataSet_SummaryDescription.csv']
    train_label_files = ['../data/jira_2/TrainingDataSet_Component.csv']
    test_data_files =   ['../data/jira_2/TestDataSet_SummaryDescription.csv']
    test_label_files =  ['../data/jira_2/TestDataSet_Component.csv']

    train_data = []
    for data_file in train_data_files:
        with open(data_file, 'r', encoding='latin-1') as f:
            train_data.extend([s.strip() for s in f.readlines()])
            train_data = [clean_str(s) for s in train_data]

    test_data = []
    for test_data_file in test_data_files:
        with open(test_data_file, 'r', encoding='latin-1') as f:
            test_data.extend([s.strip() for s in f.readlines()])
            test_data = [clean_str(s) for s in test_data]

    train_labels = pd.read_csv(train_label_files[0])
    train_label_files.pop(0)
    for labels_file in train_label_files:
        labels_df = pd.read_csv(labels_file)
        train_labels.append(labels_df)

    test_labels = pd.read_csv(test_label_files[0])
    test_label_files.pop(0)
    for test_label_file in test_label_files:
        test_label_df = pd.read_csv(test_label_file)
        test_labels.append(test_label_df)

    data = pd.read_csv("../data/jira_2/List_ID_Sort.csv", encoding='latin-1')
    x = data.Summary
    y = data.Component
    ##TODO: try cross-validation
    dev_sample_index = -1 * int(0.2 * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

    document_length_df = pd.DataFrame([len(xx.split(" ")) for xx in x_train])
    document_length = np.int64(document_length_df.quantile(0.8))
    vocabulary_processor = learn.preprocessing.VocabularyProcessor(document_length)
    t_train = vocabulary_processor.fit_transform(x_train)

    x_train = np.array(list(vocabulary_processor.fit_transform(x_train)), dtype=np.float32)
    x_dev = np.array(list(vocabulary_processor.transform(x_dev)))
    label_processor = learn.preprocessing.VocabularyProcessor(1)
    y_train = np.array(list(label_processor.fit_transform(y_train)), dtype=np.float32)
    y_dev = np.array(list(label_processor.transform(y_dev)))

    return x_train.T, y_train.T, x_dev.T, y_dev.T

