# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:15:27 2018

@author: Nikola Zivkovic
This repository contains examples used in blogpost - https://rubikscode.net/2018/02/05/introduction-to-tensorflow-with-python-example/
"""

"""
@reference site
https://www.tensorflow.org/tutorials/estimator/premade#create_input_functions

"""

# Import `tensorflow` and `pandas`
#!/usr/bin/env python
import psutil
import tensorflow as tf
import pandas as pd
import csv
import os
import datetime
 
countstages = 0
COLUMN_NAMES = [
        'SepalLength', 
        'SepalWidth',
        'PetalLength', 
        'PetalWidth', 
        'Species'
        ]

def resetfiles() :
    if os.path.exists('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/people1.csv'):
        os.remove("d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/people1.csv")
        print("File Removed!")

def resetcpuusagefile() :
    if os.path.exists('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/cpuusage.csv'):
        os.remove("d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/cpuusage.csv")
        print("File Removed!")

rowheadcpu = ['stages','total','available','percent','used','free', 'timetaken', 'stage_description']
resetcpuusagefile()
def writeCPUUsage(stagedesc) :
    global countstages
    # gives a single float value
    psutil.cpu_percent()
    # gives an object with many fields
    psutil.virtual_memory()
    # you can convert that object to a dictionary 
    result_dict = dict(psutil.virtual_memory()._asdict())

    print(result_dict)
    data = []
    with open('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/cpuusage.csv', 'a') as csvFile:
        with open("d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/cpuusage.csv", "r") as f:
            data = list(csv.reader(f))
        
        writer = csv.writer(csvFile)
        if data.__len__() == 0 :
            writer.writerow(rowheadcpu)
        
        if stagedesc != "" :
            countstages=countstages+1
            rowheadcpudata = [countstages]

            for key, value in result_dict.items() :
                print(key)
                rowheadcpudata.append(value)
            rowheadcpudata.append(datetime.datetime.now())
            rowheadcpudata.append(stagedesc)
            writer.writerow(rowheadcpudata)
        elif data.__len__() == 0 :
            writer.writerow(rowheadcpu)

    csvFile.close()
writeCPUUsage("******Program starts*******")
writeCPUUsage("before Import training dataset ")
# Import training dataset
training_dataset = pd.read_csv('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/iris_training.csv', names=COLUMN_NAMES, header=0)
writeCPUUsage("before Import training dataset :: after pandas read the csv")
train_x = training_dataset.iloc[:, 0:4]
writeCPUUsage("before Import training dataset :: after tensorflow prepare data x")
train_y = training_dataset.iloc[:, 4]
writeCPUUsage("before Import training dataset :: after tensorflow prepare data y")

writeCPUUsage("before Import testing dataset :: before tensorflow prepare test data")
# Import testing dataset
test_dataset = pd.read_csv('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/iris_test.csv', names=COLUMN_NAMES, header=0)
writeCPUUsage("before Import testing dataset :: after pandas read the csv")
test_x = test_dataset.iloc[:, 0:4]
writeCPUUsage("before Import testing dataset :: after tensorflow prepare test data x")
test_y = test_dataset.iloc[:, 4]
writeCPUUsage("before Import testing dataset :: after tensorflow prepare test data y")

writeCPUUsage("Setup feature columns")
# Setup feature columns
columns_feat = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]
writeCPUUsage("after setup feture columns")

writeCPUUsage("before Build Neural Network - Classifier")
# Build Neural Network - Classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=columns_feat,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model is classifying 3 classes
    n_classes=3)
writeCPUUsage("after Build Neural Network - Classifier")
# Define train function
def train_function(inputs, outputs, batch_size):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), outputs))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return dataset

writeCPUUsage("before Train the Model")
# Train the Model.
classifier.train(
    input_fn=lambda:train_function(train_x, train_y, 100),
    steps=1000)
writeCPUUsage("after Train the Model")

# Define evaluation function
def evaluation_function(attributes, classes, batch_size):
    attributes=dict(attributes)
    if classes is None:
        inputs = attributes
    else:
        inputs = (attributes, classes)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

writeCPUUsage("before Evaluate the Model")
# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:evaluation_function(test_x, test_y, 100))
writeCPUUsage("after Evaluate the Model")
print('\nAccuracy: \n',eval_result)
print('\nAccuracy: {accuracy:0.3f}\n'.format(**eval_result))


# Generate predictions from the model
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
expected = ['Setosa', 'Versicolor', 'Virginica']

#Setosa data
predict_setosa = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

#Virginica data
predict_viricolor = {
    'SepalLength': [5.1, 5.5, 4.3],
    'SepalWidth': [3.3, 4.2, 3],
    'PetalLength': [1.7, 1.4, 1.1],
    'PetalWidth': [0.5, 0.2, 0.1],
}

#Virginica data
predict_virginica = {
    'SepalLength': [6.9, 6.3, 6.7],
    'SepalWidth': [3.1, 2.8, 2.5],
    'PetalLength': [5.4, 5.1, 5.8],
    'PetalWidth': [2.1, 1.5, 1.8],
}

def predict_fn(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

writeCPUUsage("before reset the csv storage ")
resetfiles()
#writeCPUUsage()
writeCPUUsage("after reset the csv storage ")

rowhead = ['prediction', 'probability', 'expected', 'for flower']
with open('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/people1.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        
        writer.writerow(rowhead)
        rowhead = []

csvFile.close()
writeCPUUsage("after reset the csv storage :: written the first row ")

writeCPUUsage("before predict data ")
predit_result = classifier.predict(
    input_fn=lambda:predict_fn(test_x, 256)
)
writeCPUUsage("after predict data ")

writeCPUUsage("before creating the prediction data log ")
print("++++++++++++++++++++for testx+++++++++++++++++++++++++++")
for pred_dict, expec in zip(predit_result, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))
    row = []
    row.append(SPECIES[class_id])
    row.append(100*probability)
    row.append(expec)
    row.append("Setosa")
    with open('d:/vscode-workspace/NMZivkovic_tensorflow/IntrouductionToTensorFlow/people1.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        row.clear()

    csvFile.close()

writeCPUUsage("after creating the prediction data log ")
writeCPUUsage("******Program ends*******")