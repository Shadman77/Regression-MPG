from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


def downloadDataSet():
    dataset_path = keras.utils.get_file(
        "auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    return dataset_path


def importDataSet():
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv("auto-mpg.data", names=column_names,
                              na_values="?", comment='\t', sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    print(dataset.tail())

    return dataset


def cleanDataSet(dataset):
    print(dataset.isna().sum())

    # Dropping rows with empty values
    dataset = dataset.dropna()

    # The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
    #Covert the numerical values to name of the origin in the 1st step
    dataset['Origin'] = dataset['Origin'].map(
        lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    print(dataset.tail())

    # Basically a column for each origin with the value of either 0 or 1
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    print(dataset.tail())
    return dataset

def prepareDataSet(dataset):
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    return train_dataset, test_dataset

def prepareLabels(train_dataset, test_dataset):
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    return train_labels, test_labels

def inspectDataSet(dataset):
    #Graphs
    '''
    sns.pairplot(dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    plt.show()
    '''

    #Stats
    stats = dataset.describe()
    stats.pop("MPG")
    stats = stats.transpose()
    print(stats)
    return stats

def normalize(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']


def build_model(train_dataset):
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model


def training(model, normed_train_data, train_labels):
    EPOCHS = 1000
    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])
    return history, model

def trainingEarlyStops(model, normed_train_data, train_labels):
    EPOCHS = 1000
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    early_history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    return early_history, model

def trainingStat(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [MPG]')
    plt.show()

    plotter.plot({'Basic': history}, metric = "mse")
    plt.ylim([0, 20])
    plt.ylabel('MSE [MPG^2]')
    plt.show()

def buildAndTrain(train_dataset, train_labels, normed_train_data):
    #Building the model
    model=build_model(train_dataset)
    model.summary()

    #Test the model
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    #Train the model 1000 epochs
    history, model = training(model, normed_train_data, train_labels)
    model.save("A.model")

    #Train the model early stop
    #history, model = trainingEarlyStops(model, normed_train_data, train_labels)
    #model.save("B.model")

    #Training stat
    trainingStat(history)

def finalEvaluation(normed_test_data, test_labels):
    #Load the model
    model = tf.keras.models.load_model("A.model")

    #Evaluate the final model
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    #Make predictions
    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()


def main():
    print('Tensorflow version =', tf.__version__)
    #dataset_path = downloadDataSet()
    # print(dataset_path)
    dataset = importDataSet()
    dataset = cleanDataSet(dataset)
    train_dataset, test_dataset = prepareDataSet(dataset)
    train_stats = inspectDataSet(train_dataset)
    train_labels, test_labels = prepareLabels(train_dataset, test_dataset)
    print(train_labels)

    #Normalize the datasets
    normed_train_data = normalize(train_dataset, train_stats)
    normed_test_data = normalize(test_dataset, train_stats)
    print(normed_train_data.tail())

    #Build and train model
    buildAndTrain(train_dataset, train_labels, normed_train_data)

    #Final Evaluation
    finalEvaluation(normed_test_data, test_labels)


if __name__ == "__main__":
    main()
