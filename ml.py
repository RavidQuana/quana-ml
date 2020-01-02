# %load main.py
import zipfile

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import itertools
import random
import glob
import os
import matplotlib.pyplot as plt
from csv import reader
from io import StringIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from concurrent.futures import ThreadPoolExecutor
import pickle

def time_to_state(time):
    if time < 15:
        return 0
    elif time < 40:
        return 1
    else:
        return 2

tags_list = {'Mold', 'Pesticide'}
tags_order = ['Mold', 'Pesticide']

def allowed_tags():
    return tags_list

def sensor_columns(df):
    return df.columns[df.columns.str.contains('qcm') | df.columns.str.contains("temp") | df.columns.str.contains("humidity")]

def string_to_arr(text):
    r = reader(StringIO(text[1:-1]), delimiter=',', quotechar='"')
    arr = next(r, [])
    return [x for x in arr if x in tags_list]

def preprocess(df, laggings, smoothing):
    # store for later use
    card = df['card'].iloc[0]
    sample = df['sample'].iloc[0]

    # drop sample id column
    df.drop("sample", axis=1, inplace=True)
    df.drop("card", axis=1, inplace=True)
    #df.drop("temp", axis=1, inplace=True)
    #df.drop("humidiy", axis=1, inplace=True)

    df.rename(columns={'humidiy': 'humidity'}, inplace=True)

    # get mean value of the first 8 seconds (didnt work because it used float and we need integers), so we just take the values at time=8
    relative_points = df.loc[df.time < 8].iloc[-1]

    columns = sensor_columns(df)
    # we only handle 1 tags so 0 = no tags and 1 = mold
    df.tags = df.tags.apply(string_to_arr)
    
    mlb = MultiLabelBinarizer()
    tags = pd.DataFrame(mlb.fit_transform(df['tags']),columns=mlb.classes_, index=df.index)
    
    # store for later use
    tags_index = df.columns.get_loc("tags")
    df = df.drop('tags', axis=1).join(tags)

    for tag in tags_list:
        if tag not in df:
            df[tag] = 0

    #reorder columns. important.
    df = df.reindex(columns=list(df.columns[:tags_index]) + tags_order, copy=False)

    # transform to relative data
    for column in columns:
        df[column] -= relative_points[column]


    if smoothing > 0:
        # grouping
        for column in columns:
            df[column] = df[column].rolling(window=7).mean()
            df[column] = df[column].groupby(df[column].index // smoothing).mean()
        df.time = df.time.groupby(df.time.index // smoothing).mean()
        df = df[:len(df.index) // smoothing]

    # remove 8 seconds from data
    df = df.drop(df[df.time < 8].index).reset_index(drop=True)

    #debug
    #print(df)

    #add lagging data
    for column in columns:
        index = df.columns.get_loc(column)
        # lagging by n prev values
        lags = laggings

        for l in lags:
            if l > 0:
                df.insert(index, column + "_prev_" + str(l),
                      df[column].shift(l, fill_value=0))
        # lagging by n next values
        index = df.columns.get_loc(column) + 1
        for l in lags:
            if l > 0:
                df.insert(index, column + "_next_" + str(l),
                      df[column].shift(-l, fill_value=0))

    # add step to help the learner
    #df.insert(3, "step", df.time.apply(time_to_state))
    # map time from secs to 0...1 where 0 is start and 1 is end
    #df.time = minmax_scale(df[['time']], copy=False)

    tags_index = df.columns.get_loc("Mold") 
    df._sample = sample
    df._card = card
    df._tags_index = tags_index

    return df


def accuracy(result, tag):
    return (np.count_nonzero(result[:, tag]) / result.size) * 100


def graph_for_mold(df, results):
    ax = plt.gca()
    for column in sensor_columns(df):
        df.plot(kind='line', x='time', y=column, ax=ax, figsize=(10, 5))
    # df.plot(kind='scatter',x='time',y="result",ax=ax,color='red')
    for index, r in enumerate(results):
        if r[1] == 1:
            ax.axvline(x=df.time.iloc[index],
                   color='blue', linestyle='--', alpha=0.1)
        if r[0] == 1:
            ax.axvline(x=df.time.iloc[index],
                       color='black', linestyle='--', alpha=0.1)
    plt.show(block=False)


def check_for_mold(path):
    df = preprocess(pd.read_csv(path))
    result = agents[df._card].predict(df.iloc[:, 0:df._tags_index].values)
    print("testing " + path, accuracy(result, 0))


def load_df(df, laggings, smoothing):
    return preprocess(df, laggings, smoothing)

def export(dfs, lagging, smoothing, algo, algo_rand):
    # load all filees
    print("Preprocessing...")

    with ThreadPoolExecutor(max_workers=12) as executor:
        dfs = executor.map(lambda x: load_df(x.copy(deep=True), lagging, smoothing), dfs)

    df_cards = {}
    agents = {}
    # split by card
    for member in dfs:
        card = member._card
        if card in df_cards:
            df_cards[card].append(member)
        else:
            df_cards[card] = [member]
        
    for key, value in df_cards.items():
        train = pd.concat(value, sort=False)

        # we get all input data
        x_train = train.iloc[:, 0:value[0]._tags_index].values
        # we get tags data
        y_train = train.iloc[:, value[0]._tags_index:].values

        #regressor = RandomForestClassifier(n_estimators=100, random_state=0, criterion="entropy")
        # using cart
        regressor = DecisionTreeClassifier(random_state=algo_rand)
        agents[key] = regressor

        # train
        regressor.fit(x_train, y_train)

    return agents


def run(dfs, lagging, smoothing, algo, target, file_rands, algo_rands):
    # load all filees

    print("Preprocessing...")

    #pool = Pool(8)
    with ThreadPoolExecutor(max_workers=12) as executor:
        dfs = executor.map(lambda x: load_df(x.copy(deep=True), lagging, smoothing), dfs)
    #pool.close()
    #pool.join()

    print("Starting routines")

    df_cards = {}
    agents = {}
    # split by card
    for member in dfs:
        card = member._card
        if card in df_cards:
            df_cards[card].append(member)
        else:
            df_cards[card] = [member]

    final_results = {}

    for file_rand in file_rands:
        rand = random.Random(file_rand)
        results = {}
        for key, value in df_cards.items():
            # split test and train, we take only small amount for testing because we have dont have enough data
            mold = list(filter(lambda x: x[target].iloc[0] == 1, value))
            not_mold = list(filter(lambda x: x[target].iloc[0] != 1, value))

            print("Card id:", key)
            print(target + " count:", len(mold))
            print("Non-" + target + " count:", len(not_mold))

            test_count_mold = 3
            test_count_non_mold = 3

            if (len(mold) < test_count_mold or len(not_mold) < test_count_non_mold):
                print("card does not have enough samples, skipping")
                continue

            rand.shuffle(mold)
            rand.shuffle(not_mold)

            train = itertools.chain(mold[:-test_count_mold],
                                    not_mold[:-test_count_non_mold])
            test = itertools.chain(mold[-test_count_mold:],
                                   not_mold[-test_count_non_mold:])

            print("Training size", target, ":", len(mold)-test_count_mold,
                  "non-", target, ":", len(not_mold)-test_count_non_mold)
            print("Testing size", target ,":", test_count_mold,
                  "non-", target, ":", test_count_non_mold)

            # combine all files
            train = pd.concat(train, sort=False)
            test = pd.concat(test, sort=False)

            # we get all input data
            x_train = train.iloc[:, 0:value[0]._tags_index].values
            # we get tags data
            y_train = train.iloc[:, value[0]._tags_index:].values

            # testing not used here because we do manually testing
            # we get all input data
            x_test = test.iloc[:, 0:value[0]._tags_index].values
            # we get tags data
            y_test = test.iloc[:, value[0]._tags_index:].values

            print("Training...")
            for algo_rand in algo_rands:
                #regressor = RandomForestClassifier(n_estimators=100, random_state=0, criterion="entropy")
                # using cart
                regressor = DecisionTreeClassifier(random_state=algo_rand)
                agents[key] = regressor

                # train
                regressor.fit(x_train, y_train)

                y_pred = regressor.predict(x_test)
                #print("confusion_matrix: ", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
                #print(classification_report(y_test, y_pred))
                for index, tag in enumerate(tags_order):
                    if tag != target:
                        y_test[:, index] = 0
                        y_pred[:, index] = 0

                print("accuracy_score: ", accuracy_score(y_test, y_pred) * 100)
                results[key] = results.get(key, 0) + accuracy_score(y_test, y_pred) * 100
                continue

                print("\nTesting on testing data...\n")

                for sample in mold[-test_count_mold:]:
                    result = regressor.predict(sample.iloc[:, 0:sample._tags_index].values)
                    #result_prob = regressor.predict_proba(sample.iloc[:, 0:sample._tags_index].values)
                    #print(result)
                    print(target + " test " + sample._path + ":", accuracy_score(sample.iloc[:, sample._tags_index:], result) * 100)
                    #print("prob:", result_prob[0])
                    #graph_for_mold(sample, result)
                    print(
                        "https://quana-server-production.herokuapp.com/admin/sample_betas/" + str(sample._sample))
                    print()

                for sample in not_mold[-test_count_non_mold:]:
                    result = regressor.predict(sample.iloc[:, 0:sample._tags_index].values)
                    #result_prob = regressor.predict_proba(sample.iloc[:, 0:sample._tags_index].values)
                    print("non " + target + "test " + sample._path + ":",  accuracy_score(sample.iloc[:, sample._tags_index:], result) * 100)
                    #print("prob:", result_prob[0])
                    #graph_for_mold(sample, result)
                    print(
                        "https://quana-server-production.herokuapp.com/admin/sample_betas/" + str(sample._sample))
                    print()

                print("\n########\n")

        for key in results:
            final_results[key] = final_results.get(key, 0) + (results[key] / len(algo_rands))

    for key in final_results:
        final_results[key] /= len(file_rands)

    return final_results


def read_csv(file):
    df = pd.read_csv(file)
    df._path = file
    return df

def mainLag():
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()

    dfs = list(map(read_csv, files))
    
    cards = {}

    lag_max = 10
    x = list(range(0, lag_max))
    for lag in x:
        print("Lag:", lag)
        results = run(dfs, [lag], 0, "entropy", "Pesticide", range(0, 10), range(0, 10))
        for key, value in results.items():
            cards[key] = cards.get(key, np.zeros(lag_max))
            cards[key][lag] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot(x, value, label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Lag')
    plt.legend(title='Cards:')
    plt.title('Lagging Check Pesticide')
    plt.show(block=True)


def mainSmooth():
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()

    dfs = list(map(read_csv, files))

    cards = {}

    arg_max = 10
    x = list(range(0, arg_max))
    for smooth in x:
        print("Smooth:", smooth)
        results = run(dfs, [], smooth, "entropy", "Mold", range(0, 10), range(0, 10))
        for key, value in results.items():
            cards[key] = cards.get(key, np.zeros(arg_max))
            cards[key][smooth] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot(x, value, label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Smoothing')
    plt.legend(title='Cards:')
    plt.title('Smoothing Check Mold')
    plt.show(block=True)

def mainCombined():
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()

    dfs = list(map(read_csv, files))

    cards = {}

    results = run(dfs, [1,2], 2, "entropy", "Pesticide", range(0, 10), range(0, 10))
    for key, value in results.items():
        cards[key] = cards.get(key, 0)
        cards[key] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot([1,2,3], [value,value,value], label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Dummy')
    plt.legend(title='Cards:')
    plt.title('Pesticide Smooth 2 Lag [1,2]')
    plt.show(block=True)


def open_zip_agent(file):
    agent = {}
    try:
        with zipfile.ZipFile(file, "r") as f:
            for name in f.namelist():
                agent[name] = pickle.load(f.open(name))
    except Exception as e:
        print("Open zip error", e)
        return None

    return agent   

def open_zip(file_path):
    dfs = []
    try:
        with zipfile.ZipFile(file_path, "r") as f:
            for name in f.namelist():
                df = pd.read_csv(f.open(name))
                df._path = name
                dfs.append(df)
    except Exception as e:
        print("Open zip error", e)
        return None

    return dfs   

def classify(agent, file):
    df = pd.read_csv(file)
    df._path = "Sample"
    df = load_df(df, [], 0)

    if str(df._card) not in agent:
        return None

    regressor = agent[str(df._card)]
    result = {}

    test_data = df.iloc[:, 0:df._tags_index].values
    pred = regressor.predict(test_data)

    tags_counter = {}
    for index, tag in enumerate(tags_order):
        tags_counter[index] = 0

    for x in pred:
        for index, value in enumerate(x):
            tags_counter[index] += value

    result = {}
    for index, counter in tags_counter.items():
        result[tags_order[index]] = (counter / len(pred)) * 100.0
    print(result)
    return result

# train(open('./samples.zip', 'r'))
# train(urllib.request.urlopen('http://samples.zip'))

# import requests, zipfile, io
# r = requests.get(zip_file_url)
# z = zipfile.ZipFile(io.BytesIO(r.content))


