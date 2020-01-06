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
import scipy.signal as signal


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
    return df.columns[
        df.columns.str.contains('qcm') | df.columns.str.contains("temp") | df.columns.str.contains("humidity")]

def gcm_columns(df):
    return df.columns[df.columns.str.contains('qcm')]

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
    # df.drop("temp", axis=1, inplace=True)
    # df.drop("humidiy", axis=1, inplace=True)

    df.rename(columns={'humidiy': 'humidity'}, inplace=True)

    # get mean value of the first 8 seconds (didnt work because it used float and we need integers), so we just take the values at time=8
    relative_points = df.loc[df.time < 8].iloc[-1]

    columns = sensor_columns(df)
    # we only handle 1 tags so 0 = no tags and 1 = mold
    df.tags = df.tags.apply(string_to_arr)

    mlb = MultiLabelBinarizer()
    tags = pd.DataFrame(mlb.fit_transform(df['tags']), columns=mlb.classes_, index=df.index)

    # store for later use
    tags_index = df.columns.get_loc("tags")
    df = df.drop('tags', axis=1).join(tags)

    for tag in tags_list:
        if tag not in df:
            df[tag] = 0

    # reorder columns. important.
    df = df.reindex(columns=list(df.columns[:tags_index]) + tags_order, copy=False)

    # transform to relative data
    for column in columns:
        df[column] -= relative_points[column]

    # for column in columns:
    #    df[column] = df[column].rolling(window=3).mean()

    if smoothing > 0:
        for column in columns:
            # First, design the Buterworth filter
            N = 3  # Filter order
            Wn = [0.03, 0.14]  # Cutoff frequency
            B, A = signal.butter(N, Wn, output='ba', btype="bandstop")
            # Second, apply the filter
            df[column] = signal.filtfilt(B, A, df[column])

        # S = 2
        # for column in columns:
        # #     #df[column] = df[column].rolling(window=3).mean()
        #      df[column] = df[column].groupby(df[column].index // S).mean()
        # df.time = df.time.groupby(df.time.index // S).mean()
        # df = df[:len(df.index) // S]

    # if smoothing > 0:
    # grouping

    #

    # remove 8 seconds from data
    df = df.drop(df[df.time < 8].index).reset_index(drop=True)

    # debug
    # print(df)

    # add lagging data
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
    # df.insert(3, "step", df.time.apply(time_to_state))
    # map time from secs to 0...1 where 0 is start and 1 is end
    # df.time = minmax_scale(df[['time']], copy=False)

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
    plt.show(block=True)


def check_for_mold(path):
    df = preprocess(pd.read_csv(path))
    result = agents[df._card].predict(df.iloc[:, 0:df._tags_index].values)
    print("testing " + path, accuracy(result, 0))


def load_df(df, laggings, smoothing):
    return preprocess(df, laggings, smoothing)


def export(dfs, lagging, smoothing, algo, algo_rand):
    # load all filees
    print("Preprocessing...")

    #with ThreadPoolExecutor(max_workers=12) as executor:
    dfs = map(lambda x: load_df(x.copy(deep=True), lagging, smoothing), dfs)

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

        # regressor = RandomForestClassifier(n_estimators=100, random_state=0, criterion="entropy")
        # using cart
        regressor = DecisionTreeRegressor(random_state=algo_rand)
        agents[key] = regressor

        # train
        regressor.fit(x_train, y_train)

    return agents


def run(dfs, lagging, smoothing, algo, target, file_rands, algo_rands):
    # load all filees

    print("Preprocessing...")

    # pool = Pool(8)
    with ThreadPoolExecutor(max_workers=12) as executor:
        dfs = executor.map(lambda x: load_df(x.copy(deep=True), lagging, smoothing), dfs)
    # pool.close()
    # pool.join()

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

            if (len(mold) < 3 or len(not_mold) < 3):
                print("card does not have enough samples, skipping")
                continue

            test_count_mold = max(int(0.1 * len(mold)), 3)
            test_count_non_mold = test_count_mold  # max(int(0.1 * len(not_mold)), 3)

            rand.shuffle(mold)
            rand.shuffle(not_mold)

            train = itertools.chain(mold[:-test_count_mold],
                                    not_mold[:-test_count_non_mold])
            test = itertools.chain(mold[-test_count_mold:],
                                   not_mold[-test_count_non_mold:])

            print("Training size", target, ":", len(mold) - test_count_mold,
                  "non-", target, ":", len(not_mold) - test_count_non_mold)
            print("Testing size", target, ":", test_count_mold,
                  "non-", target, ":", test_count_non_mold)

            # combine all files
            train = pd.concat(train, sort=False)
            test = pd.concat(test, sort=False)

            # print(train)

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
                #regressor = RandomForestClassifier(n_estimators=1 + est * 10, random_state=algo_rand,
                 #                                  criterion="entropy")
                # using cart
                regressor = DecisionTreeClassifier(random_state=algo_rand)
                agents[key] = regressor

                # train
                regressor.fit(x_train, y_train)

                y_pred = regressor.predict(x_test)
                # print("confusion_matrix: ", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
                # print(classification_report(y_test, y_pred))
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
                    # result_prob = regressor.predict_proba(sample.iloc[:, 0:sample._tags_index].values)
                    # print(result)
                    print(target + " test " + sample._path + ":",
                          accuracy_score(sample.iloc[:, sample._tags_index:], result) * 100)
                    # print("prob:", result_prob[0])
                    # graph_for_mold(sample, result)
                    print(
                        "https://quana-server-production.herokuapp.com/admin/sample_betas/" + str(sample._sample))
                    print()

                for sample in not_mold[-test_count_non_mold:]:
                    result = regressor.predict(sample.iloc[:, 0:sample._tags_index].values)
                    # result_prob = regressor.predict_proba(sample.iloc[:, 0:sample._tags_index].values)
                    print("non " + target + "test " + sample._path + ":",
                          accuracy_score(sample.iloc[:, sample._tags_index:], result) * 100)
                    # print("prob:", result_prob[0])
                    # graph_for_mold(sample, result)
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
    target = "Mold"

    lag_max = 2
    x = list(range(0, lag_max))
    for lag in x:
        print("Lag:", lag)
        results = run(dfs, [lag], 0, "entropy", target, range(0, 10), range(0, 5))
        for key, value in results.items():
            cards[key] = cards.get(key, np.zeros(lag_max))
            cards[key][lag] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot(x, value, label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Lag')
    plt.legend(title='Cards:')
    plt.title('Lagging Check ' + target)
    plt.show(block=True)


def mainSmooth():
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()

    dfs = list(map(read_csv, files))

    cards = {}
    target = "Mold"

    arg_max = 2
    x = list(range(0, arg_max))
    for smooth in x:
        print("Smooth:", smooth)
        results = run(dfs, [1, 2, 3, 4], smooth, "entropy", target, range(0, 5), range(0, 5))
        for key, value in results.items():
            cards[key] = cards.get(key, np.zeros(arg_max))
            cards[key][smooth] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot(x, value, label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Smoothing')
    plt.legend(title='Cards:')
    plt.title('Smoothing Check ' + target)
    plt.show(block=True)


def mainEstimators():
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()

    dfs = list(map(read_csv, files))

    cards = {}
    target = "Mold"

    arg_max = 10
    x = list(range(0, arg_max))
    for smooth in x:
        print("Estimators:", smooth)
        results = run(dfs, [], 0, "entropy", target, range(0, 5), range(0, 3), smooth)
        for key, value in results.items():
            cards[key] = cards.get(key, np.zeros(arg_max))
            cards[key][smooth] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot(x, value, label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Smoothing')
    plt.legend(title='Cards:')
    plt.title('Smoothing Check ' + target)
    plt.show(block=True)


def mainCombined():
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()

    dfs = list(map(read_csv, files))

    cards = {}
    target = "Mold"

    results = run(dfs, [1, 2], 2, "entropy", target, range(0, 10), range(0, 5))
    for key, value in results.items():
        cards[key] = cards.get(key, 0)
        cards[key] = value

    ax = plt.gca()
    for key, value in cards.items():
        ax.plot([1, 2, 3], [value, value, value], label=str(key))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Dummy')
    plt.legend(title='Cards:')
    plt.title('Smooth 2 Lag [1,2] ' + target)
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


class Agent:
    def __init__(self, **options):
        self.options = options
        self.agents = {}

    @staticmethod
    def from_file(file):
        return Agent()

    @staticmethod
    def open_sample(file):
        df = pd.read_csv(file)
        df._path = file
        return df


    @staticmethod
    def preprocess(df, **options):
        split_gcm = options.get('split_gcm', False)


        # remove_gcms = {
        #     1: ["qcm_1", "qcm_5"],
        #     34: ["qcm_1", "qcm_2", "qcm_5"],
        #     35: ["qcm_1", "qcm_3"],
        #     36: ["qcm_1", "qcm_3", "qcm_5"],
        #     37: ["qcm_2", "qcm_3", "qcm_4", "qcm_5"],
        #     38: ["qcm_1", "qcm_3", "qcm_4"],
        #     39: ["qcm_2", "qcm_4"],
        # }
        #
        # card = df['card'].iloc[0]
        # if card in remove_gcms:
        #     for sensor in remove_gcms[card]:
        #         df.drop(sensor, axis=1, inplace=True)

        if split_gcm == False:
            df = Agent.single_preprocess(df, **options)
            df._category = "Combined"
            return [df]

        dfs = []
        columns = gcm_columns(df)
        for column in columns:
            new_df = df.copy(deep=True)

            for drop_column in columns:
                if drop_column != column:
                    new_df.drop(drop_column, axis=1, inplace=True)

            new_df = Agent.single_preprocess(new_df, **options)
            new_df._category = column
            dfs.append(new_df)

        return dfs

    @staticmethod
    def single_preprocess(df, **options):
        smoothing = options.get('smoothing', None)
        # filter_pass = options.get('filter_pass', {'N': 3, 'Wn': 0.08})
        filter_pass = options.get('filter_pass', None)
        strip_begin = options.get('strip_begin', 8)
        relative_begin = options.get('relative_begin', 8)
        rolling = options.get('rolling', None)
        lag = options.get('lag', None)
        drops = options.get('drop', [])

        # store for later use
        card = df['card'].iloc[0]
        sample = df['sample'].iloc[0]



        # drop sample id column
        df.drop("sample", axis=1, inplace=True)
        df.drop("card", axis=1, inplace=True)

        df.rename(columns={'humidiy': 'humidity'}, inplace=True)

        for drop in drops:
            df.drop(drop, axis=1, inplace=True)

        # get mean value of the first 8 seconds (didnt work because it used float and we need integers), so we just take the values at time=8
        if strip_begin is not None:
            relative_points = df.loc[df.time < relative_begin].iloc[-1]

        columns = sensor_columns(df)
        gcms_columns = gcm_columns(df)
        # we only handle 1 tags so 0 = no tags and 1 = mold
        df.tags = df.tags.apply(string_to_arr)

        mlb = MultiLabelBinarizer()
        tags = pd.DataFrame(mlb.fit_transform(df['tags']), columns=mlb.classes_, index=df.index)

        # store for later use
        tags_index = df.columns.get_loc("tags")
        df = df.drop('tags', axis=1).join(tags)

        for tag in tags_list:
            if tag not in df:
                df[tag] = 0

        # reorder columns. important.
        df = df.reindex(columns=list(df.columns[:tags_index]) + tags_order, copy=False)

        # transform to relative data
        for column in columns:
            df[column] -= relative_points[column]

        if rolling is not None:
            for column in columns:
                df[column] = df[column].rolling(window=3).mean()

        if filter_pass is not None:
            for column in gcms_columns:
                # N Filter order
                # Wn Cutoff frequency
                B, A = signal.butter(filter_pass['N'], filter_pass['Wn'], output='ba')
                df[column] = signal.filtfilt(B, A, df[column])

        # filtering is better
        if smoothing is not None:
            for column in columns:
                df[column] = df[column].groupby(df[column].index // smoothing).mean()
            df.time = df.time.groupby(df.time.index // smoothing).mean()
            df = df[:len(df.index) // smoothing]

        # remove 8 seconds from data
        if strip_begin is not None:
            df = df.drop(df[df.time < strip_begin].index).reset_index(drop=True)

        # debug
        #print(df)

        # add lagging data
        if lag is not None:
            for column in columns:
                index = df.columns.get_loc(column)
                # lagging by n prev values
                lags = lag

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
        # df.insert(3, "step", df.time.apply(time_to_state))
        # map time from secs to 0...1 where 0 is start and 1 is end
        # df.time = minmax_scale(df[['time']], copy=False)

        tags_index = df.columns.get_loc("Mold")
        df._sample = sample
        df._card = card
        df._tags_index = tags_index

        return df

    def get_classifier(self, df):
        if str(df._card) not in self.agents:
            return None
        return self.agents[str(df._card)]

    @staticmethod
    def graph_df(df, results=None, block=True):
        ax = plt.gca()
        for column in sensor_columns(df):
            df.plot(kind='line', x='time', y=column, ax=ax, figsize=(10, 5))
        # df.plot(kind='scatter',x='time',y="result",ax=ax,color='red')
        if results is not None:
            for index, r in enumerate(results):
                if r[1] == 1:
                    ax.axvline(x=df.time.iloc[index],
                               color='blue', linestyle='--', alpha=0.1)
                if r[0] == 1:
                    ax.axvline(x=df.time.iloc[index],
                               color='black', linestyle='--', alpha=0.1)
        plt.show(block=block)

    def classify(self, file):
        df = self.preprocess(file)
        classifier = self.get_classifier(df)

        if classifier is None:
            return None

        data = df.iloc[:, 0:df._tags_index].values
        pred = classifier.predict(data)

        tags_counter = {}
        for index, tag in enumerate(tags_order):
            tags_counter[index] = 0

        for x in pred:
            for index, value in enumerate(x):
                tags_counter[index] += value

        result = {}
        for index, counter in tags_counter.items():
            result[tags_order[index]] = (counter / len(pred)) * 100.0

        return result

    @staticmethod
    def graph_accuracy(dfs, target, **options):
        print("Starting routines")

        df_cards = {}
        agents = {}
        # split by card
        for member in dfs:
            card = member._card
            df_cards.setdefault(card, []).append(member)

        # split by category
        for key, dfs_card in df_cards.items():
            category = {}
            for member in dfs_card:
                category.setdefault(member._category, []).append(member)
            df_cards[key] = category

        shuffles = options.get("shuffles", 5)
        ml_samples = options.get("ml_samples", 5)
        classifier = options.get("classifier", DecisionTreeClassifier)
        options = options.get("classifier_options", {})

        results = {}
        for file_rand in range(0, shuffles):
            rand = random.Random(file_rand)
            for card_id, categories in df_cards.items():
                for category, samples in categories.items():
                    # split test and train, we take only small amount for testing because we have dont have enough data
                    targets = list(filter(lambda x: x[target].iloc[0] == 1, samples))
                    non_targets = list(filter(lambda x: x[target].iloc[0] != 1, samples))

                    print("Card id:", card_id)
                    print(target + " count:", len(targets))
                    print("Non-" + target + " count:", len(non_targets))

                    if (len(targets) < 3 or len(non_targets) < 3):
                        print("card does not have enough samples, skipping")
                        continue

                    accuracies = results.setdefault(card_id, {})

                    test_count_target = max(int(0.1 * len(targets)), 3)
                    test_count_non_target = test_count_target  # max(int(0.1 * len(not_mold)), 3)

                    rand.shuffle(targets)
                    rand.shuffle(non_targets)

                    train = itertools.chain(targets[:-test_count_target],
                                            non_targets[:-test_count_non_target])
                    test = itertools.chain(targets[-test_count_target:],
                                           non_targets[-test_count_non_target:])

                    print("Training size", target, ":", len(targets) - test_count_target,
                          "non-", target, ":", len(non_targets) - test_count_non_target)
                    print("Testing size", target, ":", test_count_target,
                          "non-", target, ":", test_count_non_target)

                    # combine all files
                    train = pd.concat(train, sort=False)
                    test = pd.concat(test, sort=False)

                    # print(train)

                    # we get all input data
                    x_train = train.iloc[:, 0:samples[0]._tags_index].values
                    # we get tags data
                    y_train = train.iloc[:, samples[0]._tags_index:].values

                    # testing not used here because we do manually testing
                    # we get all input data
                    x_test = test.iloc[:, 0:samples[0]._tags_index].values
                    # we get tags data
                    y_test = test.iloc[:, samples[0]._tags_index:].values

                    #print(x_test)
                    #print(y_test)

                    print("Training...")
                    accuracy = 0
                    for algo_rand in range(0, ml_samples):
                        options['random_state'] = algo_rand
                        regressor = classifier(**options)
                        # train
                        regressor.fit(x_train, y_train)

                        y_pred = regressor.predict(x_test)
                        # print("confusion_matrix: ", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
                        # print(classification_report(y_test, y_pred))
                        for index, tag in enumerate(tags_order):
                            if tag != target:
                                y_test[:, index] = 0
                                y_pred[:, index] = 0

                        print("accuracy_score: ", accuracy_score(y_test, y_pred) * 100)
                        accuracy += accuracy_score(y_test, y_pred) * 100
                    print("#######" + str(accuracies.get(category, 0)))
                    accuracies[category] = accuracies.get(category, 0) + accuracy / ml_samples

        for card_id, categories in results.items():
            for cat, result in categories.items():
                categories[cat] /= shuffles

        ax = plt.gca()

        for card, categories in results.items():
            for cat, result in categories.items():
                plt.scatter("Card " + str(card), result)
                plt.text("Card " + str(card), result, " " + cat, fontsize=9)
        #for key, value in results.items():
        #    ax.plot([1, 2, 3], [value, value, value], label=str(key))

        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Cards')
        plt.legend(title='IDs:')
        plt.title(target)
        plt.show(block=True)

        return results


if True:
    files = glob.glob(os.path.join('data', "*.csv"))
    files.sort()
    print("Preprocessing...")

    dfs = []
    for file in files:
        df = Agent.open_sample(file)
        #r = Agent.preprocess(df, split_gcm=False, strip_begin=15, relative_begin=7)
        r = Agent.preprocess(df, split_gcm=False, strip_begin=10, relative_begin=10)
        #r = Agent.preprocess(df, split_gcm=True, filter_pass={'N': 3, 'Wn': 0.08})
        dfs.extend(r)

    #Agent.graph_accuracy(dfs, "Mold", shuffles=5, classifier=RandomForestClassifier)
    Agent.graph_accuracy(dfs, "Pesticide", shuffles=20, classifier_options={'criterion': "entropy"})

#mainLag()

# train(open('./samples.zip', 'r'))
# train(urllib.request.urlopen('http://samples.zip'))

# import requests, zipfile, io
# r = requests.get(zip_file_url)
# z = zipfile.ZipFile(io.BytesIO(r.content))

# mainEstimators()

# from sklearn.tree import export_graphviz
# import pydotplus
# from IPython.display import Image
# agents = open_zip_agent("./agents.zip")

# dot_data = StringIO()
# export_graphviz(agents['37'], out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png("./graph.png")

file = glob.glob(os.path.join('data', "*.csv"))[38]
#df = read_csv(file)
#df = preprocess(df, [], 0)

df = Agent.open_sample(file)
r = Agent.preprocess(df, strip_begin=12, relative_begin=8)
print(r[0])
graph_for_mold(r[0], [])



# file = glob.glob(os.path.join('data', "*.csv"))[0]
# df = read_csv(file)
# df = preprocess(df, [], 1)
# print(df)
# graph_for_mold(df, [])

# file = glob.glob(os.path.join('data', "*.csv"))[0]
# df = read_csv(file)
# df = preprocess(df, [], 2)
# print(df)
# graph_for_mold(df, [])

# file = glob.glob(os.path.join('data', "*.csv"))[0]
# df = read_csv(file)
# df = preprocess(df, [], 3)
# print(df)
# graph_for_mold(df, [])