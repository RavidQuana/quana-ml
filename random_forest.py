import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import constants
import sample_file_parser


def runRandomForest(dataframe):
    featureList = []
    tagsCoulmnIndex = dataframe.columns.get_loc(sample_file_parser.tags_col_name)
    channels_list = dataframe[constants.channel_out_col_title].unique()
    
    features_list = dataframe.columns[(tagsCoulmnIndex+1):]
    
    for channel in channels_list:
        for feature in dataframe.columns[(tagsCoulmnIndex+1):]:
            featureList.append(channel + "_" + feature)
    randomForestDF = pd.DataFrame(columns=[sample_file_parser.tags_col_name] + featureList)
    for product in dataframe[sample_file_parser.product_col_name].unique():
        rows_array = dataframe.loc[dataframe[sample_file_parser.product_col_name] == product]     
        out_rows = []
        for channel in rows_array[constants.channel_out_col_title].unique():
            channel_rows = rows_array.loc[rows_array[constants.channel_out_col_title] == channel]
            i = 0
            for index, row in channel_rows.iterrows():
                row_flat = row.values.tolist()
                if len(out_rows) < (i + 1):
                    out_rows.append(row_flat[3:])
                else:
                    if len(out_rows[i]) == 0:
                        out_rows[i] += row_flat[3:]
                    else:
                        out_rows[i] += row_flat[4:]
                i += 1
        j = 0
        while j < i:
            randomForestDF.loc[len(randomForestDF)] = out_rows[j]
            j += 1
    #RandomForest
    
    x = randomForestDF[featureList]
    
    y = randomForestDF[ sample_file_parser.tags_col_name]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    clf=RandomForestClassifier(n_estimators=1000)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))    
  
    feature_imp = pd.Series(clf.feature_importances_,index=featureList).sort_values(ascending=False)
    top_feature_imp = feature_imp[0:35]
    y_pos = np.arange(len(top_feature_imp))
    plt.figure(figsize=(20,10))
    bars = plt.bar(y_pos, top_feature_imp, align='center', alpha=1, color=np.random.rand(len(top_feature_imp), 3))
    
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % height, ha='center', va='bottom', fontsize=6)
    #features_names = []
    #for ind in indices:
     #   features_names.append(randomForestDF.columns[ind])
    plt.xticks(y_pos, top_feature_imp.index, rotation='vertical', fontsize=8)
    plt.yticks(np.arange(min(feature_imp), max(feature_imp) , step=0.01), fontsize=6)
    #plt.xticks(y_pos, names, fontsize=8)
    # Add labels to your graph
    plt.ylabel('Feature Importance Score')
    plt.xlabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.subplots_adjust(bottom=0.4)
    datestr = constants.get_date_str()
    graphs_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir    
    graph_file_name = "random_forest_" + datestr + ".png"
    out_dir_path = graphs_results_dir + constants.path_random_forest_graphs_dir
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    plt.savefig(out_dir_path + graph_file_name)
    plt.show()    
    