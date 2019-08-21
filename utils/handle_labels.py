import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# convert string to list : used for reuse categories
def convert_str_to_list(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: literal_eval(x))
    print(type(df[column_name].values[0]))
    print(df[column_name].values[0])

    return df


# tag counts and label columns
def get_tag_counts_and_labels(df):
    tags_counts = Counter()

    for tags in df['categories'].values:
        for tag in list(tags):
            tags_counts[tag] += 1

    most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:]

    for item in most_common_tags:
        print(item[0], ":", item[1])

    target_columns = list(tags_counts.keys())

    return tags_counts, target_columns


# drop labels
def drop_labels(df, labels):
    """
    Drop some of labels

    Args:
    df - Dataframs
    labels - List of labels name to drop

    Returns:
    new_df -  new dataframe
    """
    # remove labels in categories list
    new_df = df.copy()
    new_df['categories'] = new_df['categories'].apply(lambda row: [item for item in row if item not in labels])

    # remove columns
    new_df = new_df.drop(labels, axis=1)

    # remove columns which have no labels after removing labels
    new_df['number_of_labels'] = new_df['categories'].apply(lambda row: len(row))
    new_df = new_df[new_df['number_of_labels'] != 0].reset_index(drop=True)
    new_df = new_df.drop(['number_of_labels'], axis=1)

    return new_df


# group some of labels
def group_labels(df, labels_to_group, new_label):
    '''
    Group some of labels

    Args:
        df - dataframe
        labels_to_group -  List of labels you want to group
        new_label -  string - new label name of grouped labels

    Returns:
        new_df - dataframe after grouped
    '''
    new_df = df.copy()

    # generate new labels by group labels
    def create_new_label(row, labels):
        new_label = 0  # initialize new label
        for label in labels:
            if row[label] == 1:
                new_label = 1  # if one of labels in grouped labels is 1 the new label is 1
        return new_label

    new_df[new_label] = df.apply(lambda row: create_new_label(row, labels_to_group), axis=1)

    # drop old labels
    new_df = new_df.drop(labels_to_group, axis=1)

    # generate list of new_categories

    return new_df

# group labels
def group_labels_new(df, labels_to_group, new_label):
    '''
    Group some of labels

    Args:
        df - dataframe
        labels_to_group -  List of labels you want to group
        new_label -  string - new label name of grouped labels

    Returns:
        new_df - dataframe after grouped
    '''
    new_df = df.copy()

    # generate new labels by group labels
    def create_new_label(row, labels):
        new_label = 0  # initialize new label
        for label in labels:
            if row[label] == 1:
                new_label = 1  # if one of labels in grouped labels is 1 the new label is 1
        return new_label

    new_df[new_label] = df.apply(lambda row: create_new_label(row, labels_to_group), axis=1)

    # generate list of new_categories

    return new_df


# labels counter
def categories_count(df, target_labels, verbose=True):
    '''
    count number of labels of each catergories

    Args:
        df - Dataframe
        target_labels - List of labels to count
        verbose - Boolean - whether to show result

    Returns:
        Count Dict

    '''
    # calculate count
    count_cat = {}
    for label in target_labels:
        count_cat[label] = df[df[label] == 1].shape[0]
    # print result
    if verbose == True:
        for k, v in count_cat.items():
            print(k + ' : ' + str(v))
    return count_cat

# show imbalance of each label
def get_imbalance(df, target_columns, show_plot = True):
    '''
    show imbalance of a each categories
    
    Args:
        df - Dataframe - data
        target_columns - List of column name you want to show
    
    Returns: imbalance_result - dictionary of imbalanced information
    '''
    # class imbalance
    sns.set(font_scale = 0.8)
    plt.figure(figsize=(12,20))
    
    imbalance_result = {}

    # plot result
    for index, label in enumerate(target_columns):
        #print('================================================================')
        #print(index, label)
        #print(df[label].value_counts()[0], ' ',df[label].value_counts()[1])

        binary_class = df[label].value_counts()
        
        # add result
        imbalance_result[label] = {}
        for item in binary_class.index:
            imbalance_result[label][item] = binary_class[item]
        if show_plot == True:
            plt.subplot(6,4,index+1)
            plt.bar(binary_class.index, binary_class.values)
            plt.xticks([0,1])
            plt.xlabel(label, fontsize=10)
    
    return imbalance_result

# number of commits of each labels
def label_distribution(df, target_columns):
    '''
    show label distribution
    
    Args:
        df - Dataframe - data
        target_columns - List of column name you want to show
    
    Returns: None
    '''
    sns.set(font_scale = 0.8)
    plt.figure(figsize=(20,10))
    ax= sns.barplot(target_columns, df[target_columns].sum().values)
    plt.title("Commits in each category", fontsize=15)
    plt.ylabel('Number of commits', fontsize=15)
    plt.xlabel('Commit Type ', fontsize=15)
    #adding the text labels
    rects = ax.patches
    labels = df[target_columns].sum().values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
    plt.show()

# number of commits have multiple labels
def number_of_labels(df, target_columns):
    '''
    show label distribution
    
    Args:
        df - Dataframe - data
        target_columns - List of column name you want to show
    
    Returns: multi_counts - pandas Series
    '''
    rowSums = df[target_columns].sum(axis=1)
    multiLabel_counts = rowSums.value_counts()
    multiLabel_counts = multiLabel_counts.iloc[0:]
    print(multiLabel_counts)

    sns.set(font_scale = 1)
    plt.figure(figsize=(10,6))
    ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
    plt.title("Commits having multiple labels ")
    plt.ylabel('Number of commits', fontsize=15)
    plt.xlabel('Number of labels', fontsize=15)

    #adding the text labels
    rects = ax.patches
    labels = multiLabel_counts.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    plt.show()
    
    return multiLabel_counts