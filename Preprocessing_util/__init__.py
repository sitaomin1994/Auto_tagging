"""
This script is a utility file for manipulating pandas DataFrame containing Roots
and Json files containing GumTree clusters
"""
import pandas as pd
import json
import re
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class Processing:

    def group_labels_new(self, df, labels_to_group, new_label):
        """
        Groups labels provided in a list into a single label
        and returns a DataFrame
        :param df: Current DataFrame
        :param labels_to_group: List of labels to be grouped
        :param new_label: <string> new label name of the grouped labels
        :return: DataFrame after grouping
        """
        new_df = df.copy()

        # Generate new labels by group labels
        def create_new_label(row, labels):
            new_label = 0  # Initialize new label
            for label in labels:
                if row[label] == 1:
                    new_label = 1  # If one of the labels in grouped labels is 1 the ne label is 1
            return new_label

        new_df[new_label] = df.apply(lambda row: create_new_label(row, labels_to_group), axis=1)

        # Generate list of new categories
        return new_df

    def parse_update(self, update_list): 
        """
         Gets the unique tokens from UPDATE operations in **update_list 
         :param update_list: list of update actions 
         :return a list containing unique UPDATE tokens 
        """
        update_set = set()
        for update in update_list:
            token = update.split(":", 1)[0]
            token = token.replace("UPD", "UPDATE")
            update_set.add(token)
        return list(update_set)

    def parse_json(self, filepath, *args):
        """
         This function parses the json of each commit json file
        :param filepath: File path which contains all json files
        :param args: optional file path argument
        :return: List of parsed json clusters and commit csha
        """
        if len(args) > 1:
            print("Please provide only 1 extra file path")
            return None

        files_json = []
        commit_ids = []

        if len(args) == 1 and isinstance(args[0], list):
            files = args[0]
            # each commits
            # files = os.listdir(filepath)
            for path in files:
                commit_id = path.split("_")[1].split(".")[0]
                try:
                    if os.stat(filepath + path).st_size != 0 and path != 'desktop.ini':
                        with open(filepath + path, encoding="utf8") as f:
                            data = json.load(f)
                            files_list = []
                            # each file in commits
                            for file in data['files']:
                                # parse only cluster file
                                for key in file.keys():
                                    if re.match('^.*_cluster$', key):
                                        actions_list = []
                                        actions = file[key]['actions']
                                        # each action in file
                                        for action in actions:
                                            if re.search('^U.*$', action['root']):
                                                parse_up = self.parse_update(action['operations'])
                                                for update in parse_up:
                                                    actions_list.append(update)
                                            else: 
                                               actions_list.append(action['root'])
                                        if len(actions_list) != 0:  
                                            files_list.append(actions_list)
                        if len(files_list) != 0:
                            files_json.append(files_list)
                            commit_ids.append(commit_id)
                except FileNotFoundError as e:
                    continue
            assert (len(commit_ids) == len(files_json))

        if len(args) == 0:
            files = os.listdir(filepath)
            for path in files:
                if os.stat(filepath + path).st_size != 0 and path != 'desktop.ini':
                    commit_id = path.split("_")[1].split(".")[0]
                    with open(filepath + path, encoding="utf8") as f:
                            data = json.load(f)
                            files_list = []
                            # each file in commits
                            for file in data['files']:
                                # parse only cluster file
                                for key in file.keys():
                                    if re.match('^.*_cluster$', key):
                                        actions_list = []
                                        actions = file[key]['actions']
                                        # each action in file
                                        for action in actions:
                                            if re.search('^U.*$', action['root']):
                                                parse_up = self.parse_update(action['operations'])
                                                for update in parse_up:
                                                    actions_list.append(update)
                                            else: 
                                               actions_list.append(action['root'])
                                        if len(actions_list) != 0:  
                                            files_list.append(actions_list)
                    if len(files_list) != 0:
                        files_json.append(files_list)
                        commit_ids.append(commit_id)
        return files_json, commit_ids

    def denoise_token(self, text):    
        # parse token
        if re.match('^("|\').*"$', text):
            result = 'TextElement'
    #     elif re.match("^'.", text): 
    #         result = 'TextElement'
        elif re.match('.*(a_href|A_HREF).*', text): 
            result = 'HTMLTagElement'
    #     elif re.match('.*A_HREF.*', text): 
    #         result = 'HTMLTagElement'
        elif re.match('^[0-9].*', text): 
            result = 'NumberLiteral'
        elif re.match('^http:.*$', text):
            result = 'Url'
        elif re.match('@.*', text): 
            result = 'TagElement'
        elif re.match('^(style|href=).*', text): 
            result = 'HTMLTagElement'
    #     elif re.match('^href=.*', text): 
    #         result = 'HTMLTagElement'
        else:
            result = text
        return result


    def preprocess_roots(self, files_data):
        """
         This function pre-processes roots in json file
        :param files_data: List of json files containing GumTree clusters
        :return: dictionary mapping <roots, pos>, multi-list of commits, frequency dictionary for roots
        """
        counting = {}
        for file_index, files in enumerate(files_data):
            for root_index, roots in enumerate(files):
                for action_index, actions in enumerate(roots):
                    temp = actions.split(' at ')[0].strip()
                    tempq = []
                    if temp.startswith('INS'):
                        tempq.append('INS')
                        words = [temp.split('INS ')[1].split('to ')[0].strip()] + [
                            temp.split('INS ')[1].rsplit('to ')[-1].strip()
                        ]
                        for items in words:
                            items = items.split(':')[0].strip()
                            tempq.append(items)
                        if tempq[1] == 'TextElement' and tempq[-1] not in ['TagElement', 'TextElement']:
                            tempq[-1] = 'EmptyString'
                        temp = '_'.join(tempq)

                    if temp.startswith('UPDATE'):
                        if "from" not in temp.split():
                            tempq = temp.split()
                            temp = '_'.join(tempq)
                        elif temp.split()[2] == "from":
                            if len(temp.split()) == 5:
                                tempq.append("UPDATE")
                                word = self.denoise_token(temp.split()[4]) 
                                tempq.append(word)
                                temp = '_'.join(tempq)
                            if len(temp.split()) == 4:
                                tempq.append("UPDATE")
                                word = "EmptyString"
                                tempq.append(temp.split()[1])
                                tempq.append(word)
                                temp = '_'.join(tempq)
                        else:
                            if "<a" in temp.split(): 
                                temp = temp.replace("<a href", "a_href")
                            if "<A" in temp.split(): 
                                temp = temp.replace("<A HREF", "A_HREF")
                            if "*" in temp.split(): 
                                temp = temp.replace("*", "")
                            tempq.append('UPDATE')
                            words = [temp.split('UPDATE from ')[1].split('to ')[0].strip()] + ['to'] + [
                                temp.split('UPDATE from ')[1].rsplit('to ')[-1].strip()
                            ]
                            words[0] = self.denoise_token(words[0])
                            words[2] = self.denoise_token(words[2])
                            tempq.append(words[0])
                            tempq.append(words[2])
                            temp = '_'.join(tempq)
                            if re.search(".*<.*", temp): 
                                temp = "UPDATE_HTMLTagElement"
                            elif re.search(".*https:.*", temp): 
                                temp = "UPDATE_Url"
                            elif re.search(".*\.(.*)", temp): 
                                temp = "UPDATE_MethodInvocation"
                            if len(temp.split()) > 1:
                                temp = "UPDATE_TextElement"
                    if temp.startswith('MOVE'):
                        temp2 = temp.split('from ')[1].strip()
                        tempq.append('MOVE')
                        tempq.append(temp2.split(':')[0].strip())
                        temp = '_'.join(tempq)

                    if temp.startswith('DEL'):
                        tempq.append('DEL')
                        tempq.append(temp.split('DEL ')[1].split(':')[0].strip())
                        temp = '_'.join(tempq)
                    temp = temp.replace(' ', '_')
                    if temp != "":
                        counting[temp] = counting.get(temp, 0) + 1
                        files_data[file_index][root_index][action_index] = temp
        dic = {}
        i = 0
        for k, v in counting.items():
            dic[k] = i
            i += 1
        return dic, files_data, counting

    def actions2sentence(self, datas):
        """
        This function converts the actions in a file into a sentence
        :param datas:
        :return: multi-list
        """
        data_total = []
        for files in datas:
            data4file = []
            for roots in files:
                sentence = ' '.join(roots)
                data4file.append(sentence)
            data_total.append(data4file)
        return data_total

    def permutate_files(self, csha, training_data):
        """
         This function permutes the order of files with training_data
        :param csha: commit sha
        :param training_data: List of commit data
        :return: Dictionary mapping <csha, training data>
        """
        commits_dic = dict()
        for sha, training_file in zip(csha, training_data):
            commits_dic[sha] = []
            if len(training_file) <= 5:
                tmp_permutate = list(itertools.permutations(training_file))
                for permutated_file in tmp_permutate:
                    commits_dic[sha].append(list(permutated_file))
            else:
                commits_dic[sha].append(training_file)
        return commits_dic

    def expand_list(self, commits_labels_df):
        """
         This function expands list of files in DataFrame
        :param commits_labels_df:
        :return:
        """
        s = commits_labels_df.apply(lambda x: pd.Series(x['Files']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = "Files"
        commits_labels_df = commits_labels_df.drop("Files", axis=1)
        commits_labels_df = commits_labels_df.join(s)
        return commits_labels_df

    def concat_files_to_sentence(self, expanded_train_list):
        """
         This function concatenates the all files in a commit into a sentence
        :param expanded_train_list:
        :return:
        """
        tmp_list = []
        for items in expanded_train_list:
            concat_data = " ".join(items)
            tmp_list.append(concat_data)
        return tmp_list

    def get_seqlength(self, training_data):
        max_root_len = 0
        seqlength_list = []
        for item in training_data:
            seqlength_list.append(len(item.split()))
            if len(item.split()) > max_root_len:
                max_root_len = len(item.split())
        return max_root_len, seqlength_list

    def plot_hist(self, seqlength_list, xlim, ylim):
        """
         plotting function
        :param seqlength_list: List of the length of each file in training data
        :param xlim: tuple of x limit
        :param ylim: tuple of y limit
        :return:
        """
        plt.figure(figsize=(20, 10))
        number_of_files = np.array(seqlength_list)
        bincount = np.bincount(seqlength_list)
        x = np.arange(1, len(bincount) + 1)
        n, bins, patches = plt.hist(seqlength_list, x)
        plt.xlim(xlim)
        plt.ylim(ylim)

    def get_file_threshold(self, number_of_files, threshold=0.95):
        """
         Get padding threshold for files dimension
        :param number_of_files: Array of the array of the number of files in each commits
        :param threshold: Drop all commits with the number of files beyond the threshold
        :return: threshold number
        """

        total_files = len(number_of_files)
        number_of_files = np.array(number_of_files)
        bincount = np.bincount(number_of_files)

        sum_file = 0
        padding_files_threshold = 0
        for index, item in enumerate(bincount):
            sum_file += item
            # print(index,item)
            # print(sum_file)
            if sum_file > threshold * total_files:
                padding_files_threshold = index
                break
        return padding_files_threshold

    def load_embedding(self, filename):
        """
        load embedding as python dictionary {root<str>: embeddings<np_array>}
        :param filename: embedding.txt
        :return: dictionary object mapping root to embeddings
        """
        if not os.path.exists(filename):
            print("please run 'Stored Pre-Trained Embeddings Cell!'")
        else:
            with open(filename, "r") as f:
                lines = f.readlines()
                f.close()
                # create map of words to vectors
                embedding = dict()
                for line in lines:
                    comp = line.split()
                    # map of <str, numpy array>
                    embedding[comp[0]] = np.asarray(comp[1:], dtype='float32')
                return embedding

    def tokenize_train_data(self, concat_data, dic):
        """
         Utility function for embedding script.
         This function is used to tokenize concatenated data from --> concat_files_to_sentence
        :param concat_data: concatenated data
        :param dic: Dictionary mapping <roots, pos>
        :return: list tokenized commits
        """
        data = []
        for word in concat_data.split():
            index = dic.get(word)
            if index is None:  # for debug
                print(word)  # for debug
            data.append(index)
        return data

    def viz_data(self, embed_mat, reverse_dictionary, n_viz=20):
        """
         Visualize embedding matrix
        :param embed_mat: embedding matrix
        :param reverse_dictionary: Dictionary mapping <pos, root>
        :param n_viz: number of embeddings to visualize
        :return: plot
        """
        tsne = TSNE()
        embed_tsne = tsne.fit_transform(embed_mat[:n_viz, :])
        fig, ax = plt.subplots(figsize=(14, 14))
        for idx in range(n_viz):
            plt.scatter(*embed_tsne[idx, :], color="#ff8243")
            plt.annotate(reverse_dictionary[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

    def split_df_train_test_val(self, df, train_split, val_split, seed=42):
        """
        This function splits a given DataFrame into Training, Validation and Testing set
        :param df: DataFrame to split
        :param train_split: Percentage of training set
        :param val_split: Percentage of testing set
        :param seed: Random seed initializer
        :return: training, validation and testing DataFrames
        """
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(train_split * m)
        validate_end = int(val_split * m) + train_end
        train = df.iloc[perm[:train_end]]
        validate = df.iloc[perm[train_end:validate_end]]
        test = df.iloc[perm[validate_end:]]
        return train, validate, test
