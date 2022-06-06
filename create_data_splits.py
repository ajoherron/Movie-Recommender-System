#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Creates Train, Val and Test Splits from Input Dataset
Usage:
    $ python create_data_splits.py --config=/path/to/config/file
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import yaml

import numpy as np
import random
import pandas as pd
from IPython.display import display


def load_input_data(input_data_config):
    """Loads the Input Ratings, Movies, Links and Tags Data"""
    data_dir = input_data_config['data_dir']

    #Convert small ratings file to dataframe
    input_ratings_path = input_data_config['ratings_csv']
    input_ratings_path = os.path.join(data_dir, input_ratings_path)
    ratings_df = pd.read_csv(input_ratings_path)

    #Convert small movies file to dataframe
    input_movies_path = input_data_config['movies_csv']
    input_movies_path = os.path.join(data_dir, input_movies_path)
    movies_df = pd.read_csv(input_movies_path)

    #Convert small links file to dataframe
    input_links_path = input_data_config['links_csv']
    input_links_path = os.path.join(data_dir, input_links_path)
    links_df = pd.read_csv(input_links_path)

    #Convert small tags file to dataframe
    input_tags_path = input_data_config['tags_csv']
    input_tags_path = os.path.join(data_dir, input_tags_path)
    tags_df = pd.read_csv(input_tags_path)

    return ratings_df, movies_df, links_df, tags_df


def create_train_set(ratings_df, movies_df, links_df,
                     tags_df, output_data_config,
                     train_split=0.7, debug=False):
    """Creates the Train Set from Input Files"""

    # 1.1 Create Train Set by Sampling 70% of Entries for Each User
    train_ratings_df = ratings_df.groupby('userId').sample(frac=train_split)

    if debug:
        # check the min count of the user counts.
        display(train_ratings_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

        # check the user corresponding to the min user counts.
        # min_count_user = train_ratings_df.groupby('userId').size().to_frame('user_counts').idxmin()[0]
        # display(train_ratings_df.groupby('userId').size().to_frame('user_counts').loc[[min_count_user]].reset_index())


    # 1.2 Merge Ratings Data with Other Datasets to Create Consolidated Train Set
    train_consolidated_df = train_ratings_df.merge(movies_df,
                                                   left_on='movieId',
                                                   right_on='movieId',
                                                   how='inner')

    train_consolidated_df = train_consolidated_df.merge(links_df,
                                                        left_on='movieId',
                                                        right_on='movieId',
                                                        how='inner')

    train_consolidated_df = train_consolidated_df.merge(tags_df,
                                                        left_on=['userId', 'movieId'],
                                                        right_on=['userId', 'movieId'],
                                                        how='left',
                                                        suffixes=['_ratings', '_tags'])

    if debug:
        display(train_consolidated_df.head(2))
        # check the min count of the user counts.
        display(train_consolidated_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

        # check the user corresponding to the min user counts
        # min_count_user = train_ratings_df.groupby('userId').size().to_frame('user_counts').idxmin()[0]
        # display(train_ratings_df.groupby('userId').size().to_frame('user_counts').loc[[min_count_user]].reset_index())

    # 1.3 Save Train Ratings and Consolidated Train Set
    train_ratings_df_savepath = os.path.join(output_data_config['data_dir'],
                                             output_data_config['train']['ratings_csv'])

    print(f"Saving train_ratings_df at {train_ratings_df_savepath}")
    train_ratings_df.to_csv(train_ratings_df_savepath, index=False, header=True)

    # save consolidated train csv
    train_consolidated_df_savepath = os.path.join(output_data_config['data_dir'],
                                                  output_data_config['train']['consolidated_csv'])

    print(f"Saving train_consolidated_df at {train_consolidated_df_savepath}")
    train_consolidated_df.to_csv(train_consolidated_df_savepath, index=False, header=True)

    return train_ratings_df, train_consolidated_df


def create_val_and_test_set(train_ratings_df, ratings_df, movies_df,
                   links_df, tags_df, output_data_config,
                   val_split=0.5, debug=False):
    """Creates the Val Set from Input Files and Train Set"""
    # 1.4 Drop Rows from Training Set from the Input Dataset
    train_ratings_cond = ratings_df.index.isin(train_ratings_df.index.tolist())
    not_train_ratings_df = ratings_df.loc[~train_ratings_cond]

    # ensure there is no overlap between the two dataframes
    train_key = train_ratings_df['userId'].astype(str) + '-' + train_ratings_df['movieId'].astype(str)
    train_ratings_key = train_key.values.tolist()

    not_train_ratings_key = not_train_ratings_df['userId'].astype(str) + '-' + not_train_ratings_df['movieId'].astype(str)
    not_train_ratings_key = not_train_ratings_key.values.tolist()

    assert set(train_ratings_key).isdisjoint(set(not_train_ratings_key))

    if debug:
        # check the min count of the user counts.
        display(not_train_ratings_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

        # check the user corresponding to the min user counts.
        # min_count_user = train_ratings_df.groupby('userId').size().to_frame('user_counts').idxmin()[0]
        # display(not_train_ratings_df.groupby('userId').size().to_frame('user_counts').loc[[min_count_user]].reset_index())


    # 1.5 Create Validation Split By Sampling 50% of Remaining Dataset
    val_and_test_userIds = not_train_ratings_df['userId'].unique()
    user_Id_list = list(val_and_test_userIds)

    random.shuffle(user_Id_list)

    split = int(len(user_Id_list) * val_split)

    user_Id_val = user_Id_list[split:]
    user_Id_test = user_Id_list[:split]

    user_Id_val = sorted(user_Id_val)
    user_Id_test = sorted(user_Id_test)

    # index the val set users
    val_set_cond = not_train_ratings_df['userId'].isin(user_Id_val)
    val_ratings_df = not_train_ratings_df.loc[val_set_cond]

    # index the test set users
    test_set_cond = not_train_ratings_df['userId'].isin(user_Id_test)
    test_ratings_df = not_train_ratings_df.loc[test_set_cond]


    if debug:
        # check the min count of the user counts.
        display(val_ratings_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

        # check the user corresponding to the min user counts.
        # min_count_user = train_ratings_df.groupby('userId').size().to_frame('user_counts').idxmin()[0]
        # display(val_ratings_df.groupby('userId').size().to_frame('user_counts').loc[[min_count_user]].reset_index())


    # 1.6 Create Consolidated Validation Dataset
    val_consolidated_df = val_ratings_df.merge(movies_df,
                                               left_on='movieId',
                                               right_on='movieId',
                                               how='inner')

    val_consolidated_df = val_consolidated_df.merge(links_df,
                                                    left_on='movieId',
                                                    right_on='movieId',
                                                    how='inner')

    val_consolidated_df = val_consolidated_df.merge(tags_df,
                                                    left_on=['userId', 'movieId'],
                                                    right_on=['userId', 'movieId'],
                                                    how='left',
                                                    suffixes=['_ratings', '_tags'])

    if debug:
        # check the min count of the user counts.
        display(val_ratings_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

        # check the user corresponding to the min user counts.
        # min_count_user = train_ratings_df.groupby('userId').size().to_frame('user_counts').idxmin()[0]
        # display(val_ratings_df.groupby('userId').size().to_frame('user_counts').loc[[min_count_user]].reset_index())


    # 1.7 Save Ratings Val and Consolidated Validation Set
    val_ratings_df_savepath = os.path.join(output_data_config['data_dir'],
                                           output_data_config['val']['ratings_csv'])

    print(f"Saving val_ratings_df at {val_ratings_df_savepath}")
    val_ratings_df.to_csv(val_ratings_df_savepath, index=False, header=True)

    # save the consolidated val data
    val_consolidated_df_savepath = os.path.join(output_data_config['data_dir'],
                                                output_data_config['val']['consolidated_csv'])

    print(f"Saving val_consolidated_df at {val_consolidated_df_savepath}")
    val_consolidated_df.to_csv(val_consolidated_df_savepath, index=False, header=True)


    # 1.8 Create Consolidated Test Dataset
    test_consolidated_df = test_ratings_df.merge(movies_df,
                                                 left_on='movieId',
                                                 right_on='movieId',
                                                 how='inner')

    test_consolidated_df = test_consolidated_df.merge(links_df,
                                                      left_on='movieId',
                                                      right_on='movieId',
                                                      how='inner')

    test_consolidated_df = test_consolidated_df.merge(tags_df,
                                                      left_on=['userId', 'movieId'],
                                                      right_on=['userId', 'movieId'],
                                                      how='left',
                                                      suffixes=['_ratings', '_tags'])

    if debug:
        # check the min count of the user counts.
        display(test_ratings_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

        # check the user corresponding to the min user counts.
        # min_count_user = train_ratings_df.groupby('userId').size().to_frame('user_counts').idxmin()[0]
        # display(test_ratings_df.groupby('userId').size().to_frame('user_counts').loc[[min_count_user]].reset_index())


    # 1.10 Save Ratings Test and Consolidated Test Set
    test_ratings_df_savepath = os.path.join(output_data_config['data_dir'],
                                            output_data_config['test']['ratings_csv'])

    print(f"Saving test_ratings_df at {test_ratings_df_savepath}")
    test_ratings_df.to_csv(test_ratings_df_savepath, index=False, header=True)

    test_consolidated_df_savepath = os.path.join(output_data_config['data_dir'],
                                                 output_data_config['test']['consolidated_csv'])

    print(f"Saving test_consolidated_df at {test_consolidated_df_savepath}")
    test_consolidated_df.to_csv(test_consolidated_df_savepath, index=False, header=True)

    return val_ratings_df, val_consolidated_df, test_ratings_df, test_consolidated_df


def verify_data_splits(train_ratings_df, val_ratings_df,
                       test_ratings_df, ratings_df):
    """Verifies that the data splits meet necessary conditions"""

    # 1.11 Verify The Three Datasets Do Not Contain Common Rows
    train_key = train_ratings_df['userId'].astype(str) + '-' + train_ratings_df['movieId'].astype(str)
    train_key = train_key.values.tolist()

    val_key = val_ratings_df['userId'].astype(str) + '-' + val_ratings_df['movieId'].astype(str)
    val_key = val_key.values.tolist()

    test_key = test_ratings_df['userId'].astype(str) + '-' + test_ratings_df['movieId'].astype(str)
    test_key = test_key.values.tolist()

    assert set(train_key).isdisjoint(set(val_key))
    assert set(train_key).isdisjoint(set(test_key))
    assert set(val_key).isdisjoint(set(test_key))

    # 1.12 Verify The Train Set Contains At Least One Review Per User
    userIds = ratings_df['userId'].unique().tolist()

    train_userIds = train_ratings_df['userId'].unique().tolist()
    train_missing_userIds = set(train_userIds) - set(userIds)

    # train cond
    assert len(train_missing_userIds) == 0

    # 1.13 Verify The Val and Test sets don't contain common users
    val_userIds = val_ratings_df['userId'].unique().tolist()
    test_userIds = test_ratings_df['userId'].unique().tolist()

    common_userIds = set(val_userIds).intersection(set(test_userIds))

    assert len(common_userIds) == 0

    return


def main(config):
    '''Main Execution for Creating Data Splits

    Args:
        config : dict
            user-defined params to use when creating data splits

    Returns

    '''
    use_small_data = config['use_small_data']

    if use_small_data:
        data_config = config['small']
    else:
        data_config = config['large']

    # define input and output data configs
    input_data_config = data_config['input']
    output_data_config = data_config['output']
    debug = config['debug']

    if not os.path.exists(output_data_config['data_dir']):
        os.makedirs(output_data_config['data_dir'])

    # load the input data files
    ratings_df, movies_df, links_df, tags_df = load_input_data(input_data_config)

    # Part 1: Partition data into train / validation / test splits
    # 70% of all observations to training set
    # Remaining 30%: split 50% (by user) into test, and 50% into validation
    if debug:
        display(ratings_df.groupby('userId').size().to_frame('user_counts').reset_index().describe()['user_counts'].to_frame().T)

    # create the train ratings and consolidated datasets
    train_ratings_df, train_consolidated_df = create_train_set(ratings_df,
                                                               movies_df,
                                                               links_df,
                                                               tags_df,
                                                               output_data_config,
                                                               train_split=config['train_split'],
                                                               debug=debug)

    val_ratings_df, val_consolidated_df, test_ratings_df, test_consolidated_df = \
        create_val_and_test_set(train_ratings_df,
                                ratings_df,
                                movies_df,
                                links_df,
                                tags_df,
                                output_data_config,
                                val_split=config['val_split'],
                                debug=debug)

    # verify the data splits meet the necessary conditions
    verify_data_splits(train_ratings_df, val_ratings_df,
                       test_ratings_df, ratings_df)

    #Evaluate sizes of training, validation, and test sets
    train_percent = round(len(train_ratings_df) / len(ratings_df) * 100, 2)
    val_percent = round(len(val_ratings_df) / len(ratings_df) * 100, 2)
    test_percent = round(len(test_ratings_df) / len(ratings_df) * 100, 2)

    print(f"Train: {train_percent}%")
    print(f"Validation: {val_percent}%")
    print(f"Test: {test_percent}%")

    # print first 2 rows of df to stdout
    display(train_ratings_df.head(2))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for Creating Data Splits")
    parser.add_argument('--config',
                        help='''path to config file''',
                        default="../config/data_utils_config.yaml",
                        required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            raise(exc)

    main(config)
