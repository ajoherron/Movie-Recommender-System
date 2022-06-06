#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Fits Baseline model on MovieLens Dataset
Usage:
    $ python baseline_model.py --config=/path/to/config/file
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import argparse
import json

from itertools import product
import numpy as np
import pandas as pd
from IPython.display import display

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

class PopularityModel(object):
    """Predicts an entry of the utility matrix for a given user-movie pair"""
    def __init__(self, ratings_df, model_type='damped',
                 count_threshold=5, beta_item=100):
        self.ratings_df = ratings_df
        self.model_type = model_type
        self.count_threshold = count_threshold
        self.beta_item = beta_item

        if model_type not in ['vanilla', 'drop_threshold', 'damped']:
            raise ValueError(f'Invalid model_type provided {model_type}')

    def vanilla_model(self):
        '''Computes Average Score Across All Available User Ratings'''
        item_df = self.ratings_df.groupby('movieId').agg(prediction=('rating', 'mean')).reset_index()

        return item_df

    def drop_threshold_model(self):
        '''Computes Average Score Across All Users With At Least N Ratings'''
        item_df = self.ratings_df.groupby('movieId').agg(ratings_sum=('rating', 'sum'),
                                                         ratings_count=('rating', 'count')).reset_index()

        # filter out users who do not meet min num of ratings threshold
        filtered_item_df = item_df.loc[item_df['ratings_count'] >= self.count_threshold, :]
        filtered_item_df['prediction'] = filtered_item_df['ratings_sum'] / filtered_item_df['ratings_count']

        return filtered_item_df

    def damped_model(self):
        '''Computes Average Across All Users With Additional Damping Factor Beta'''
        item_df = self.ratings_df.groupby('movieId').agg(ratings_sum=('rating', 'sum'),
                                                         ratings_count=('rating', 'count')).reset_index()

        # compute average with damping factor in denominator
        item_df['prediction'] = item_df['ratings_sum'] / (item_df['ratings_count'] + self.beta_item)

        return item_df

    def predict(self):
        '''Computes the Predicted Utility Matrix'''
        if self.model_type == 'vanilla':
            item_df = self.vanilla_model()
            return item_df

        if self.model_type == 'drop_threshold':
            item_df = self.drop_threshold_model()
            return item_df

        elif self.model_type == 'damped':
            item_df = self.damped_model()
            return item_df

        else:
            raise NotImplementedError


def main(spark, config):
    '''Main Execution for Baseline Model

    Args:
        config : dict
            user-defined params to use when defining baseline model

    Returns

    '''
    use_small_data = config['data']['use_small_data']

    if use_small_data:
        data_size_split = 'small'
        data_config = config['data']['small']
    else:
        data_size_split = 'large'
        data_config = config['data']['large']

    # read train data
    train_data_dir = data_config['data_dir']
    train_fname = data_config['train_csv']

    train_path = os.path.join(train_data_dir, train_fname)
    train_data = pd.read_csv(train_path)

    # read val data
    val_data_dir = data_config['data_dir']
    val_fname = data_config['val_csv']

    val_path = os.path.join(val_data_dir, val_fname)
    val_data = pd.read_csv(val_path)

    val_movieIds = val_data.groupby('userId')['movieId'].apply(list).to_frame().reset_index()
    val_movieIds = val_movieIds['movieId'].to_list()

    # read test data
    test_data_dir = data_config['data_dir']
    test_fname = data_config['test_csv']

    test_path = os.path.join(test_data_dir, test_fname)
    test_data = pd.read_csv(test_path)
    test_movieIds = test_data.groupby('userId')['movieId'].apply(list).to_frame().reset_index()
    test_movieIds = test_movieIds['movieId'].to_list()

    # define baseline model for train set
    baseline_model_config = config['model']['baseline']
    train_pop_model = PopularityModel(train_data, **baseline_model_config)
    train_pop = train_pop_model.predict()

    train_pop = train_pop.sort_values(by='prediction',ascending=False)
    top_100_preds = train_pop['movieId'][:100].to_list()
    pred_movieIds = [top_100_preds] * train_data['userId'].nunique()

    val_predictionAndLabels = spark.sparkContext.parallelize(list(zip(pred_movieIds, val_movieIds)))

    val_metrics = RankingMetrics(val_predictionAndLabels)
    merged_df = val_data.merge(train_pop, left_on='movieId', right_on='movieId')
    merged_df['mse'] = merged_df['rating'] - merged_df['prediction']
    merged_df['mse'] = merged_df['mse'] ** 2
    mse = merged_df['mse'].mean()
    val_rmse = np.sqrt(mse)

    print(f"VAL {data_size_split} Precision at 100:",val_metrics.precisionAt(100))
    print(f"VAL {data_size_split} Mean Average Precision", val_metrics.meanAveragePrecision)
    print(f"VAL {data_size_split} ndcg at 100", val_metrics.ndcgAt(100))
    print(f"VAL {data_size_split} RMSE", val_rmse)

    # TEST METRICS
    test_predictionAndLabels = spark.sparkContext.parallelize(list(zip(pred_movieIds, test_movieIds)))
    test_metrics = RankingMetrics(test_predictionAndLabels)
    merged_df = test_data.merge(train_pop, left_on='movieId', right_on='movieId')
    merged_df['mse'] = merged_df['rating'] - merged_df['prediction']
    merged_df['mse'] = merged_df['mse'] ** 2
    mse = merged_df['mse'].mean()
    test_rmse = np.sqrt(mse)

    print(f"TEST {data_size_split} Precision at 100:",test_metrics.precisionAt(100))
    print(f"TEST {data_size_split} Mean Average Precision", test_metrics.meanAveragePrecision)
    print(f"TEST {data_size_split} ndcg at 100", test_metrics.ndcgAt(100))
    print(f"TEST {data_size_split} RMSE", test_rmse)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for Baseline Model")
    parser.add_argument('--config',
                        help='''path to config file''',
                        default="./config/model_config.json",
                        required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as config_fh:
        config = json.load(config_fh)


    # Create the spark session object
    # spark = SparkSession.builder.appName('baseline')\
    #                             .master('local[2]')\
    #                             .config('spark.submit.deployMode', 'client')\
    #                             .getOrCreate()


    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Call main routine
    main(spark, config)
