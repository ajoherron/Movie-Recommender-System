#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Trains ALS model on MovieLens Dataset
Usage:
    $ spark-submit --files=/path/to/config/file als_model.py
    path specific in --files need not be in HDFS. Can be present locally
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import functions as F


#NEWLY IMPORTED STUFF
import numpy as np
import pandas as pd
from pyspark.sql.functions import udf, col
from IPython.display import display

def run_htune(spark, config):
    '''Fits an ALS model on training data
    Args:
        spark : spark session object
        config : dict
            user-defined params to use when defining ALS model
    Returns
        model: trained spark ALS model object
    '''
    use_small_data = config['data']['use_small_data']

    if use_small_data:
        data_size_split = 'small'.upper()
        data_config = config['data']['small']
    else:
        data_size_split = 'large'.upper()
        data_config = config['data']['large']

    # load the train dataset
    train_data_dir = data_config['data_dir']
    train_fname = data_config['train_csv']

    train_path = os.path.join(train_data_dir, train_fname)
    train_data = spark.read.csv(train_path, header=True,
                                schema='''userId INT,
                                          movieId INT,
                                          rating FLOAT,
                                          timestamp INT''')

    print("*** TRAIN DATA***")
    train_data.show(5)

    # load the val dataset
    val_data_dir = data_config['data_dir']
    val_fname = data_config['val_csv']

    val_path = os.path.join(val_data_dir, val_fname)
    val_data = spark.read.csv(val_path, header=True,
                              schema='''userId INT,
                                        movieId INT,
                                        rating FLOAT,
                                        timestamp INT''')

    print("*** VAL DATA***")
    val_data.show(5)

    # parse val input data into format required for metrics eval
    val_labels_df = val_data.groupby('userId').agg(F.collect_list("movieId").alias('movieId_labels'))

    print("***** DISPLAYING val_labels_df Spark df ****")
    val_labels_df.show(5)

    # get the set of users in the val dataset
    val_users = val_labels_df.select('userId').distinct().collect()
    val_users = np.array(val_users).ravel().astype(int)

    # get the set of movieIds in the val dataset
    val_movieIds = val_labels_df.select('movieId_labels').collect()
    val_movieIds = list(map(lambda row: row.movieId_labels, val_movieIds))

    # define the als model
    checkpoint_dir = config['model']["checkpoint_dir"]
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    metrics_list = []
    for rank in np.linspace(350, 450, 3):
        for reg_param in np.logspace(-3, -2, 2):
            print(f"USING RANK: {rank} REG PARAM: {reg_param}")
            config['model']['als']['rank'] = rank
            config['model']['als']['regParam'] = reg_param

            model_config = config['model']['als']
            als = ALS(**model_config)
            model = als.fit(train_data)

            # Generate top 100 movie recommendations for each user on val set
            topk_preds = config['eval']['topk_preds']

            pred_start_time = time.time()
            userRecs = model.recommendForAllUsers(topk_preds).select("userId", "recommendations.movieId")

            # print("***** DISPLAYING userRecs Spark df ****")
            # userRecs.show(5)

            # Evaluate model on the validation users
            val_pred_movieIds = userRecs.filter(userRecs.userId.isin(val_users.tolist())).sort("userId")
            val_pred_movieIds = val_pred_movieIds.select("movieId").collect()

            pred_end_time = time.time()
            print("*" * 50)
            print(f"TIME TAKEN FOR ALS PREDICTION: {pred_end_time - pred_start_time}")
            print("*" * 50)

            # parse the output recommendations for the val set
            val_pred_movieIds = list(map(lambda row: row.movieId, val_pred_movieIds))
            val_pred_movieIds = list(map(lambda arr: np.array(arr).tolist(), val_pred_movieIds)) # convert ints to floats\

            val_pred_and_labels = list(zip(val_pred_movieIds, val_movieIds))
            val_pred_and_labels = spark.sparkContext.parallelize(val_pred_and_labels)

            val_ranking_metrics = RankingMetrics(val_pred_and_labels)
            val_map = val_ranking_metrics.meanAveragePrecision
            val_atK = val_ranking_metrics.precisionAt(100)
            val_ndcg = val_ranking_metrics.ndcgAt(100)

            print("*" * 50)
            print("***** DISPLAYING VAL METRICS **** \n")
            print(f"VAL {data_size_split} DATASET MAP = {val_map}")
            print(f"VAL {data_size_split} DATASET AT_K = ", val_atK)
            print(f"VAL {data_size_split} DATASET NDCG = ", val_ndcg)
            print("*" * 50)

            metrics = {'val_map': val_map,
                       'val_atK': val_atK,
                       'val_ndcg': val_ndcg,
                       'reg_param': reg_param,
                       'rank': rank}

            metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for Baseline Model")

    parser.add_argument('--config',
                        help='''config filename''',
                        default="./config/model_config.json",
                        required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as fh:
        config = json.load(fh)

    # Create the spark session object
    # spark = SparkSession.builder.appName('als_model_small')\
    #                             .master('local')\
    #                             .config('spark.submit.deployMode', 'client')\
    #                             .getOrCreate()

    spark = SparkSession.builder.appName('als_model_small')\
                                .getOrCreate()

    # Call main routine
    metrics_df = run_htune(spark, config)

    display("HTUNE RESULTS")
    display(metrics_df)
