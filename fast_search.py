import os
import argparse
import json
import time

import numpy as np
import pandas as pd
import nmslib

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from IPython.display import display


def create_index(itemFactors, space='cosinesimil', space_params=None):
    '''Creates Index using Hierarchical Navigable Small World Graph.

    Args:
        space: string. Distance metric to use.
        M: int. Defines the maximum number of neighbors in the zero
            and above-zero layers. The reasonable range of values for these
            parameters is 5-100. Increasing the values of these parameters
            (to a certain degree) leads to better recall and
            shorter retrieval times (at the expense of longer indexing time)
        ef: int. Increasing the value of ef improves recall at the
            expense of longer retrieval time. The reasonable range of values
            for these parameters is 100-2000
    '''
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space=space, space_params=space_params)
    index.addDataPointBatch(itemFactors)
    index.createIndex({'post': 2}, print_progress=True)

    return index


def query_index(index, userFactors, k=10, num_threads=4, batch=True):
    if not batch:
        # query for the nearest neighbours of the first datapoint
        ids, distances = index.knnQuery(userFactors, k=k)
        neighbours = [ids, distances]
        display(ids, distances)
    else:
        # get all nearest neighbours for all the datapoint
        # using a pool of 4 threads to compute
        neighbours = index.knnQueryBatch(userFactors,
                                         k=k,
                                         num_threads=num_threads)
    return neighbours


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for Baseline Model")

    parser.add_argument('--config',
                        help='''config filename''',
                        default="./config/fast_search_config.json",
                        required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as fh:
        config = json.load(fh)

    spark = SparkSession.builder.appName('fast_search').getOrCreate()

    model_load_path = config["model"]["als_model_loadpath"]
    model = ALSModel.load(model_load_path)

    # extract user and item factors from ALS model
    userFactors = model.userFactors.orderBy('id')
    userFactor_vals = userFactors.select('features').rdd.flatMap(lambda x: x)
    userFactor_vals = userFactor_vals.collect()
    userFactor_vals = np.array(userFactor_vals)

    itemFactors = model.itemFactors.orderBy('id')
    itemFactor_vals = itemFactors.select('features').rdd.flatMap(lambda x: x)
    itemFactor_vals = itemFactor_vals.collect()
    itemFactor_vals = np.array(itemFactor_vals)

    # create index on Item Factors
    space_params = config['index_query_params']["space_params"]
    space = config['index_query_params']["space"]
    index = create_index(itemFactor_vals,
                         space=space,
                         space_params=space_params)

    print("\n\n********************")
    start = time.time()
    index_query_params = config['index_query_params']
    neighbours = query_index(index, userFactor_vals,
                             k=index_query_params["topk_preds"],
                             num_threads=index_query_params["num_threads"],
                             batch=index_query_params["batch_input"])
    end = time.time()
    print(f"TIME FOR ANN QUERY: {end-start}")
    print("********************\n\n")
    # display(neighbours[:2])

    # Read in the GT Input Data
    use_small_data = config['data']['use_small_data']

    if use_small_data:
        data_size_split = "small"
        data_config = config['data']['small']
    else:
        data_size_split = "large"
        data_config = config['data']['large']

    # load the input dataset
    if config['data']['data_split'] == "train":
        data_split = "train_csv"
    elif config['data']['data_split'] == "val":
        data_split = "val_csv"
    elif config['data']['data_split'] == "test":
        data_split = "test_csv"
    else:
        raise ValueError(f"Data split must be either train, val or test, Provided {config['data']['data_split']}")

    data_dir = data_config['data_dir']
    fname = data_config[data_split]
    input_path = os.path.join(data_dir, fname)

    print(f"\n*** READING {data_split} from {input_path}***\n")
    gt_data = pd.read_csv(input_path)

    print("*** Preview of Data ***")
    display(gt_data.head(5))

    labels_df = gt_data.groupby('userId')["movieId"].apply(list).reset_index()
    labels_df = labels_df.sort_values(by='userId')

    print("***** DISPLAYING labels_df ****")
    display(labels_df.head(5))

    # get list of users in GT data
    users = labels_df['userId'].unique().ravel().astype(int)

    # parse the predicted user recommendation
    userRecs, pred_scores = zip(*neighbours)
    userRecs = [x.tolist() for x in userRecs]

    # filter to include preds for which we have gt data
    print("*** BEFORE FILTERING userRecs *** ")
    display(len(userRecs))
    # display(userRecs[:2])

    print("*** AFTER FILTERING userRecs *** ")
    # indexing starts from 0 so substract 1 from each userId
    user_ids_to_index = [idx - 1 for idx in users]
    userRecs = np.array(userRecs, dtype=object)[user_ids_to_index].tolist()
    display(len(userRecs))
    # display(userRecs[:2])

    # extract the movie ids for each user in the GT
    movieIds = labels_df['movieId'].values
    # print("DISPLAYING movieId")
    # display(movieIds[:2])

    preds_and_labels = list(zip(userRecs, movieIds))
    # print("DISPLAYING preds_and_labels")
    # display(preds_and_labels[:2])

    pred_and_labels = spark.sparkContext.parallelize(preds_and_labels)

    val_ranking_metrics = RankingMetrics(pred_and_labels)
    val_map = val_ranking_metrics.meanAveragePrecision
    val_atK = val_ranking_metrics.precisionAt(index_query_params["topk_preds"])
    val_ndcg = val_ranking_metrics.ndcgAt(index_query_params["topk_preds"])

    print(f"{data_split.split('_')[0].upper()} {data_size_split.upper()} DATASET MAP = {val_map}")
    print(f"{data_split.split('_')[0].upper()} {data_size_split.upper()} DATASET AT_K = ", val_atK)
    print(f"{data_split.split('_')[0].upper()} {data_size_split.upper()} DATASET NDCG = ", val_ndcg)


