''' Gives metrics for LightFM as a function of the dataset sample
Usage:
    $ python3 LightFm_Final.py <train_path> <val_path> <test_path>
'''

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from IPython.display import display_html
import warnings
import scipy
import sys
import time

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.datasets import fetch_movielens
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

import warnings
warnings.filterwarnings('ignore')

def compute_val_metric(train_path, val_path):

    train = pd.read_csv(train_path)
    train = train[['userId', 'movieId', 'rating']]
    val = pd.read_csv(val_path)
    val = val[['userId', 'movieId', 'rating']]

    column_names = ["Fraction", "Count", "Time_Taken", "Precision_at_100"]
    val_score = pd.DataFrame(columns=column_names)

    for frac in np.linspace(0.1,1,10):


        sample = train.sample(frac=frac)
        counts = len(sample)
        dataset = Dataset()
        dataset.fit(sample.userId.unique(), sample.movieId.unique())
        (interactions, weights) = dataset.build_interactions([tuple(i) for i in sample.values])
        
        val = val[val['movieId'].isin(sample['movieId'])]
        val = val[val['userId'].isin(sample['userId'])]
        (val_interactions, val_weights) = dataset.build_interactions([tuple(i) for i in val.values])

        start=time.time()
        
        model = LightFM(loss='warp')
        model.fit(interactions)
        val_precision = precision_at_k(model, val_interactions, k=100).mean()
        
        end=time.time()

        total_time = end-start

        ls = [frac, counts, total_time, val_precision]
        row = pd.Series(ls, index=val_score.columns)

        val_score = val_score.append(row, ignore_index = True)

    print(val_score)
    val_score.to_csv("LightFM_Small_VALSCORE.csv")

def compute_test_metric(train_path, test_path):

    train = pd.read_csv(train_path)
    train = train[['userId', 'movieId', 'rating']]
    test = pd.read_csv(test_path)
    test = test[['userId', 'movieId', 'rating']]

    column_names = ["Fraction", "Count", "Time_Taken", "Precision_at_100"]
    test_score = pd.DataFrame(columns=column_names)


    for frac in np.linspace(0.1,1,10):
        #frac=0.5
        print("FRAC :", frac)
        sample = train.sample(frac=frac)
        counts = len(sample)
        dataset = Dataset()
        dataset.fit(sample.userId.unique(), sample.movieId.unique())
        (interactions, weights) = dataset.build_interactions([tuple(i) for i in sample.values])
        test = test[test['movieId'].isin(sample['movieId'])]
        test = test[test['userId'].isin(sample['userId'])]
        (test_interactions, test_weights) = dataset.build_interactions([tuple(i) for i in test.values])
        start=time.time()
        
        model = LightFM(loss='warp')
        model.fit(interactions)
        test_precision = precision_at_k(model, test_interactions, k=100).mean()
        
        end=time.time()
        total_time = end-start
        
        ls = [frac, counts, total_time, test_precision]
        row = pd.Series(ls, index=test_score.columns)
        #print(np.round(frac,2),counts, np.round(total_time,4), test_precision)
        test_score = test_score.append(row, ignore_index = True)
    print(test_score)
    test_score.to_csv("LightFM_Small_TESTSCORE.csv")

if __name__ == '__main__':
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    print("***** PRINTING VAL RESULTS *****\n")
    compute_val_metric(train_path, val_path)
    print("***** PRINTING TEST RESULTS *****\n")
    compute_test_metric(train_path, test_path)
