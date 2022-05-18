from os import makedirs
from os.path import dirname, join, isdir
import argparse
import pickle
import json
import time

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.utils import class_weight

import biol_metrics       # custom functions model evaluation


# options to display all rows and columns for large DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


PATH = dirname(__file__)
DEFAULT_DATA_PATH = join(PATH, '../io/data/')
DEFAULT_SAVE_PATH = join(PATH, '../io/')
DEFAULT_MODEL_PATH = join(DEFAULT_SAVE_PATH, 'models/')

parser = argparse.ArgumentParser(description='Script to train XGBoost classifiers using grid search')

parser.add_argument('--dataset', type=str, default='debug', help='Name of the dataset to train on')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data folder')
parser.add_argument('--model_name', type=str, default='debug', help='Model name')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path where to save the model and its checkpoints')
parser.add_argument('--mode', type=str, default='default', help='Which type of extracted features to use: resize or default')
parser.add_argument('--no_pca', action='store_true', dest='no_pca', default=False, help='Option to use extracted features with no PCA')

parser.add_argument('--parameters', type=str, default='', help='Path to a json file containing parameters for the grid search')
parser.add_argument('--non_biol_classes', type=str, default='', help='Labels of non biological classes separated by commas for evaluation')

args = parser.parse_args()

dataset_name = args.dataset
data_path = args.data_path
model_name = args.model_name
model_path = join(args.model_path, model_name)
# get the correct files depending on the given options
name_suffix = '' if args.mode == 'default' else '_{}'.format(args.mode)
if args.no_pca:
    name_suffix += '_no_pca'
non_biol_classes = [] if args.non_biol_classes == '' else args.non_biol_classes.split(',')

print('Set options')

tree_method = "gpu_hist"
random_state = 3
n_jobs = 10

print('Read training labels and features')

train_csv_path = join(join(data_path, dataset_name), 'train_labels.csv')
df_train = pd.read_csv(train_csv_path)
train_labels    = df_train.label
hand_feat = df_train.drop(['img_path', 'label'], axis=1)

train_deep_feat_path = join(model_path, 'train_deep_features{}.csv'.format(name_suffix))
deep_feat = pd.read_csv(train_deep_feat_path)

# class weights inversely proportional to class count
classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=train_labels)

# combine handcrafted and deep features
train_features = hand_feat.join(deep_feat)

print('Read test labels and features')

test_csv_path = join(join(data_path, dataset_name), 'test_labels.csv')
df_test = pd.read_csv(test_csv_path)
test_labels    = df_test.label
hand_feat = df_test.drop(['img_path', 'label'], axis=1)

test_deep_feat_path = join(model_path, 'test_deep_features{}.csv'.format(name_suffix))
deep_feat = pd.read_csv(test_deep_feat_path)

test_features = hand_feat.join(deep_feat)

if args.parameters == '':
    print('No parameters given.')
    exit()
    
else:
    with open(args.parameters) as params_file:
        parameters = json.load(params_file)
    parameter_grid = ParameterGrid(parameters)
    
    for params in parameter_grid:
        
        print('Define and train classifier for parameters:')
        print(params)
        since = time.time()
        
        xgb = XGBClassifier(n_jobs=n_jobs, tree_method=tree_method, random_state=random_state, **params)
        
        xgb.fit(X=train_features, y=train_labels, sample_weight=classes_weights)
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

        print('Apply classifier')
        since = time.time()
        
        probs = xgb.predict_proba(test_features)

        # get the list of classes, defined at the time the model is fitted
        classes = xgb.classes_

        # extract highest score and corresponding label
        predicted_scores = np.max(probs, axis=1)
        predicted_labels = np.array(classes)[np.argmax(probs, axis=1)]
        
        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

        print('Save results')
        
        eval_path = join(model_path, 'evaluation_results_xgb{}/'.format(name_suffix))
        makedirs(eval_path, exist_ok=True)
        # make a name prefix to recognize results for given parameters
        name_prefix = ''
        for param in params:
            name_prefix += '{}{}_'.format(param, params[param])
        
        df_test['predicted_label'] = predicted_labels
        for i, label in enumerate(classes):
            df_test[label] = probs[:,i]
        df_test.to_csv(join(eval_path, '{}_predictions.csv'.format(name_prefix)))
        
        # compute a few scores
        cr = biol_metrics.classification_report(y_true=df_test.label, y_pred=predicted_labels, y_prob=probs,
                                                non_biol_classes = non_biol_classes)
        print('\n{}\n'.format(cr))
        cr.to_csv(join(eval_path, '{}_classification_report.csv'.format(name_prefix)))
