#!/usr/bin/python3
#
# Train a RandomForest classifier
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

from os import makedirs
from os.path import dirname, join, isdir
import argparse
import time

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import biol_metrics       # custom functions model evaluation


# options to display all rows and columns for large DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


PATH = dirname(__file__)
DEFAULT_DATA_PATH = join(PATH, '../io/data/')
DEFAULT_SAVE_PATH = join(PATH, '../io/')
DEFAULT_MODEL_PATH = join(DEFAULT_SAVE_PATH, 'models/')

parser = argparse.ArgumentParser(description='Script to train a RF')

parser.add_argument('--dataset', type=str, default='debug', help='Name of the dataset to train on')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data folder')
parser.add_argument('--model_name', type=str, default='debug', help='Model name')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path where to save the model and its checkpoints')
parser.add_argument('--n_estimators', type=int, default=300, help='Number of trees in the Random Forest')
parser.add_argument('--min_samples_leaf', type=int, default=5, help='Min number of objects per leaf')
parser.add_argument('--non_biol_classes', type=str, default='', help='Labels of non biological classes separated by commas for evaluation')
parser.add_argument('--mode', type=str, default='default', help='Which type of extracted features to use: resize or default')
parser.add_argument('--no_pca', action='store_true', dest='no_pca', default=False, help='Option to use extracted features with no PCA')

args = parser.parse_args()

dataset_name = args.dataset
data_path = args.data_path
model_name = args.model_name
model_path = join(args.model_path, model_name)
non_biol_classes = [] if args.non_biol_classes == '' else args.non_biol_classes.split(',')
# get the correct files depending on the given options
name_suffix = '' if args.mode == 'default' else '_{}'.format(args.mode)
if args.no_pca:
    name_suffix += '_no_pca'

print('Set options')

n_estimators = args.n_estimators          # number of trees in the RF
min_samples_leaf = args.min_samples_leaf  # min number of object per leaf
class_weight = 'balanced'                 # class weights inversely proportional to class count
n_jobs = 10                               # number of parallel threads


print('Read training labels and features')

train_csv_path = join(join(data_path, dataset_name), 'train_labels.csv')
df_train = pd.read_csv(train_csv_path)
labels    = df_train.label
hand_feat = df_train.drop(['img_path', 'label'], axis=1)

train_deep_feat_path = join(model_path, 'train_deep_features{}.csv'.format(name_suffix))
deep_feat = pd.read_csv(train_deep_feat_path)

# combine handcrafted and deep features
features = hand_feat.join(deep_feat)


print('Define and train classifier')
since = time.time()

RF = RandomForestClassifier(n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf,
                            class_weight=class_weight,
                            n_jobs=n_jobs)

RF.fit(X=features, y=df_train.label.values)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

# save it to disk for the next step
#with open(join(model_path, 'rf_classifier.pickle'),'wb') as rf_file:
    #pickle.dump(RF, rf_file)
# NB: In EcoTaxa, this used to be possible but resulted in very large models, which were nearly unusable.
#     So this feature is removed for now and the prediction (step 5) is done right after the fitting of the model.


print('Read unknown images features')

test_csv_path = join(join(data_path, dataset_name), 'test_labels.csv')
df_test = pd.read_csv(test_csv_path)
labels    = df_test.label
hand_feat = df_test.drop(['img_path', 'label'], axis=1)

test_deep_feat_path = join(model_path, 'test_deep_features{}.csv'.format(name_suffix))
deep_feat = pd.read_csv(test_deep_feat_path)

# combine handcrafted and deep features
features = hand_feat.join(deep_feat)


print('Apply classifier')
since = time.time()

probs = RF.predict_proba(features)

# get the list of classes, defined at the time the model is fitted
classes = RF.classes_

# extract highest score and corresponding label
predicted_scores = np.max(probs, axis=1)
predicted_labels = np.array(classes)[np.argmax(probs, axis=1)]

time_elapsed = time.time() - since
print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

# save prediction
eval_path = join(model_path, 'evaluation_results/')
df_test['predicted_label'] = predicted_labels
for i, label in enumerate(classes):
    df_test[label] = probs[:,i]
df_test.to_csv(join(eval_path, 'rf_{}{}_predictions.csv'.format(model_name, name_suffix)))

# compute a few scores
cr = biol_metrics.classification_report(y_true=df_test.label, y_pred=predicted_labels, y_prob=probs,
  non_biol_classes = non_biol_classes)
print(cr)
cr.to_csv(join(eval_path, 'rf_{}{}_classification_report.csv'.format(model_name, name_suffix)))