#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import plot_roc_curve, roc_auc_score, precision_recall_curve, make_scorer
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, roc_curve, auc, classification_report, f1_score
from pandas_profiling import ProfileReport
from IPython.display import IFrame
import matplotlib as mpl
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
# from imblearn import FunctionSampler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier
import shap
