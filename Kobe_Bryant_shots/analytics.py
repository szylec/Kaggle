
# coding: utf-8

# In[410]:


get_ipython().magic(u'reset')
# create pure python file from notebook
get_ipython().system(u'jupyter nbconvert --to script analytics.ipynb')

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import f_oneway, skew
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split
from os import path
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

get_ipython().magic(u'matplotlib inline')

## HELPER FUNCTIONS
# calculate one-way ANOVA
def CalculateAnova(df, col, target):
    groups = list(df[col].unique())
    anova = []
    for group in groups:
        anova.append(list(df[df[col].apply(lambda x: str(group) in str(x))][target].values))
    return f_oneway(*anova)[1]  # return anova p-value


## CONSTANTS
TARGET = ['shot_made_flag']
TARGET_STR = 'shot_made_flag'

TO_SKIP = [
    TARGET_STR,
   'source',
   'game_event_id',
   'game_id',
   'team_id',
   'shot_id',
   'action_type',
   'team_name',
   'matchup',
   'game_date',
]

PERIOD_MAPPING = {
    1: 'quarter_1',
    2: 'quarter_2',
    3: 'quarter_3',
    4: 'quarter_4',
    5: 'overtime_1',
    6: 'overtime_2_and_3',
    7: 'overtime_2_and_3',
}

FOR_ONE_HOT_ENCODING = [
    'combined_shot_type',
    'shot_type',
    'shot_zone_area',
    'shot_zone_basic',
    'shot_zone_range',
    'opponent',
    'period_mapped',
]

SKEWED = [
    'lat', 
    'playoffs'
]

BASE_MODEL_DIR = '/tmp/Kaggle/Kobe_Bryant_shots/model'


# In[411]:


## LOAD DATA
data = pd.read_csv('data.csv')
data.head()

data['source'] = data[TARGET_STR].isnull().apply(lambda x: 'test' if x else 'train').copy()
train = data[data['source'] == 'train'].copy()
test = data[data['source'] == 'test'].copy()

## TARGET VARIABLE
# nothing to be transformed here
#sns.countplot(train[TARGET_STR])  # target classes are not imbalanced

## CONCLUSTIONS:
# - Remove obvious outliers ???
# - Nothing to be transformed from quantitative to qualitative
# - No NULLs to handle :-)
# - Binning:
# -- 'season' to year
# -- 'game_date' to year and quarter
# - ANOVA (check if we have features which groups do not differentiate for target) -> Nope :-)
# - Focus only on one-hot encoding atm
# - Consider outlier removal 

## PREPROCESSING
def PreprocessFeatures(x):
    x['period_mapped'] = x['period'].map(PERIOD_MAPPING)
    x['season'] = x['season'].apply(lambda x: x[0:4]).astype('Int32')
    x['game_date_year'] = x['game_date'].apply(lambda x: pd.to_datetime(x).year).astype('Int32')
    x['game_date_quarter'] = x['game_date'].apply(lambda x: pd.to_datetime(x).quarter).astype('Int32')
    x.drop('game_date', axis=1, inplace=True)
    x = pd.get_dummies(data=x, columns=FOR_ONE_HOT_ENCODING)
    # feature engineering
    x['seconds_from_period_end'] = 60 * x['minutes_remaining'] + x['seconds_remaining']
    x['last_5_sec_in_period'] = x['seconds_from_period_end'] < 5
    x['seconds_from_period_start'] = 60 * (11-x['minutes_remaining']) + (60 - x['seconds_remaining'])
    x['seconds_from_game_start'] = (
        (x['period'] <= 4).astype(int) * (x['period']-1) * 12 * 60 + 
        (x['period'] > 4).astype(int) * ((x['period'] - 4) * 5 * 60 + 3 * 12 * 60) + 
        x['seconds_from_period_start'])
    x.drop('minutes_remaining', axis=1, inplace=True)
    x.drop('seconds_remaining', axis=1, inplace=True)    
    # handle skewness
    lmbda = 0.15
    for c in SKEWED:
        x.loc[:,c] = boxcox1p(x[c], lmbda)
    # change column names
    chars = ['(', ')', ' ', '.', '+', '-']
    for s in chars:
        x.columns = [c.replace(s, '_') for c in list(x.columns)]

    return x

train = PreprocessFeatures(train)
test = PreprocessFeatures(test)

COLS = [c for c in list(train.columns) if c not in TO_SKIP]
QUANTITATIVE = [c for c in COLS if c in list(train._get_numeric_data())]
QUALITATIVE = [c for c in COLS if c not in QUANTITATIVE]


# In[412]:


## DATA MODELLING - TensorFlow
# randomize data and split for train and validation sets
def PrepareData(train, test, normalization_fn=None):
    train = train.reindex(np.random.permutation(train.index))
    COLS_FOR_X = [c for c in COLS if c not in TARGET]
    X = train[COLS_FOR_X]
    y = train[TARGET_STR]
    X_test = test[COLS_FOR_X + ['shot_id']]  # keep shot_id to prepare submission
    
    if normalization_fn:
        X = normalization_fn(X)
        X_test = normalization_fn(X_test)
        
    # Create examples
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    return X_train, X_valid, y_train, y_valid, X_test

# prepare data
normalization_fn = None
X_train, X_valid, y_train, y_valid, X_test = PrepareData(train, test, normalization_fn=normalization_fn)

############################################
# model parameters
LEARINING_RATE = 0.1
STEPS = 2000
BATCH_SIZE = 100
HIDDEN_UNITS = [6, 5]
############################################

def TrainModel(X_train, X_valid, y_train, y_valid, learning_rate, steps, batch_size, hidden_units):
    # define TensorBoard model dir for models comparison
    model_dir = (path.join(BASE_MODEL_DIR, 
                           'lr_{}_st_{}_btch_{}_hu_{}'
                           .format(learning_rate,
                                   steps,
                                   batch_size,
                                   '_'.join(map(str, hidden_units)))))    
    # features for modelling
    feature_columns = [tf.feature_column.numeric_column(c) for c in list(X_train.columns)]
    # classifier
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, 
                                            hidden_units=hidden_units, 
                                            n_classes=2, 
                                            model_dir=model_dir,
                                            activation_fn=tf.nn.relu,
                                            dropout=0.1,
                                            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    # input functions
    train_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train,
                                                         y=y_train,
                                                         num_epochs=None,
                                                         shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_valid,
                                                         y=y_valid,
                                                         num_epochs=None,
                                                         shuffle=False)
    # Train and validate model
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    return classifier
    
#classifier = TrainModel(X_train, X_valid, y_train, y_valid, LEARINING_RATE, STEPS, BATCH_SIZE, HIDDEN_UNITS)


# In[376]:


## CREATE EXTRACT FOR SUBMISSION
def Predict(estimator, X_test):
    test_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=X_test.drop('shot_id', axis=1).copy(),
      num_epochs=1,
      shuffle=False)

    # make prediction
    predictions = list(estimator.predict(test_input_fn))

    predictions[4]['classes']
    predicted_classes = [''.join(list(p["classes"])) for p in predictions]

    # prepare submission
    submission = pd.DataFrame({
        'shot_id': X_test['shot_id'].copy(),
        'shot_made_flag': predicted_classes})
    return submission

submission = Predict(classifier, X_test)
submission.to_csv(path_or_buf='submissions/kobe_bryant_shots_submission_20180131_01.csv',
                  sep=',',
                  header=True,
                  index=False)


# ## ADDITIONAL ANALYSIS

# In[6]:


# Qualitative vs Target
for_plot = train[QUALITATIVE + TARGET].copy()
for_plot = pd.melt(for_plot, id_vars=TARGET_STR)
g = sns.FacetGrid(data=for_plot, col='variable', col_wrap=3, size=4, sharex=False, sharey=False)

def countplot(x, hue, **kwargs):
    sns.countplot(x=x, hue=hue, **kwargs)

g.map(countplot, 'value', TARGET_STR)


# In[7]:


# calculate ANOVA
anova = []
for_anova = train[QUALITATIVE + TARGET].copy()
for col in QUALITATIVE:
    tmp = (col, CalculateAnova(for_anova, col, TARGET_STR))
    anova.append(tmp)

anova = pd.DataFrame(anova)
anova.columns = ['feature', 'p_value']

# anova[anova['p_value'] > 0.5]  # conlcusion: nothing to be removed
anova.sort_values(by='p_value').head(10)


# In[84]:


# Plot quantitative vs target
for_plot = pd.melt(frame=train[QUANTITATIVE + TARGET], id_vars=TARGET_STR)
g = sns.FacetGrid(data=for_plot, col='variable', col_wrap=3, size=4, sharex=False, sharey=False)
g.map(sns.distplot, 'value')


# In[74]:


# Transform highly skewed features
skewness = []
for c in QUANTITATIVE:
    skewness.append((c, abs(skew(train[c]))))
skewness = pd.DataFrame(skewness)
skewness.columns = ['feature', 'skewness']
skewness.sort_values(by='skewness', ascending=False)
skewness[skewness['skewness'] > 0.75]['feature'].tolist()


# In[423]:


## DATA MODELLING - alternative methods
clf = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=7,
    learning_rate=0.1,
    max_delta_step=1,
    min_child_weight=3,
    colsample_bytree=0.7,
    n_estimators=30
)

clf.fit(X=X_train, y=y_train)
prediction = model.predict(X_valid)

print(classification_report(y_valid, prediction))


# In[ ]:




