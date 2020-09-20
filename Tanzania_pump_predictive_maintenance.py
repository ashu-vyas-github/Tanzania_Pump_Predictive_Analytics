
import os, math, copy, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab

from scipy.sparse import hstack
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, log_loss
from category_encoders import TargetEncoder, WOEEncoder, HashingEncoder, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

# Read the file into pandas dataframe

main_path = '/media/ashutosh/Computer Vision/Predictive_Maintenance/Pump-it-up challenge dataset_Kaggle/data'
#main_path = 'E:\Predictive_Maintenance\Pump-it-up challenge dataset_Kaggle\data'

train_raw_features = pd.read_csv(main_path+"//X_train_raw.csv")
train_labels = pd.read_csv(main_path+"//y_train_raw.csv")
test_raw_features = pd.read_csv(main_path+"//X_test_raw.csv")

# Merge train X and y values as per matched index values
train_merged = train_raw_features.merge(train_labels,how='outer',left_index=True, right_index=True)

# encoding train labels
label_dict = {"functional":0,"functional needs repair":1,"non functional":2}
train_merged["label"] = train_merged["status_group"].map(label_dict)

# drop samples from train and test data
# Cleaning training data by removing inappropriate longitudes and latitudes,
# unknown construction years, and unknown population surrounding the pump
print("Raw training data:",train_raw_features.shape)
train_df = train_merged.query('longitude != 0')
train_df = train_df.query('latitude != -2.000000e-08')
#train_df = train_df.query('population != 0')
#train_df = train_df.query('construction_year != 0')
train_df = train_df.query('subvillage == subvillage')
train_df['longitude_cosine'] = train_df['longitude'].map(lambda longitude:(math.cos(longitude * math.pi / 180.0)))
train_df['longitude_sine'] = train_df['longitude'].map(lambda longitude:(math.sin(longitude * math.pi / 180.0)))
train_df['latitude_cosine'] = train_df['latitude'].map(lambda latitude:(math.cos(latitude * math.pi / 180.0)))
train_df['latitude_sine'] = train_df['latitude'].map(lambda latitude:(math.sin(latitude * math.pi / 180.0)))
train_df["cart_X_coord"] = train_df['longitude_cosine']*train_df['latitude_cosine']
train_df["cart_Y_coord"] = train_df['longitude_sine']*train_df['latitude_cosine']
train_df["cart_Z_coord"] = train_df['latitude_sine']

train_labels = train_df["label"]

# Similar process on testing data
print("Raw testing data:",test_raw_features.shape)
test_df = test_raw_features.query('longitude != 0')
test_df = test_df.query('latitude != -2.000000e-08')
#test_df = test_df.query('population != 0')
#test_df = test_df.query('construction_year != 0')
test_df = test_df.query('subvillage == subvillage')
test_df['longitude_cosine'] = test_df['longitude'].map(lambda longitude:(math.cos(longitude * math.pi / 180.0)))
test_df['longitude_sine'] = test_df['longitude'].map(lambda longitude:(math.sin(longitude * math.pi / 180.0)))
test_df['latitude_cosine'] = test_df['latitude'].map(lambda latitude:(math.cos(latitude * math.pi / 180.0)))
test_df['latitude_sine'] = test_df['latitude'].map(lambda latitude:(math.sin(latitude * math.pi / 180.0)))
test_df["cart_X_coord"] = test_df['longitude_cosine']*test_df['latitude_cosine']
test_df["cart_Y_coord"] = test_df['longitude_sine']*test_df['latitude_cosine']
test_df["cart_Z_coord"] = test_df['latitude_sine']

# drop columns from train and test data
features_keep = ['cart_X_coord', 'cart_Y_coord', 'quantity', 'ward', 'waterpoint_type','management','payment', 'quality_group', 'source', 'construction_year', 'extraction_type_group','quantity_group', 'subvillage', 'population', 'region_code', 'basin', 'lga', 'amount_tsh', 'district_code', 'region', 'funder', 'installer', 'wpt_name', 'public_meeting', 'scheme_management', 'scheme_name', 'permit', 'extraction_type', 'extraction_type_class', 'management_group', 'payment_type', 'water_quality', 'source_type', 'source_class', 'waterpoint_type_group']

features_train_all = list(train_df.columns.values)
features_test_all = list(test_df.columns.values)
features_remove_train = list(set(features_train_all)^set(features_keep))
features_remove_test = list(set(features_test_all)^set(features_keep))

train_df = train_df.drop(features_remove_train,axis=1)
test_df = test_df.drop(features_remove_test,axis=1)

print("\nCleaned training data:",train_df.shape)
print("Cleaned testing data:",test_df.shape)
print("\nOriginal Features used:",len(features_keep)," out of",test_raw_features.shape[1])

#### Split training and validation data for measuring performance of the model ####

X_train, X_valid, y_train, y_valid = train_test_split(train_df, train_labels, test_size = 0.2, stratify=train_labels, random_state=42)

#### Label Encoding ####

water_quality_dict = {'soft':7, 'salty':6, 'unknown':0, 'coloured':3, 'fluoride':1, 'salty abandoned':5, 'milky':4, 'fluoride abandoned':2}
quality_group_dict = {'good':5, 'salty':4, 'unknown':0, 'colored':2, 'fluoride':1, 'milky':3}
X_train["water_quality_le"] = X_train["water_quality"].map(water_quality_dict)
X_train["quality_group_le"] = X_train["quality_group"].map(quality_group_dict)
del X_train["water_quality"]
del X_train["quality_group"]
X_valid["water_quality_le"] = X_valid["water_quality"].map(water_quality_dict)
X_valid["quality_group_le"] = X_valid["quality_group"].map(quality_group_dict)
del X_valid["water_quality"]
del X_valid["quality_group"]
test_df["water_quality_le"] = test_df["water_quality"].map(water_quality_dict)
test_df["quality_group_le"] = test_df["quality_group"].map(quality_group_dict)
del test_df["water_quality"]
del test_df["quality_group"]

stdscl = StandardScaler()
train_df_le_ss = stdscl.fit_transform(X_train[["water_quality_le", "quality_group_le"]])
valid_df_le_ss = stdscl.transform(X_valid[["water_quality_le", "quality_group_le"]])
test_df_le_ss = stdscl.transform(test_df[["water_quality_le", "quality_group_le"]])

#### End ####


########## Feature engineering ##########

#### Hashing Encoding ####

features_hashenc = ['funder', 'installer', 'wpt_name', 'public_meeting', 'scheme_name', 'permit']
hash_enc = HashingEncoder(drop_invariant=True, cols=features_hashenc, max_process=0, max_sample=0, n_components=32)

train_df_he = hash_enc.fit_transform(X_train[features_hashenc])
valid_df_he = hash_enc.transform(X_valid[features_hashenc])
test_df_he = hash_enc.transform(test_df[features_hashenc])

stdscl = StandardScaler()
train_df_he_ss = stdscl.fit_transform(train_df_he)
valid_df_he_ss = stdscl.transform(valid_df_he)
test_df_he_ss = stdscl.transform(test_df_he)

#### End ####

#### Standard Scaling Numerical Features ####

features_numerical = ['cart_X_coord', 'cart_Y_coord']
stdscl = StandardScaler()
train_df_stdscl = stdscl.fit_transform(X_train[features_numerical])
valid_df_stdscl = stdscl.transform(X_valid[features_numerical])
test_df_stdscl = stdscl.transform(test_df[features_numerical])

#### End ####

# # #### One-Hot Encoding ####

# # features_ohenc = ['quantity', 'ward', 'waterpoint_type','management','payment', 'quality_group', 'source', 'construction_year', 'extraction_type_group']
# # #one_hot_enc = OneHotEncoder(categories='auto', drop='first', sparse=True, handle_unknown='error')
# # one_hot_enc = OneHotEncoder(cols=features_ohenc, drop_invariant=False, return_df=True, handle_missing='value', handle_unknown='value', use_cat_names=False)

# # train_df_ohe = one_hot_enc.fit_transform(X_train[features_ohenc])
# # valid_df_ohe = one_hot_enc.transform(X_valid[features_ohenc])
# # test_df_ohe = one_hot_enc.transform(test_df[features_ohenc])

# # #### End ####

#### Target Encoding ####

features_targenc = ['amount_tsh', 'basin', 'subvillage', 'region_code', 'ward', 'extraction_type_group', 'extraction_type_class', 'quantity', 'source_type', 'waterpoint_type',  'population', 'construction_year', 'management', 'payment']#['amount_tsh', 'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga', 'ward', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group', 'population', 'construction_year', 'management', 'management_group', 'payment', 'payment_type']

targ_enc = TargetEncoder(cols=features_targenc, smoothing=1000, min_samples_leaf=50)

train_df_te = targ_enc.fit_transform(X_train[features_targenc],y_train)
valid_df_te = targ_enc.transform(X_valid[features_targenc],y_valid)
test_df_te = targ_enc.transform(test_df[features_targenc])#,train_labels)

stdscl = StandardScaler()
train_df_te_ss = stdscl.fit_transform(train_df_te)
valid_df_te_ss = stdscl.transform(valid_df_te)
test_df_te_ss = stdscl.transform(test_df_te)

#### End ####

#### Joining Encoded Data ####

train_df_all_enc = np.hstack((train_df_stdscl, train_df_he_ss, train_df_te_ss, train_df_le_ss))#, format='csr') #, train_df_ohe
valid_df_all_enc = np.hstack((valid_df_stdscl, valid_df_he_ss, valid_df_te_ss, valid_df_le_ss))#, format='csr')#, valid_df_ohe
test_df_all_enc = np.hstack((test_df_stdscl, test_df_he_ss, test_df_te_ss, test_df_le_ss))#, format='csr')#, test_df_ohe

print("\nTraining Samples:",train_df_all_enc.shape[0]," Engineered Features:",train_df_all_enc.shape[1])
print("Validation Samples:",valid_df_all_enc.shape[0]," Engineered Features:",valid_df_all_enc.shape[1])
print("Testing Samples:",test_df_all_enc.shape[0]," Engineered Features:",test_df_all_enc.shape[1],"\n")

#### End ####


print("Beginning model training.....\n")

# # Logistic Regression
# ml_model = LogisticRegression(C=0.125, penalty='l2', solver='lbfgs', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=42, max_iter=10000, multi_class='auto', n_jobs=-1)
# #ml_model.fit(train_df_all_enc, y_train)
# #y_pred = ml_model.predict(valid_df_all_enc)
# acc_logreg = round(accuracy_score(y_valid,y_pred) * 100, 2)
# print("Logistic Regression Acc:",acc_logreg)


#Random Forest
#rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
ml_model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=20, min_samples_split=45, min_samples_leaf=5, max_features='sqrt', max_leaf_nodes=None, oob_score=False, n_jobs=-1, random_state=42, class_weight='balanced_subsample')
ml_model.fit(train_df_all_enc, y_train) # RFC classifier for 35 features with ss and target encoding
# y_pred = ml_model.predict(valid_df_all_enc)
# acc_rfc = round(accuracy_score(y_valid,y_pred) * 100, 2)
# print("Random Forest Classifier Acc:",acc_rfc)

# Support Vector Classifier
# ml_model_svc = SVC(C=10.0, kernel='linear', gamma='scale', shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)
# ml_model.fit(train_df_all_enc, y_train)
# y_pred = ml_model.predict(valid_df_all_enc)
# acc_svc = round(accuracy_score(y_valid,y_pred) * 100, 2)
# print(acc_svc)

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
rfecv = RFECV(ml_model, step=1, min_features_to_select=1, cv=cv, scoring='neg_log_loss', verbose=0, n_jobs=-1)
X_train_new = rfecv.fit_transform(train_df_all_enc, y_train)
print("Optimal features: %d" % rfecv.n_features_)
X_valid_new = rfecv.transform(valid_df_all_enc)
ml_model.fit(X_train_new, y_train)
y_pred = ml_model.predict(X_valid_new)
acc_model = round(accuracy_score(y_valid,y_pred) * 100, 2)
print("Classifier Acc:",acc_model)
# ml_model.fit(X_train_new, y_train)

title = "Learning Curves RFC"
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
plot_learning_curve(ml_model, title, X_train_new, y_train, axes=None, ylim=(0.6, 1.0), cv=cv, n_jobs=-1)
# plot_learning_curve(ml_model, title, train_df_all_enc, y_train, axes=None, ylim=(0.6, 1.0), cv=cv, n_jobs=-1)
plt.show()

### Confusion Matrix
plot_confusion_matrix(ml_model, X_valid_new, y_valid, labels=None, sample_weight=None, normalize='true', display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None)
# plot_confusion_matrix(ml_model, valid_df_all_enc, y_valid, labels=None, sample_weight=None, normalize='true', display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None)
plt.show()

print("\n\nDone dOnE DoNe DONE done!!!!")
os.system('spd-say "your program has finished, please check the output now"')


