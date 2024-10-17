
import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import seaborn as sns 
import sklearn 
import imblearn 
import sys 
# Ignore warnings 
import warnings 
warnings.filterwarnings('ignore') 
# Settings 
pd.set_option('display.max_columns', None) 
np.set_printoptions (threshold=sys.maxsize)
np.set_printoptions (precision=3) 
sns.set(style="darkgrid") 
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
print("pandas: {0}".format(pd.__version__)) 
print("numpy : {0}".format(np.__version__)) 
print("matplotlib: {0}".format(matplotlib.__version__))
print("seaborn: {0}".format(sns.__version__))
print("sklearn: {0}".format(sklearn.__version__))
print("imblearn: {0}".format(imblearn.__version__))


# Dataset field names
datacols = ["duration", "protocol_type", "service", "flag", "src_bytes", 
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
            "num_failed_logins", "logged_in", "num_compromised", "root_shell", 
            "su_attempted", "num_root", "num_file_creations", "num_shells", 
            "num_access_files", "num_outbound_cmds", "is_host_login", 
            "is_guest_login", "count", "srv_count", "serror_rate", 
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
            "dst_host_srv_rerror_rate", "attack", "last_flag"]

# Load NSL_KDD train dataset
dfkdd_train = pd.read_csv("kddcup.data_10_percent_corrected", sep=",", names=datacols)  # Ensure the path to your dataset is correct
dfkdd_train = dfkdd_train.iloc[:, :-1]  # Remove the unwanted extra field (if necessary)

# Load NSL_KDD test dataset
dfkdd_test = pd.read_csv("kddcup.data_10_percent_corrected", sep=",", names=datacols)  # Ensure the path to your dataset is correct
dfkdd_test = dfkdd_test.iloc[:, :-1]  # Remove the unwanted extra field (if necessary)

# View train data
print(dfkdd_train.head(3))

# Train set dimension
print('Train set dimension: {} rows, {} columns'.format(dfkdd_train.shape[0], dfkdd_train.shape[1]))

# Define the mapping of attack types to their classes
mapping = {
    'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 
    'saint': 'Probe', 'mscan': 'Probe', 'teardrop': 'DOS', 'pod': 'DoS', 
    'land': 'DOS', 'back': 'DOS', 'neptune': 'DOS', 'smurf': 'DoS', 
    'mailbomb': 'DOS', 'udpstorm': 'DoS', 'apache2': 'DOS', 'processtable': 'DoS', 
    'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R', 
    'xterm': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'httptunnel': 'U2R', 
    'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 
    'warezclient': 'R2L', 'imap': 'R2L', 'spy': 'R2L', 'multihop': 'R2L', 
    'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'snmpgetattack': 'R2L', 
    'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L', 
    'normal': 'Normal'
}

# Apply attack class mappings to the dataset
dfkdd_train['attack_class'] = dfkdd_train['attack'].apply(lambda v: mapping.get(v, 'Unknown'))
dfkdd_test['attack_class'] = dfkdd_test['attack'].apply(lambda v: mapping.get(v, 'Unknown'))

# Drop the 'attack' field from both train and test data
dfkdd_train.drop(['attack'], axis=1, inplace=True)
dfkdd_test.drop(['attack'], axis=1, inplace=True)

# View top 3 rows of train data
print(dfkdd_train.head(3))

# Descriptive statistics dfkdd_train.describe() 
dfkdd_train.describe()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

# Check the value counts for 'num_outbound_cmds'
print(dfkdd_train['num_outbound_cmds'].value_counts())
print(dfkdd_test['num_outbound_cmds'].value_counts())

# Since 'num_outbound_cmds' field has all 0 values, drop it from both datasets
dfkdd_train.drop(['num_outbound_cmds'], axis=1, inplace=True)
dfkdd_test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Attack Class Distribution
attack_class_freq_train = dfkdd_train[['attack_class']].value_counts()
attack_class_freq_test = dfkdd_test[['attack_class']].value_counts()

# Calculate frequency percentages
attack_class_freq_train = attack_class_freq_train.rename_axis('attack_class').reset_index(name='frequency')
attack_class_freq_train['frequency_percent_train'] = round((100 * attack_class_freq_train['frequency'] / attack_class_freq_train['frequency'].sum()), 2)

attack_class_freq_test = attack_class_freq_test.rename_axis('attack_class').reset_index(name='frequency')
attack_class_freq_test['frequency_percent_test'] = round((100 * attack_class_freq_test['frequency'] / attack_class_freq_test['frequency'].sum()), 2)

# Concatenate the frequency data
attack_class_dist = pd.concat([attack_class_freq_train[['attack_class', 'frequency_percent_train']], 
                                attack_class_freq_test[['attack_class', 'frequency_percent_test']]], 
                               axis=1)

# Attack class bar plot
plot = attack_class_dist[['frequency_percent_train', 'frequency_percent_test']].plot(kind="bar")
plot.set_title("Attack Class Distribution", fontsize=20)
plot.grid(color='lightgray', alpha=0.5)

# View the first few rows of the train data
print(dfkdd_train.head())

# Scaling numerical attributes
scaler = StandardScaler()
cols = dfkdd_train.select_dtypes(include=['float64', 'int64']).columns

# Scale train and test sets
sc_train = scaler.fit_transform(dfkdd_train[cols])
sc_test = scaler.transform(dfkdd_test[cols])  # Use transform here to avoid fitting again

# Turn the result back to DataFrame
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)

# Encoding categorical attributes
encoder = LabelEncoder()
cattrain = dfkdd_train.select_dtypes(include=['object']).copy()
cattest = dfkdd_test.select_dtypes(include=['object']).copy()

# Encode categorical attributes
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

# Separate target column from encoded data
enctrain = traincat.drop(['attack_class'], axis=1)
enctest = testcat.drop(['attack_class'], axis=1)

cat_Ytrain = traincat[['attack_class']].copy()
cat_Ytest = testcat[['attack_class']].copy()

# Define columns and extract encoded train set for sampling
X = np.concatenate((sc_train, enctrain.values), axis=1)

# Reshape target column to 1D array
y_train = cat_Ytrain.values.ravel()
y_test = cat_Ytest.values.ravel()

# Apply random over-sampling
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y_train)

# Print original and resampled dataset shape
print('Original dataset shape {}'.format(Counter(y_train)))
print('Resampled dataset shape {}'.format(Counter(y_res)))

# Fit Random Forest Classifier on the training set
rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(X_res, y_res)

# Extract important features
importances = np.round(rfc.feature_importances_, 3)
feature_names = np.concatenate((cols, enctrain.columns))  # Combine scaled and encoded feature names
importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort and plot importances
importances_df = importances_df.sort_values('importance', ascending=False).set_index('feature')
plt.rcParams['figure.figsize'] = (11, 4)
importances_df.plot.bar()
plt.title("Feature Importances", fontsize=20)
plt.ylabel("Importance")
plt.grid(color='lightgray', alpha=0.5)
plt.show()

from sklearn.feature_selection import RFE
import itertools
from collections import defaultdict

# Create the RFE model and select 10 attributes
rfe = RFE(rfc, n_features_to_select=10)
rfe.fit(X_res, y_res)

# Summarize the selection of the attributes
feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), refclasscol)]
selected_features = [v for i, v in feature_map if i]

# Print the selected features
print(selected_features)

# Define columns for the new DataFrame
newcol = list(refclasscol)
newcol.append('attack_class')

# Add a dimension to the target
new_y_res = y_res[:, np.newaxis]

# Create a DataFrame from the sampled data
res_arr = np.concatenate((X_res, new_y_res), axis=1)
res_df = pd.DataFrame(res_arr, columns=newcol)

# Create test DataFrame
reftest = pd.concat([sc_testdf, testcat], axis=1)

# Ensure attack class and protocol_type, flag, service are correctly typed
reftest['attack_class'] = reftest['attack_class'].astype(np.float64)
reftest['protocol_type'] = reftest['protocol_type'].astype(np.float64)
reftest['flag'] = reftest['flag'].astype(np.float64)
reftest['service'] = reftest['service'].astype(np.float64)

# Prepare for class dictionary creation
classdict = defaultdict(list)

# Create two-target classes (normal class and an attack class)
attacklist = [('DOS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]
normalclass = [('Normal', 1.0)]

def create_classdict():
    '''This function subdivides train and test dataset into two-class attack labels'''
    for j, k in normalclass:
        for i, v in attacklist:
            restrain_set = res_df.loc[(res_df['attack_class'] == k) | (res_df['attack_class'] == v)]
            classdict[j + '_' + i].append(restrain_set)  # train labels

            # Test labels
            reftest_set = reftest.loc[(reftest['attack_class'] == k) | (reftest['attack_class'] == v)]
            classdict[j + '_' + i].append(reftest_set)

create_classdict()

# Print class dictionary items
for k, v in classdict.items():
    print(k)

# Prepare training and test data
pretrain = classdict['Normal_DoS'][0]
pretest = classdict['Normal_DoS'][1]

# One-hot encoding for categorical features
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
Xresdf = pretrain
newtest = pretest

# Select features
Xresdfnew = Xresdf[selected_features]

# Drop service column for numerical and categorical separation
Xresdfnum = Xresdfnew.drop(['service'], axis=1)
Xresdfcat = Xresdfnew[['service']].copy()

# Prepare test features
Xtest_features = newtest[selected_features]
Xtestdfnum = Xtest_features.drop(['service'], axis=1)
Xtestcat = Xtest_features[['service']].copy()

# Fit train data
enc.fit(Xresdfcat)

# Transform train data
X_train_1hotenc = enc.transform(Xresdfcat).toarray()

# Transform test data
X_test_1hotenc = enc.transform(Xtestcat).toarray()

# Concatenate numerical and one-hot encoded features for train and test
X_train = np.concatenate((Xresdfnum.values, X_train_1hotenc), axis=1)
X_test = np.concatenate((Xtestdfnum.values, X_test_1hotenc), axis=1)

# Prepare target variable
y_train = Xresdf[['attack_class']].copy()
c, r = y_train.values.shape
y_train = y_train.values.reshape(c,)

# Prepare the test target
Y_test = newtest[['attack_class']].copy()
c, r = Y_test.values.shape
y_test = Y_test.values.reshape(c,)

# Import classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Train KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, y_train)

# Train Logistic Regression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, y_train)

# Train Gaussian Naive Bayes Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, y_train)

# Train Decision Tree Model
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, y_train)

print("Training class distribution:", Counter(y_train))

# If needed, apply random oversampling again
if len(np.unique(y_train)) < 2:
    print("Only one class present in y_train. Applying Random Over Sampler again.")
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y_train)

    # Reassign to y_train
    y_train = y_res
    print("After resampling, new training class distribution:", Counter(y_train))
# Prepare for model evaluation
models = []
models.append(('Naive Bayes Classifier', BNB_Classifier))
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('Logistic Regression', LGR_Classifier))

# Evaluate models
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=10)
    accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(y_train, model.predict(X_train))
    classification = metrics.classification_report(y_train, model.predict(X_train))

    print(f'=== {name} Model Evaluation ===')
    print("Cross Validation Mean Score:", scores.mean())
    print("Model Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix)
    print("Classification Report:\n", classification)
    print()

# Test set evaluation
for name, model in models:
    accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
    classification = metrics.classification_report(y_test, model.predict(X_test))

    print(f'=== {name} Model Test Results ===')
    print("Model Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix)
    print("Classification Report:\n", classification)
    print()
