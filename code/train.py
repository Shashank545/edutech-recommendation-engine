# Code owner : Shashank Sahoo
# Import standard ML packages

import gzip
import pickle

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the given dataset
print("Loading Dataset.........")
raw_data = pd.read_csv("data/MachineLearingtest-Dataset.csv")
data_cols = list(raw_data.columns)

assert raw_data.empty is False, "Dataframe is empty"
assert isinstance(raw_data, pd.DataFrame), f"df param must be a pandas DataFrame, received {type(df)} instead"
assert isinstance(data_cols, list), f"col_names param must be a list, received {type(col_names)} instead"

print("Loading Dataset.........Completed!!")

# Column types differenciation for feature engineering
category_cols = list(raw_data.select_dtypes(include=['object']).columns)
num_cols = list(raw_data.select_dtypes(include=['int']).columns)



# Exploratory Data Analysis- None done so far as per instruction

# Preprocess dataset - None done so far as per instruction

print("Feature engineeing ....")
# Feature encoding & engineering for categorical columns
encoded_cols = []
mapping_dicts = {}
for feat in category_cols:

    raw_data[feat] = raw_data[feat].astype('category')
    feat_coded =  feat + "_code"
    encoded_cols.append(feat_coded)
    raw_data[feat_coded] = raw_data[feat].cat.codes
    code_map = dict(enumerate(raw_data[feat].cat.categories))
    mapping_dicts[feat] = code_map


# Preselected feature & prediction target
X = raw_data[encoded_cols[:-1] + num_cols]
y = raw_data[encoded_cols[-1]]

print("Feature engineeing ....completed!!")

# Split into train and test set, 80%-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create an ensemble of 4 models
estimators = []
estimators.append(('dtree', DecisionTreeClassifier(max_depth = 2)))
estimators.append(('svm', SVC(kernel = 'linear', C = 1)))
estimators.append(('knc', KNeighborsClassifier(n_neighbors = 7)))
estimators.append(('gnb', GaussianNB()))

# Create the Ensemble Model by voting on outcomes
ensemble = VotingClassifier(estimators)

# Make preprocess Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer()),  # Missing value Imputer
    ('scaler', MinMaxScaler(feature_range=(0, 1))),  # Min Max Scaler
    ('model', ensemble)  # Ensemble Model
])
assert isinstance(pipe, Pipeline), "Pipeline from sklearn not imported properly"
# Train the model
pipe.fit(X_train, y_train)
print("Model training ...completed !!")

# Test Accuracy on evaluation dataset
print("Accuracy reported: %s%%" % str(round(pipe.score(X_test, y_test), 3) * 100))

# Export model
joblib.dump(pipe, gzip.open("model/model_binary.dat.gz", "wb"))

#Export Target Mapping
with open('model/model_target_map.pkl', 'wb') as f:
    pickle.dump(mapping_dicts, f)

print("Exporting trained Model files and artifacts ...completed !!")