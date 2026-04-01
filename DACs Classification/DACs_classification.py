#Step 1:
# Import libraries
# In this section, you can use a search engine to look for the functions that will help you implement the following steps
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import pickle

#Step 2:
# Load dataset and show basic statistics
data = pd.read_csv("disadvantaged_communities.csv")

# 1. Show dataset size (dimensions)
print("dataset dimensions: ", data.shape)

# 2. Show what column names exist for the 49 attributes in the dataset
print("column names: ")
print()
print(data.columns.tolist())

# 3. Show the distribution of the target class CES 4.0 Percentile Range column
print("distribution of target class CES 4.0 Percentile Range: ")
print()
print(data['CES 4.0 Percentile Range'].value_counts().sort_index())

# 4. Show the percentage distribution of the target class CES 4.0 Percentile Range column
print("percentage distribution of target class CES 4.0 Percentile Range: ")
print()
print(data['CES 4.0 Percentile Range'].value_counts(normalize = True).sort_index() * 100)


# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.
data_cleaned_up = data.copy()  
data_cleaned_up = data_cleaned_up.fillna(data_cleaned_up.mean(numeric_only = True))


# Step 4:   
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers
from category_encoders import OrdinalEncoder
# find object type columns
categorical_columns = data_cleaned_up.select_dtypes(include = 'object').columns.tolist() 
# dont include target
categorical_columns = [col for col in categorical_columns if col != 'CES 4.0 Percentile Range']
# use orginal encoding
encoder = OrdinalEncoder()
data_cleaned_up[categorical_columns] = encoder.fit_transform(data_cleaned_up[categorical_columns])


# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
X = data_cleaned_up.drop(columns = ['CES 4.0 Percentile Range'])  
y = data_cleaned_up['CES 4.0 Percentile Range']  

# Create train and test splits for model development. Use the 90% and 10% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5, stratify = y)

# X_train = [] # Remove this line after implementing train test split
# X_test = [] # Remove this line after implementing train test split


# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)
scaler_standardizer = StandardScaler()
X_train_scaled = scaler_standardizer.fit_transform(X_train)
X_test_scaled = scaler_standardizer.transform(X_test)


# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps
cols = X_train.columns
X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd


# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)
SVM_model = SVC(kernel = 'rbf', C = 10.0, gamma = 0.3)
SVM_model.fit(X_train_scaled, y_train)

# Test the above developed SVC on unseen pulsar dataset samples
y_predictions_SVM = SVM_model.predict(X_test_scaled)

# compute and print accuracy score
accuracy = accuracy_score(y_test, y_predictions_SVM)
print('SVM classification accuracy : {0:0.4f}'.format(accuracy))

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
with open('SVMclassifier.sav', 'wb') as model_file:
    pickle.dump(SVM_model, model_file)


# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix

cm_SVM = confusion_matrix(y_test, y_predictions_SVM)
print("SVM Confusion Matrix:\n", cm_SVM)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm_SVM[0,0]
TN = cm_SVM[1,1]
FP = cm_SVM[0,1]
FN = cm_SVM[1,0]

# Compute Precision and use the following line to print it
precision = precision_score(y_test, y_predictions_SVM, average = 'weighted', zero_division = 0)
print('SVM Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = recall_score(y_test, y_predictions_SVM, average = 'weighted', zero_division = 0)
print('SVM Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
FP_total = cm_SVM.sum(axis = 0) - np.diag(cm_SVM)
FN_total = cm_SVM.sum(axis = 1) - np.diag(cm_SVM)
TP_total = np.diag(cm_SVM)
TN_total = cm_SVM.sum() - (FP_total + FN_total + TP_total)
specificity = np.mean(TN_total / (TN_total + FP_total))
print('SVM Specificity : {0:0.3f}'.format(specificity))


# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)
RF_model = RandomForestClassifier(n_estimators = 10, random_state = 0)
RF_model.fit(X_train, y_train)

# Test the above developed Random Forest model on unseen DACs dataset samples
y_predictions_RF = RF_model.predict(X_test)

# compute and print accuracy score
# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
with open('RFclassifier.sav', 'wb') as model_file:
    pickle.dump(RF_model, model_file)

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm_RF = confusion_matrix(y_test, y_predictions_RF)
print("Random Forest Confusion Matrix:\n", cm_RF)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm_RF[0,0]
TN = cm_RF[1,1]
FP = cm_RF[0,1]
FN = cm_RF[1,0]


# Compute Classification Accuracy and use the following line to print it
classification_accuracy = accuracy_score(y_test, y_predictions_RF)
print('RF classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = precision_score(y_test, y_predictions_RF, average = 'weighted', zero_division = 0)
print('RF Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = recall_score(y_test, y_predictions_RF, average = 'weighted', zero_division = 0)
print('RF Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
FP_total = cm_RF.sum(axis=0) - np.diag(cm_RF)
FN_total = cm_RF.sum(axis=1) - np.diag(cm_RF)
TP_total = np.diag(cm_RF)
TN_total = cm_RF.sum() - (FP_total + FN_total + TP_total)
specificity = np.mean(TN_total / (TN_total + FP_total))
print('RF Specificity : {0:0.3f}'.format(specificity))