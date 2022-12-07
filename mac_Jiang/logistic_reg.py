# Logistic Regression Example in Python: Step-by-Step Guide



# Step #1: Import Python Libraries -----------------------------------------------------------------



import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix


# Step #2: Explore and Clean the Data --------------------------------------------------------------
# The dataset we are going to use is a Heart Attack directory from Kaggle. 
# The goal of the project is to predict the binary target, whether the patient has heart disease or not.

# https://www.kaggle.com/imnikhilanand/heart-attack-prediction/data?select=data.csv 
#df = pd.read_csv('datasets_23651_30233_data.csv', na_values='?')
df = pd.read_csv('C:/Jiang/Courses/ML_model_validation/codes/heart.csv', na_values='?')
# First, let’s take a look at the variables by calling the columns of the dataset.

print(df.columns)


# Let’s rename the target variable num to target, and also print out the classes and their counts.

df = df.rename(columns={'output': 'target'})


#df['target'].value_counts(dropna=False)


# Next, let’s take a look at the summary information of the dataset.

df.info()


# To keep the cleaning process simple, we’ll remove:
# the columns with many missing values, which are slope, ca, thal.
# the rows with missing values.


df = df.drop(['slp', 'caa', 'thall'], axis=1)

df = df.dropna().copy()

# Let’s recheck the summary to make sure the dataset is cleaned.

df.info()

# We can also take a quick look at the data itself by printing out the dataset.

df


# Step #3: Transform the Categorical Variables: Creating Dummy Variables ---------------------------------
# Among the five categorical variables, sex, fbs, and exang only have two levels of 0 and 1, 
# so they are already in the dummy variable format. 
# But we still need to convert cp and restecg into dummy variables.

df['cp'].value_counts(dropna=False)

df['restecg'].value_counts(dropna=False)

# We can use the get_dummies function to convert them into dummy variables. 
# The drop_first parameter is set to True so that the unnecessary first level dummy variable is removed.

df = pd.get_dummies(df, columns=['cp', 'restecg'], drop_first=True)

df




# To recap, we can print out the numeric columns and categorical columns as numeric_cols and cat_cols below.

numeric_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
cat_cols = list(set(df.columns) - set(numeric_cols) - {'target'})
cat_cols.sort()

print(numeric_cols)
print(cat_cols)



# Step #4: Split Training and Test Datasets-----------------------------------------

random_seed = 888
df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, stratify=df['target'])


print(df_train.shape)
print(df_test.shape)
print()
print(df_train['target'].value_counts(normalize=True))
print()
print(df_test['target'].value_counts(normalize=True))


# Step #5: Transform the Numerical Variables: Scaling -------------------------------
# performs standardization on the numeric_cols of df to return the new array X_numeric_scaled. 
# transforms cat_cols to a NumPy array X_categorical.
# combines both arrays back to the entire feature array X.
# assigns the target column to y.

scaler = StandardScaler()
scaler.fit(df_train[numeric_cols])

def get_features_and_target_arrays(df, numeric_cols, cat_cols, scaler):
    X_numeric_scaled = scaler.transform(df[numeric_cols])
    X_categorical = df[cat_cols].to_numpy()
    X = np.hstack((X_categorical, X_numeric_scaled))
    y = df['target']
    return X, y

X, y = get_features_and_target_arrays(df_train, numeric_cols, cat_cols, scaler)


# Step #6: Fit the Logistic Regression Model
# We first create an instance clf of the class LogisticRegression. Then we can fit it using the training dataset.

clf = LogisticRegression(penalty='none') # logistic regression with no penalty term in the cost function.

clf.fit(X, y)


# Step #7: Evaluate the Model -----------------------------------------------------------
# Before starting, we need to get the scaled test dataset.


X_test, y_test = get_features_and_target_arrays(df_test, numeric_cols, cat_cols, scaler)

# We can plot the ROC curve.

plot_roc_curve(clf, X_test, y_test)

# We can also plot the precision-recall curve

plot_precision_recall_curve(clf, X_test, y_test)


# To calculate other metrics, we need to get the prediction results from the test dataset:

# predict_proba to get the predicted probability of the logistic regression for each class in the model.
# The first column of the output of predict_proba is P(target = 0), and the second column is P(target = 1). 
# So we are calling for the second column by its index position 1.
# predict the test dataset labels by choosing the class with the highest probability, which means a threshold of 0.5 in this binary example.

test_prob = clf.predict_proba(X_test)[:, 1]
test_pred = clf.predict(X_test)


# Using the below Python code, we can calculate some other evaluation metrics:

# Log loss
# AUC
# Average Precision
# Accuracy
# Precision
# Recall
# F1 score
# Classification report, which contains some of the above plus extra information

print('Log loss = {:.5f}'.format(log_loss(y_test, test_prob)))
print('AUC = {:.5f}'.format(roc_auc_score(y_test, test_prob)))
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, test_prob)))
print('\nUsing 0.5 as threshold:')
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, test_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, test_pred)))
print('Recall = {:.5f}'.format(recall_score(y_test, test_pred)))
print('F1 score = {:.5f}'.format(f1_score(y_test, test_pred)))

print('\nClassification Report')
print(classification_report(y_test, test_pred))


# Also, it’s a good idea to get the metrics for the training set for comparison, which we’ll not show in this tutorial. 
# For example, if the training set gives accuracy that’s much higher than the test dataset, there could be overfitting.

# To show the confusion matrix, we can plot a heatmap, which is also based on a threshold of 0.5 for binary classification.

print('Confusion Matrix')
plot_confusion_matrix(clf, X_test, y_test)


# Step #8: Interpret the Results ------------------------------------------------------------------

coefficients = np.hstack((clf.intercept_, clf.coef_[0]))
pd.DataFrame(data={'variable': ['intercept'] + cat_cols + numeric_cols, 'coefficient': coefficients})


# Since the numerical variables are scaled by StandardScaler, we need to think of them in terms of standard deviations. 
# Let’s first print out the list of numeric variable and its sample standard deviation.

pd.DataFrame(data={'variable': numeric_cols, 'unit': np.sqrt(scaler.var_)})


# For example, holding other variables fixed, there is a 41% increase in the odds of having a heart disease for
# every standard deviation increase in cholesterol (63.470764) since exp(0.345501) = 1.41.




























































































































