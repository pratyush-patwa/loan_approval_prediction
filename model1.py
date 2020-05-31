import pandas as pd
import numpy as np  # For mathematical calculations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs
import warnings  # To ignore any warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")

train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.show()

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24, 6), title='Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16, 5))
plt.show()

train.boxplot(column='ApplicantIncome', by='Education')

train.isnull().sum()

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
sns.distplot(train['Total_Income']);

train['Total_Income_log'] = np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log']);
train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term']
sns.distplot(train['EMI']);
train['Balance Income'] = train['Total_Income'] - (train['EMI'] * 1000)
sns.distplot(train['Balance Income']);
train = train.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)

train['Gender'].replace('Male', 0, inplace=True)
train['Gender'].replace('Female', 1, inplace=True)

train['Married'].replace('No', 0, inplace=True)
train['Married'].replace('Yes', 1, inplace=True)

train['Self_Employed'].replace('No', 0, inplace=True)
train['Self_Employed'].replace('Yes', 1, inplace=True)

train['Education'].replace('Not Graduate', 0, inplace=True)
train['Education'].replace('Graduate', 1, inplace=True)

train['Dependents'].replace('3+', 3, inplace=True)

train['Property_Area'].replace('Rural', 0, inplace=True)
train['Property_Area'].replace('Semiurban', 1, inplace=True)
train['Property_Area'].replace('Urban', 2, inplace=True)

X = train.drop('Loan_Status', 1)
y = train.Loan_Status

from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)
regressor = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                               max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=1,
                               solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
regressor.fit(X, y)


pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

# pred_cv = regressor.predict(x_cv)
#
# accuracy_score(y_cv, pred_cv)
#
# final_features = pd.DataFrame([[0, 0, 1, 1, 1, 1500, 1000, 2500, 120, 0, 2, 5.298317366548036]])
# # prediction = model.predict(final_features)
# pred_cv = regressor.predict(final_features)
