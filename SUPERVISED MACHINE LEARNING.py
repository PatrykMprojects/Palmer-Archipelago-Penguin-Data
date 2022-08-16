#!/usr/bin/env python
# coding: utf-8

# In[152]:


import warnings 
warnings.filterwarnings('ignore')


# In[189]:


import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[154]:

#https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data

ping_ = pd.read_csv("penguins_lter.csv")
ping_df = pd.DataFrame(data=ping_)
ping_df


# In[155]:


ping_df.drop(columns=ping_df.columns[-1], 
        axis=1, 
        inplace=True)
ping_df
# ping_df.apply(pd.Series.value_counts)


# In[156]:


dummy = pd.get_dummies(ping_df['studyName'])
df = pd.concat((ping_df, dummy), axis=1)
df = df.drop(['studyName'], axis=1)

dummy = pd.get_dummies(df['Sex'])
df = pd.concat((df, dummy), axis=1)

df = df.drop(['Sex', 'FEMALE'], axis=1)
df = df.rename(columns={"MALE":"Sex"})


# In[157]:


df["Species"].value_counts()
# creating a dict file 
species = {'Adelie Penguin (Pygoscelis adeliae)': 1,'Gentoo penguin (Pygoscelis papua)': 2, 'Chinstrap penguin (Pygoscelis antarctica)': 3}
  
# traversing through dataframe
# Gender column and writing
# values where key matches
df.Species = [species[item] for item in df.Species]
df


# In[158]:


df = df.drop(["."], axis=1)


# In[159]:


df["Region"].value_counts()
df = df.drop(["Region"], axis=1)


# In[160]:


df["Island"].value_counts()


# In[161]:


# creating a dict file 
island = {'Biscoe': 1,'Dream': 2, 'Torgersen': 3}
  
# traversing through dataframe
# Gender column and writing
# values where key matches
df.Island = [island[item] for item in df.Island]
df


# In[162]:


# df['Stage'].value_counts()
df = df.drop(['Stage'], axis=1)


# In[163]:


df = df.drop(['Individual ID'], axis=1)


# In[164]:


df['Clutch Completion'].value_counts()

# creating a dict file 
cl = {'Yes': 1,'No': 0}
  
# traversing through dataframe
# Gender column and writing
# values where key matches
df['Clutch Completion'] = [cl[item] for item in df['Clutch Completion']]
df


# In[166]:


df = df.drop(['Date Egg'], axis=1)


# In[181]:


df = df.fillna(0)
df


# In[184]:


df.describe()


# In[185]:


# Get the input features
X_raw = df.drop(['Species'], axis=1)
# Get the target variable
y_raw = df["Species"]


# In[186]:


# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Scaling the raw input features
X = scaler.fit_transform(X_raw)
#feature value rasnge
print(f"The range of feature inputs are within {X.min()} to {X.max()}")


# In[190]:


# Create a LabelEncoder object
label_encoder = LabelEncoder()


# In[192]:


# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())


# In[193]:


np.unique(y, return_counts=True)


# In[213]:


# First, let's split the training and testing dataset
rs = 123
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs, shuffle=True)


# In[214]:


#shapes of the datasets ----> Training Logistic regression 
print(f"Training dataset shape, X_train: {X_train.shape}, y_train: {y_train.shape}")


# In[215]:


print(f"Testing dataset shape, X_test: {X_test.shape}, y_test: {y_test.shape}")


# In[ ]:





# In[216]:


# L2 penalty to shrink coefficients without removing any features from the model
penalty= 'l2'
# Our classification problem is multinomial as we have 3 classes 0,1,2
multi_class = 'multinomial'
# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'
# Max iteration = 1000
max_iter = 1000


# In[217]:


# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)


# In[218]:


l2_model.fit(X_train, y_train)


# In[219]:


l2_preds = l2_model.predict(X_test)


# In[220]:


def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos


# In[221]:


evaluate_metrics(y_test, l2_preds)


# In[222]:


# L1 penalty to shrink coefficients without removing any features from the model
penalty= 'l1'
# Our classification problem is multinomial
multi_class = 'multinomial'
# Use saga for L1 penalty and multinomial classes
solver = 'saga'
# Max iteration = 1000
max_iter = 1000


# In[223]:


# Define a logistic regression model with above arguments
l1_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter = 1000)


# In[224]:


l1_model.fit(X_train, y_train)
l1_model.fit(X_train, y_train)
l1_preds = l1_model.predict(X_test)


# In[225]:


odd_ratios = l1_model.predict_proba(X_test[:1, :])[0]
odd_ratios


# In[226]:


odd_ratios = l1_model.predict_proba(X_test[:1, :])[0]
odd_ratios
l1_model.predict(X_test[:1, :])[0]


# In[227]:


evaluate_metrics(y_test, l1_preds)


# In[228]:


cf = confusion_matrix(y_test, l1_preds)


# In[230]:


cf = confusion_matrix(y_test, l1_preds)
plt.figure(figsize=(16, 12))
ax = sns.heatmap(cf, annot=True, fmt="d", xticklabels=['Adelie Penguin (Pygoscelis adeliae)','Gentoo penguin (Pygoscelis papua)', 'Chinstrap penguin (Pygoscelis antarctica)'], 
                 yticklabels=['Adelie Penguin (Pygoscelis adeliae)','Gentoo penguin (Pygoscelis papua)', 'Chinstrap penguin (Pygoscelis antarctica)'])
ax.set(title="Confusion Matrix");


# In[231]:


l1_model.coef_


# In[232]:


# Extract and sort feature coefficients
def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef
    # Sort coefficients
    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict

# Generate bar colors based on if value is negative or positive
def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals

# Visualize coefficients
def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()  
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()


# In[234]:


feature_cols = list(df.drop(['Species'], axis=1).columns)
feature_cols


# In[235]:


# Get the coefficents for Class 1, Less Often
coef_dict = get_feature_coefs(l1_model, 1, feature_cols)


# In[236]:


visualize_coefs(coef_dict)


# In[237]:


# Coefficients for Class 2
coef_dict = get_feature_coefs(l1_model, 2, feature_cols)
visualize_coefs(coef_dict)


# In[238]:


# Coefficients for Class 0
coef_dict = get_feature_coefs(l1_model, 0, feature_cols)
visualize_coefs(coef_dict)


# In[372]:


#KNN model 
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Evaluation metrics related methods
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[373]:


# Define a random seed to reproduce any random process
rs = 123


# In[374]:


# Ignore any deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[375]:


# lets take the dataset that is already clean and does not contain any object values
df.head()


# In[376]:


#check columns and pull out the Species to be y variable 
df.columns


# In[377]:


X = df.drop(['Species'], axis=1)
y = df['Species']


# In[378]:


#check statistic summary of X
X.describe()


# In[379]:


#Distribution 
y.value_counts(normalize=True)


# In[380]:


y.value_counts().plot.bar(color=['green', 'red', 'blue'])


# In[381]:


from sklearn.preprocessing import StandardScaler

s = StandardScaler()
X_ss = s.fit_transform(X)


# In[382]:


# Split 80% as training dataset
# and 20% as testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs, shuffle=True)
#Split for scaled X
X_sstrain, X_sstest, y_sstrain, y_sstest = train_test_split(X_ss, y, test_size=0.2, stratify=y, random_state = rs, shuffle=True)


# In[383]:


# Define a KNN classifier with `n_neighbors=2`
knn_model = KNeighborsClassifier(n_neighbors=2)


# In[384]:


#we use ravel to convert y dataset to a vector 
knn_model.fit(X_train, y_train.values.ravel())
#scaled model 
knn_model.fit(X_sstrain, y_sstrain.values.ravel())


# In[385]:


preds = knn_model.predict(X_test)
preds_ss = knn_model.predict(X_sstest)


# In[386]:


#function to read predictions 
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average=None)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos


# In[387]:


evaluate_metrics(y_test, preds)


# In[388]:


evaluate_metrics(y_sstest, preds_ss)


# In[389]:


#Finding best number of neighbours for our case 
# Try K from 1 to 50
max_k = 50
# Create an empty list to store f1score for each k
f1_scores = []


# In[399]:


for k in range(1, max_k + 1):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the classifier
    knn = knn.fit(X_train, y_train.values.ravel())
    preds = knn.predict(X_test)
    # Evaluate the classifier with f1score
    f1 = f1_score(preds, y_test, average=None)
    f1_scores.append((k, f1_score(y_test, preds, average=None)))
# Convert the f1score list to a dataframe
f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
f1_results.set_index('K')


# In[ ]:





# In[403]:


lst_f1_avgs = []

for f in f1_results['F1 Score']:
       lst_f1_avgs.append(f.mean())
    


# In[404]:


len(lst_f1_avgs)


# In[407]:




import matplotlib.pyplot as plt

#define data
x_axis = range(0, len(lst_f1_avgs))
y_axis = lst_f1_avgs

#create line plot
plt.plot(x_axis, y_axis)

#show line plot
plt.show()


# In[415]:


#Finding best number of neighbours for our case 
# Try K from 1 to 50
max_ks = 50
# Create an empty list to store f1score for each k
f1_scoress = []


# In[416]:


for k in range(1, max_ks + 1):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the classifier
    knn = knn.fit(X_sstrain, y_sstrain.values.ravel())
    predss = knn.predict(X_sstest)
    # Evaluate the classifier with f1score
    f1 = f1_score(predss, y_sstest, average=None)
    f1_scoress.append((k, f1_score(y_sstest, predss, average=None)))
# Convert the f1score list to a dataframe
f1_resultss = pd.DataFrame(f1_scoress, columns=['K', 'F1 Score'])
f1_resultss.set_index('K')


# In[417]:


lst_f1_avgss = []

for f in f1_resultss['F1 Score']:
       lst_f1_avgss.append(f.mean())
    
import matplotlib.pyplot as plt

#define data
x_axis = range(0, len(lst_f1_avgss))
y_axis = lst_f1_avgss

#create line plot
plt.plot(x_axis, y_axis)

#show line plot
plt.show()


# In[418]:


#Random Forest
import pandas as pd
import pylab as plt
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# In[428]:


df_rf = df
df_rf
# Species is int data type therefore fulfill requirenments 


# In[444]:


def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":metrics.accuracy_score(y_test, model.predict(X_test)),"trian Accuracy": metrics.accuracy_score(y_train, model.predict(X_train))}


# In[445]:



def get_correlation(X_test, y_test,models):
    #This function calculates the average correlation between predictors  
    n_estimators=len(models.estimators_)
    prediction=np.zeros((y_test.shape[0],n_estimators))
    predictions=pd.DataFrame({'estimator '+str(n+1):[] for n in range(n_estimators)})
    
    for key,model in zip(predictions.keys(),models.estimators_):
        predictions[key]=model.predict(X_test.to_numpy())
    
    corr=predictions.corr()
    print("Average correlation between predictors: ", corr.mean().mean()-1/n_estimators)
    return corr


# In[423]:


from sklearn.utils import resample
# resampling 
df_rf[0:5]
for n in range(5):

    print(resample(df_rf[0:5]))


# In[433]:


X=df_rf[['Sample Number', 'Island',  'Clutch Completion',  'Culmen Length (mm)', 'Body Mass (g)', 'Sex', 'Delta 13 C (o/oo)']]


# In[434]:


M=X.shape[1]
M


# In[435]:


# 3 features random samples from 5 bootstrap samples
m=3

feature_index= range(M)
feature_index

import random
random.sample(feature_index,m)


# In[436]:


# randomly select features from the bootstrap samples
#in randomly selecting a subset of the features for each node to split on
for n in range(5):

    print("sample {}".format(n))
    print(resample(X[0:5]).iloc[:,random.sample(feature_index,m)])


# In[437]:


# X and y 
y = df_rf['Species']
y.head()


# In[438]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1, shuffle=True)
print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)


# In[439]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[440]:


n_estimators=20
Bag= BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth = 4,random_state=2),n_estimators=n_estimators,random_state=0,bootstrap=True)


# In[441]:


Bag.fit(X_train,y_train)


# In[442]:


Bag.predict(X_test).shape


# In[446]:


print(get_accuracy(X_train, X_test, y_train, y_test,  Bag))


# In[447]:


get_correlation(X_test, y_test,Bag).style.background_gradient(cmap='coolwarm')


# In[449]:


from sklearn.ensemble import RandomForestClassifier


# In[450]:


n_estimators=20


# In[451]:


M_features=X.shape[1]


# In[452]:


max_features=round(np.sqrt(M_features))-1
max_features


# In[453]:


y_test


# In[454]:


model = RandomForestClassifier( max_features=max_features,n_estimators=n_estimators, random_state=0)


# In[455]:


model.fit(X_train,y_train)


# In[456]:


print(get_accuracy(X_train, X_test, y_train, y_test, model))


# In[457]:


get_correlation(X_test, y_test,model).style.background_gradient(cmap='coolwarm')


# In[ ]:





# In[ ]:





# In[ ]:




