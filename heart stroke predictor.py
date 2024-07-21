#!/usr/bin/env python
# coding: utf-8

# # HEART STROKE PREDICTOR

# SUMMARY --
# 
# *   PERFORMED EDA AND VISUALIZATION ON DATA 
# *   BALANCING DATASET USING OVERSAMPLING TECHNIQUE SMOTE
# *   PERFORMED DIMENSIONALITY REDUCTION(PCA) TO REDUCE DIMENSIONS(ELBOW METHOD) AS WELL AS VIEWING DATA BY REDUCING DIMENSIONS
# *   FITTING MODELS ON OVERSAMPLED DATA(LOGISTIC ,SVM RBF,SVM POLY
# KNN,RANDOM FOREST)
# *   SCALING THE DATA FOR NEURAL NETWORKS(3 DIFFERENT NEURAL NETWORKS OF DIFFERENT ARCHITECTURE USED)(EARLY STOPPING USED)
# *   FITTING MODELS ON OVERSAMPLED SCALED DATA(LOGISTIC ,SVM RBF,
# RANDOM FOREST,KNN, GAUSSIAN NAIVE BAYES)
TEST ACCURACY TABLE OF ALL DIFFERENT MODELS 

1) OVERSAMPLED DATA 

  LOGISTIC REGRESSION ---76.8%
  SVM RBF ---------------75.9%
  SVM POLY --------------76.5%
  KNN -------------------90.9%
  RANDOM FOREST ---------98.9%  
  

2) OVERSAMPLED AND SCALED DATA 

  NEURAL NET MODEL1 -----91.8%
  NEURAL NET MODEL2 -----94.0%
  NEURAL NET MODEL3 -----95.5%
  LOGISTIC REGRESSION ---77.3%
  SVM RBF ---------------92.8%
  RANDOM FOREST ---------98.9%
  NAIVE BAYES -----------77.2%
  KNN -------------------96.3%
# ------------------------------------------------------------------------------------------------------------------

# ## IMPORTING LIBRARIES

# In[83]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import keras
from keras import models
from keras import layers,activations
from keras.callbacks import EarlyStopping


# #### IMPORTING DATASET

# In[3]:


from google.colab import files
uploaded = files.upload()


# In[4]:


import io
df = pd.read_csv(io.BytesIO(uploaded['stroke_data.csv']))


# ###  EXPLORATORY DATA ANALYSIS (EDA)

# In[5]:


df   #OUR DATASET IN PANDAS DATAFRAME FORMAT


# In[6]:


df.info()   


# AS WE CAN SEE THERE ARE NO NULL VALUES IN OUR DATASET AND THE DATATYPES OF COLUMNS ARE DIFFERENT.WE HAVE A TOTAL OF 29065 EXAMPLES IN DATASET.

# In[7]:


df.describe()  


# -----------------------------------------------------------------------------------------------------------------

# ### PRE PROCESSING DIFFERENT CATEGORICAL COLUMNS OF OUR DATASET

# In[8]:


df['gender'].unique()


# In[9]:


df['ever_married'].unique()


# In[10]:


df['work_type'].unique()


# In[11]:


df['Residence_type'].unique()


# In[12]:


df['smoking_status'].unique()


# ### LABEL ENCODING OUR CATEGORICAL

# In[13]:


df['gender'] = df['gender'].map({'Male':0,'Female':1})


# In[14]:


df['ever_married'] = df['ever_married'].map({'Yes':1,'No':0})


# In[15]:


df['Residence_type'] = df['Residence_type'].map({'Urban': 0 ,'Rural':1})


# In[16]:


df['smoking_status'] = df['smoking_status'].map({'never smoked':0,'formerly smoked':1,'smokes':2})


# In[17]:


df['work_type'] = df['work_type'].map({'Private':0,'Self-employed':1,'Govt_job':2,'children':3,'Never_worked':4})


# ----------------------------------------------------------------------------------------------------

# ### VIEWING OUR DIFFERENT FEATURES AND THEIR CORRELATIONS

# In[18]:


sns.heatmap(df.corr())   #HEATMAP SHOWING DIFFERENT CORRELATIONS BETWEEN COLUMNS


# In[19]:


sns.pairplot(df)  #DIFFERENT JOINTPLOTS BETWEEN OUR FEATURES


# In[20]:


sns.distplot(df['age'])  #DISTRIBUTION OF AGE


# In[21]:


sns.distplot(df['avg_glucose_level'])   #DISTRIBUTION OF AVERAGE GLUCOSE LEVEL


# In[22]:


sns.distplot(df['bmi'])       #DISTRIBUTION OF BMI


# In[23]:


print(df['stroke'].value_counts())
print("="*50)
sns.countplot(df['stroke'])
plt.grid()


# ## AS WE CAN SEE OUR DATA IS HIGHLY IMBALANCED 

# In[24]:


df.groupby(['work_type','stroke'])['stroke'].count()


# In[25]:


df.groupby(['smoking_status','stroke'])['stroke'].count()


# ### USING OVERSAMPLING METHOD SMOTE TO BALANCE THE DATASET

# In[26]:


sm = SMOTE()


# In[27]:


sm = SMOTE()
x = df.drop('stroke',axis=1)
y = df['stroke']


# In[28]:


x_oversampled,y_oversampled = sm.fit_sample(x,y)


# In[29]:


x_oversampled.shape


# In[30]:


y_oversampled.shape


# In[31]:


sum(y_oversampled == 0)


# In[32]:


sum(y_oversampled == 1)


# ##### AS WE CAN SEE ABOVE WE NOW HAVE EQUAL NUMBER OF SAMPLES FOR BOTH CLASSES OF OUR DATASET

# ### PERFORMED PCA TO VISUALIZE DATA AS WELL AS ELBOW METHOD TO SEE IF WE CAN REDUCE DIMENSIONS OF OUR DATA

# In[33]:


x_over = x_oversampled


# In[38]:


sc = StandardScaler()
x_over_scaled = sc.fit_transform(x_over)  #SCALED FOR PCA


# In[39]:


pca = PCA()                       # ELBOW METHOD FOR DETERMINING THE RIGHT NUMBER OF DIMENSIONS TO REDUCE THE DATA TO.
pca.fit_transform(x_over_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid()        # WE CAN SEE BELOW THAT FOR PRESERVING MORE THAT 90 PERCENT VARIANCE WE NEED AT LEAST 7 DIMENSIONS THUS NOT REDUCING ANY DIMENSION HERE


# In[40]:


pca_new =  PCA(n_components=2)   # TO VISUALIZE OUR DATA REDUCING DIMENSIONS TO 2
x_pca = pca_new.fit_transform(x_over_scaled)


# In[41]:


plt.scatter(x_pca[:,0],x_pca[:,1],c=y_oversampled)   #DATA PLOT WITH CLASS


# ----------------------

# ### SPLITTING OUR DATASET TO TRAIN , TEST

# In[42]:


x_train,x_test,y_train,y_test = train_test_split(x_oversampled,y_oversampled,test_size=0.33)


# **## APPLYING DIFFERENT MODELS ON OVERSAMPLED DATA(NOT SCALED)**
# 

# In[43]:


#LOGISTIC REGRESSION

lg_oversampled = LogisticRegression(max_iter = 500)
lg_oversampled.fit(x_train,y_train)


# In[93]:


y_pred_lg_oversampled = lg_oversampled.predict(x_test)


# In[97]:


confusion_matrix(y_test,y_pred_lg_oversampled)


# In[99]:


accuracy_score(y_test,y_pred_lg_oversampled)*100


# ###### we get 76.8% accuracy in logistic regression.

# ------------------

# In[101]:


## SUPPORT VECTOR MACHINES(RBF KERNEL)

svc = SVC()
svc.fit(x_train,y_train)


# In[103]:


y_pred_svc_oversampled = svc.predict(x_test)


# In[104]:


confusion_matrix(y_test,y_pred_svc_oversampled)


# In[105]:


accuracy_score(y_test,y_pred_svc_oversampled)*100


# ##### we get 75.9% accuracy in svc with rbf kernel.

# -------------

# In[106]:


## SUPPORT VECTOR MACHINES(POLYNOMIAL KERNEL)

svc_poly = SVC(kernel = 'poly')
svc_poly.fit(x_train,y_train)


# In[107]:


y_pred_svc_poly_oversampled = svc_poly.predict(x_test)


# In[108]:


confusion_matrix(y_test,y_pred_svc_poly_oversampled)


# In[109]:


accuracy_score(y_test,y_pred_svc_poly_oversampled)*100


# ##### we get 76.5% accuracy in svc with polynomial kernel.

# ------------------------------------------------------------------------------------------------------------------

# In[91]:


# KNN CLASSIFIER

knn_oversampled = KNeighborsClassifier()
knn_oversampled.fit(x_train,y_train)


# In[92]:


y_pred_knn_oversampled = knn_oversampled.predict(x_test)


# In[93]:


confusion_matrix(y_test,y_pred_knn_oversampled)


# In[94]:


accuracy_score(y_test,y_pred_knn_oversampled)*100


# ##### we get 90.9% accuracy in knn.

# ------------------------------------------------------------------------------------------------------------------

# In[ ]:


#RANDOM FOREST CLASSIFIER

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[ ]:


y_pred_rfc = rfc.predict(x_test)


# In[ ]:


confusion_matrix(y_test,y_pred_rfc)


# ##### we get 98.9% accuracy on random forest classifier.

# ------------------------------------------------------------------------------------------------------------------

# **## SCALING THE OVERSAMPLED DATA FOR NEURAL NETWORKS AND THEN SPLITING THE DATASET WITH TEST RATIO OF 0.25**

# In[44]:


x_oversampled_scaled = sc.fit_transform(x_oversampled)


# In[45]:


x_train_nn,x_test_nn,y_train_nn,y_test_nn = train_test_split(x_oversampled_scaled,y_oversampled,test_size=0.25)


# ------------------------------------------------------------------------------------------------------------------

# In[48]:


model1 = models.Sequential([
                   layers.Dense(x_train_nn.shape[1],activation='relu',input_dim=x_train_nn.shape[1]),
                   layers.Dense(32,activation ='relu'),
                   layers.Dense(16,activation = 'relu'),
                   layers.Dense(1,activation ='sigmoid')
])


# In[49]:


model1.summary()


# In[50]:


model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(patience=15)


# In[52]:


model1.fit(x_train_nn, y_train_nn, validation_data = (x_test_nn,y_test_nn), callbacks = [early_stop], epochs=100)


# In[53]:


y_pred_nn = model1.predict_classes(x_test_nn)


# In[54]:


confusion_matrix(y_test_nn,y_pred_nn)


# In[55]:


accuracy_score(y_test_nn,y_pred_nn)


# ##### NEURAL NETWORKS MODEL1 GIVES US 91.8% ACCURACY

# ------------------------------------------------------------------------------------------------------------------

# In[57]:


model2 = models.Sequential([
                   layers.Dense(x_train_nn.shape[1],activation='relu',input_dim=x_train_nn.shape[1]),
                   layers.Dense(32,activation ='relu'),
                   layers.Dense(64,activation = 'relu'),
                   layers.Dense(32,activation ='relu'),
                   layers.Dense(8,activation ='relu'),
                   layers.Dense(1,activation ='sigmoid')
])


# In[58]:


model2.summary()


# In[60]:


model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(patience=15)


# In[61]:


model2.fit(x_train_nn, y_train_nn, validation_data = (x_test_nn,y_test_nn), callbacks = [early_stop], epochs=100)


# In[62]:


y_pred_model2 = model2.predict_classes(x_test_nn)


# In[63]:


confusion_matrix(y_test_nn,y_pred_model2)


# In[64]:


accuracy_score(y_test_nn,y_pred_model2)


# ###### NEURAL NETWORKS MODEL2 GIVES US 94.0% ACCURACY

# ------------------------------------------------------------------------------------------------------------------

# In[65]:


model3 = models.Sequential([
                   layers.Dense(x_train_nn.shape[1],activation='relu',input_dim=x_train_nn.shape[1]),
                   layers.Dense(32,activation ='relu'),
                   layers.Dense(64,activation = 'relu'),
                   layers.Dense(128,activation = 'relu'),
                   layers.Dense(64,activation = 'relu'),
                   layers.Dense(32,activation ='relu'),
                   layers.Dense(8,activation ='relu'),
                   layers.Dense(1,activation ='sigmoid')
])


# In[66]:


model3.summary()


# In[68]:


model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop2 = EarlyStopping(patience=10)


# In[69]:


model3.fit(x_train_nn, y_train_nn, validation_data = (x_test_nn,y_test_nn), callbacks = [early_stop2], epochs=100)


# In[70]:


y_pred_model3 = model3.predict_classes(x_test_nn)


# In[71]:


confusion_matrix(y_test_nn,y_pred_model3)


# In[72]:


accuracy_score(y_test_nn,y_pred_model3)


# ###### NEURAL NETWORKS MODEL3 GIVES US 95.5% ACCURACY

# ------------------------------------------------------------------------------------------------------------------

# In[73]:


#LOGISTIC REGRESSION ON OVERSAMPLED SCALED DATA

lg_oversampled_scaled = LogisticRegression(max_iter = 500)
lg_oversampled_scaled.fit(x_train_nn,y_train_nn)


# In[74]:


y_pred_lg_oversampled_scaled = lg_oversampled_scaled.predict(x_test_nn)


# In[76]:


confusion_matrix(y_test_nn,y_pred_lg_oversampled_scaled)


# In[77]:


accuracy_score(y_test_nn,y_pred_lg_oversampled_scaled)


# ##### LOGISTIC REGRESSION AFTER SCALING THE DATA GIVES US 77.3% ACCURACY.

# ------------------------------------------------------------------------------------------------------------------

# In[78]:


#SVC RBF KERNEL ON OVERSAMPLED SCALED DATA

svc_s = SVC()
svc_s.fit(x_train_nn,y_train_nn)
y_pred_svc_s = svc_s.predict(x_test_nn)
confusion_matrix(y_test_nn,y_pred_svc_s)


# In[79]:


accuracy_score(y_test_nn,y_pred_svc_s)


# ##### SVC AFTER SCALING THE DATA GIVES US 92.8% ACCURACY.

# ------------------------------------------------------------------------------------------------------------------

# In[80]:


#RANDOM FOREST CLASSIFIER ON OVERSAMPLED SCALED DATA

rfc_scaled = RandomForestClassifier()
rfc_scaled.fit(x_train_nn,y_train_nn)
y_pred_rfc_scaled = rfc_scaled.predict(x_test_nn)


# In[81]:


confusion_matrix(y_test_nn,y_pred_rfc_scaled)


# In[82]:


accuracy_score(y_test_nn,y_pred_rfc_scaled)


# ##### RANDOM FOREST AFTER SCALING THE DATA GIVES US 98.9% ACCURACY.

# ------------------------------------------------------------------------------------------------------------------

# In[84]:


#NAIVE BAYES CLASSIFIER ON OVERSAMPLED SCALED DATA

nbc = GaussianNB()
nbc.fit(x_train_nn,y_train_nn)
y_pred_nbc = nbc.predict(x_test_nn)


# In[85]:


confusion_matrix(y_test_nn,y_pred_nbc)


# In[87]:


accuracy_score(y_test_nn,y_pred_nbc)


# ##### NAIVE BAYES AFTER SCALING THE DATA GIVES US 77.2% ACCURACY.

# ------------------------------------------------------------------------------------------------------------------

# In[88]:


# KNN CLASSIFIER ON OVERSAMPLED SCALED DATA

knn = KNeighborsClassifier()
knn.fit(x_train_nn,y_train_nn)
y_pred_knn = knn.predict(x_test_nn)


# In[89]:


confusion_matrix(y_test_nn,y_pred_knn)


# In[90]:


accuracy_score(y_test_nn,y_pred_knn)


# ##### KNN CLASSIFIER AFTER SCALING THE DATA GIVES US 96.3% ACCURACY.

# ------------------------------------------------------------------------------------------------------------------
