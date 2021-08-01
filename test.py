#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')


df = pd.read_csv('income_prediction_dataset.csv')
df.drop(['native-country', 'workclass', 'education', 'occupation',
         'race', 'capital-gain', 'capital-loss'], axis=1, inplace=True)
cat_col = list(df.select_dtypes(include='object').columns)
num_col =[el for el in df.columns if el not in cat_col]
y = df['gender']
cat_col.remove('gender')

# Pre-process

df_cat = pd.get_dummies(df[cat_col])
sc = StandardScaler()
X_num = sc.fit_transform(df[num_col])
df_num = pd.DataFrame(X_num, columns=num_col)
X = pd.concat([df_num, df_cat], axis = 1)
y = LabelEncoder().fit_transform(y)
X_train,X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.33, random_state = 42)
rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=True, random_state=None, verbose=0,
                       warm_start=False)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)



st.title("Omar's first app")

st.header('Data exploration')

option = st.sidebar.selectbox(
    'Quelle colonne voulez vous afficher?',
     cat_col)

'Colonne: ', option
f, axs = plt.subplots(figsize = (5,3))
axs = sns.countplot(df[option])
plt.xticks(rotation = 45)
plt.title('Distribution de {}'.format(option))
plt.show()
st.pyplot(f)

option2 = st.sidebar.selectbox(
    'Quelle colonne voulez vous afficher?',
     num_col)

'Colonne: ', option2

f, axs = plt.subplots(figsize = (5,3))
axs = sns.distplot(df[option2])
plt.title('Distribution de {}'.format(option))
plt.show()
st.pyplot(f)

st.header('Modélisation')
st.write(print(confusion_matrix(y_test, y_pred)))
st.write(classification_report(y_test, y_pred))

f,axs = plt.subplots(figsize =(14,10))
axs = sns.barplot(x = rfc.feature_importances_, y = X.columns)
plt.title('Feature importantce')
plt.show()
st.pyplot(f)

#'Starting a long computation...'

#latest_iteration = st.empty()
#bar = st.progress(0)

#for i in range(100):
  # Update the progress bar with each iteration.
 # latest_iteration.text(f'Iteration {i+1}')
#  bar.progress(i + 1)
#  time.sleep(0.1)


