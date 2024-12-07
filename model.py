import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("survey lung cancer.csv")
df

df.duplicated().sum()

df.duplicated().sum()

df.describe()

plt.figure(figsize=(8,6))
sns.histplot(df['AGE'],bins=30,kde=True)
plt.title('Distribution of AGE for Lung Cancer')
plt.xlabel('AGE')
plt.ylabel('Frequncy')
plt.show()
count_Gender= df['GENDER'].value_counts()
count_Gender


plt.figure(figsize=(8,6))
sns.countplot(x = 'GENDER',hue='LUNG_CANCER',data = df,palette="Set2")
plt.title('Distribution of Gender over Lung Cancer')
plt.xlabel('GENDER')
plt.ylabel('Count')
plt.show()


df2 = df.replace({2:"YES",1:'NO'})


df.columns =df.columns.str.strip()

df2.columns = df2.columns.str.strip()

df2.head()

df =df2.replace({"YES":0,'NO':1})
df.head()

plt.figure(figsize=(15,6))
sns.countplot(x = 'SMOKING',hue='LUNG_CANCER',data = df2,palette="Set2")
plt.title('Distribution of SMOKING over Lung Cancer')
plt.xlabel('SMOKING')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(15,6))
sns.countplot(x = 'SHORTNESS OF BREATH',hue='LUNG_CANCER',data = df2,palette="husl")
plt.title('Distribution of SHORTNESS OF BREATH over Lung Cancer')
plt.xlabel('SHORTNESS OF BREATH')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(x = 'COUGHING',hue='LUNG_CANCER',data = df2,palette="husl")
plt.title('Distribution of COUGHING over Lung Cancer')
plt.xlabel('COUGHING')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(15,6))
sns.countplot(x = 'CHEST PAIN',hue='LUNG_CANCER',data = df2,palette="husl")
plt.title('Distribution of CHEST PAIN over Lung Cancer')
plt.xlabel('CHEST PAIN')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(x = 'ALCOHOL CONSUMING',hue='LUNG_CANCER',data = df2,palette="pastel")
plt.title('Distribution of ALCOHOL CONSUMING over Lung Cancer')
plt.xlabel('ALCOHOL CONSUMING')
plt.ylabel('Count')
plt.show()

df2['AGE_BIN'] = pd.cut(df2['AGE'], bins=[20, 40, 60, 80], labels=['20-40', '40-60', '60-80'])
plt.figure(figsize=(10,6))
sns.countplot(x='AGE_BIN', hue='LUNG_CANCER', data=df2, palette="Set2")
plt.title('Lung Cancer Distribution by Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['GENDER'] = le.fit_transform(df['GENDER'])
df.head()
df.LUNG_CANCER.value_counts()

count_patient = df["LUNG_CANCER"].value_counts()
count_patient


plt.figure(figsize=(8,6))
sns.barplot(x=count_patient.index,y=count_patient.values,palette='Set2')
plt.title('Distribution of patient or Not')
plt.show()

df.corr()


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


df.LUNG_CANCER.value_counts()


ones=df[df["LUNG_CANCER"]==0].sample(39)
zeros=df[df["LUNG_CANCER"]==1]
df=pd.concat([ones,zeros],axis=0)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
xscale = scale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'{model_name} Accuracy: {accuracy:.2f}')
    print(f'{model_name} Classification Report:\n{class_report}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Cancer', 'Has Cancer'])
    disp.plot(cmap='Purples')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    return accuracy




rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_accuracy= evaluate_model(rf, X_test, y_test, 'Random Forest Classifier')
rf_accuracy


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

gbc_accuracy = evaluate_model(gbc, X_test, y_test, 'Gradient Boosting Classifier')
gbc_accuracy


import xgboost as xgb
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

xgboost_model.fit(X_train, y_train)


xgboost_accuracy = evaluate_model(xgboost_model, X_test, y_test, 'XGBoost Classifier')
print(f'XGBoost Classifier Accuracy: {xgboost_accuracy}')

import joblib


joblib.dump(gbc,"GBC")
joblib.dump(xgboost_model,"xgb")
joblib.dump(rf,"RFC")


print(xgboost_model.predict([list(df.iloc[40,:-1])]),gbc.predict([list(df.iloc[0,:-1])]),rf.predict([list(df.iloc[0,:-1])]))