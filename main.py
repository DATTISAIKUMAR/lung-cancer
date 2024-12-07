from flask import Flask, render_template, request
import sqlite3
import numpy as np
import joblib
import pandas as pd

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("survey lung cancer.csv")
df

df.duplicated().sum()

df.duplicated().sum()

df.describe()

df2 = df.replace({2:"YES",1:'NO'})
df.columns =df.columns.str.strip()

df2.columns = df2.columns.str.strip()

df2.head()

df =df2.replace({"YES":0,'NO':1})
df.head()

df2['AGE_BIN'] = pd.cut(df2['AGE'], bins=[20, 40, 60, 80], labels=['20-40', '40-60', '60-80'])



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df.head()
df.LUNG_CANCER.value_counts()

ones=df[df["LUNG_CANCER"]==0].sample(39)
zeros=df[df["LUNG_CANCER"]==1]
df=pd.concat([ones,zeros],axis=0)

from sklearn.model_selection import train_test_split
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
xscale = scale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


























SVM=joblib.load("xgb")
RFC=joblib.load("RFC")
GBC=joblib.load("GBC")





app = Flask(__name__)

app.config["SECRET_KEY"] = 'ajashjkjm'


d={"yes":1,"no":0}
lung={1:"YES",0:"NO"}
m={"male":1,"female":0}
list_files=["gbccm.png","rfccm.png","svccm.png"]
accu=[0.88,0.87,0.83]



@app.route('/')
def home():
    return render_template('lung_prediction.html')



@app.route('/main_page',methods=['GET', 'POST'])
def main_page():
    if request.method == "POST":
        gender=request.form["Gender"]
        age=float(request.form["age"])
        SMOKING=request.form["name1"]
        YELLOW_FINGERS = request.form["name2"]
        ANXIETY = request.form["name3"]
        PEER_PRESSURE = request.form["name4"]
        CHRONIC_DISEASE = request.form["name5"]
        FATIGUE = request.form["name6"]
        ALLERGY = request.form["name7"]
        WHEEZING = request.form["name8"]
        ALCOHOL_CONSUMING = request.form["name9"]
        COUGHING = request.form["name10"]
        SHORTNESS_OF_BREATH = request.form["name11"]
        SWALLOWING_DIFFICULTY = request.form["name12"]
        CHEST_PAIN = request.form["name13"]

        model=request.form["name"]
        values=[[m[gender],age,d[SMOKING],d[YELLOW_FINGERS],d[ANXIETY],d[PEER_PRESSURE],d[CHRONIC_DISEASE],
        d[FATIGUE],d[ALLERGY],d[WHEEZING],d[ALCOHOL_CONSUMING],d[COUGHING],d[SHORTNESS_OF_BREATH],
                d[SWALLOWING_DIFFICULTY],d[CHEST_PAIN]]]
        print(values)
        global accuracy
        global index
        if model == "SVR":
            prediction = SVM.predict(values)
            accuracy=SVM.score(X_test,y_test)
            index=2


        elif model == "RFC" :
            prediction = RFC.predict(values)
            accuracy = RFC.score(X_test, y_test)
            index=1

        elif model == "GBC" :
            prediction = GBC.predict(values)
            accuracy = GBC.score(X_test, y_test)
            index=0

        print(prediction)

        return render_template("main_page.html",prediction=lung[prediction[0]])


    return render_template("main_page.html")

@app.route('/data')
def data():
    return render_template("data.html",data_set=df2)


@app.route("/accuracy")
def accuracy():
    try:
        return render_template("accuracy.html",r2_score=accu[index])
    except Exception:
        return render_template("accuracy.html",r2_score="we can not find accuracy")

@app.route("/data_analytics")
def Data_Analytics():
    return render_template("da.html")


if __name__ == '__main__':
    app.run(debug=True)