import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


# Using double backslashes to escape the backslashes in the file path
df = pd.read_csv("C:\\Users\\kamal\\Downloads\\archive (1)\\diabetes_prediction_dataset.csv")


X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:8],df.iloc[:,-1],test_size=0.3,stratify=df.iloc[:,-1],random_state=42)

trf1= ColumnTransformer([
    ('ohe_gender_smokinghistory',OneHotEncoder(sparse=False,handle_unknown='ignore'),[0,4])
],remainder='passthrough')

trf2 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,15))
])

trf3 = DecisionTreeClassifier()

#Syntax of creating pipeline:
pipe = make_pipeline(trf1,trf2,trf3)

#train
pipe.fit(X_train,y_train)

import pickle
pickle.dump(pipe,open('pipe.pkl1','wb'))

pipe=pickle.load(open('pipe.pkl1','rb'))

#loading the saved model
loaded_model = pickle.load(open('pipe.pkl1','rb'))

s=input("Enter your sex as [male/female]: ")
a=int(input("Enter the age: "))
ht=int(input("Enter the hypertension value; "))
hd= int(input("Enter 1 if they have heart disease, else 0: "))
sh= (input("Enter their smoking habit as [No info,never,former,current]: "))
bmi= float(input("Enter the BMI value:"))
hb= float(input("Enter the HbA1c level:"))
glu=int(input("Enter the blood glucose level:"))



test_input1=np.array([s,a,ht,hd,sh,bmi,hb,glu],dtype=object).reshape(1,8)

prediction=loaded_model.predict(test_input1)
print(prediction)
def prediction_of_diabetics(prediction):
  if (prediction[0]==1):
    print("The Person is Diabetic")
  else:
    print("The Person is Not Diabetic")



prediction_of_diabetics(prediction)