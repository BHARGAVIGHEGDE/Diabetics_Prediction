import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import flask 


with st.sidebar:
   selected=option_menu(
      menu_title="Main Menu",
      options=["Home","About"],
   )

   if selected=="Home":
      st.title(f"The project predicts the onset of diabetisin a person based on the relevant medical details.When the doctor enters all the releveant medical data required in the online web application,this data is then passed on to trained model for it to make predictions,wheather the person is diabetic or non-diabetic  ")
   if selected=="About":
      st.title(f"GLUCOSE LEVEL: A fasting blood sugar level of 99 mg/dL or lower is normal, 100 to 125 mg/dL indicates you have prediabetes, and 126 mg/dL or higher indicates you have diabetes.      BLOOD PRESSURE:Blood pressure target is usually below 140/90mmHg for people with diabetes or below 150/90mmHg if you are aged 80 years or above. For some people with kidney disease the target may be below 130/80mmHg.")
  
     
      
      
loaded_model=pickle.load(open('/home/bhargavi/Documents/final/trained_dataset.csv','rb'))

def diabetes_prediction(input_data):
  
  #input_data=()

  input_numpy_data=np.asarray(input_data)

  input_data_reshaped=input_numpy_data.reshape(1,-1)
  print(input_data_reshaped)
  
  prediction=loaded_model.predict(input_data_reshaped)
  print("the res is ",prediction)
  if(prediction==1):
    return "the patient is diabetic"
  else:
    return "the patient is not diabetic"

  
  

def main():
  st.title(' Diabetes predictor')
  Pregnancies=st.text_input("Number of pregnancies")
  Glucose=st.text_input("glucose level")
  BloodPressure=st.text_input("blood pressure value")
  SkinThickness=st.text_input("skinthickness value")
  Insulin=st.text_input("insulin value")
  Bmi=st.text_input("bmi value")
  DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function ')
  Age=st.text_input('age of a person')
   
  diagnosis=''
  if st.button('Diabetes test result'):
       diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi, DiabetesPedigreeFunction,Age])  
       st.success(diagnosis)  





#app = flask.Flask(__name__)

#@app.route('/')
#def index():
#    return flask.render_template('final.html', kills=10, deaths=5)
if __name__== '__main__':
    main()


#app.run()


