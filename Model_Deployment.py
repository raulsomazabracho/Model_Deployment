import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

lr = pickle.load(open('logistic_regression_model.pkl','rb'))



# Le colocamos titulo y subitulo a la pagina
st.title("Habrías sobrevivido al titanic?")
st.subheader("Este modelo predice si hubieras sobrevivido al titanic")

st.write("Aca estudiamos mediante una serie de variables si una persona hubiera sobrevivido al titanic o no")

['Clase','Edad','Cantidad de familiares','Sexo']

#INPUTS DEL USUARIO
name = st.text_input("Nombre del pasajero ")
familia = st.slider("ESPOSA+HIJOS", 1, 10,1)
edad = st.slider("Edad", 1, 100,1)
sexo = st.selectbox("Sexo",options=['Hombre' , 'Mujer'])
#clase = st.slider("Clase", 1, 3,1)
clase = st.selectbox("Clase",options = [ 1,2,3])


#Arreglamos el sexo
sexo = 1 if sexo == 'Male' else 0

#Hacemos la predccion
input_data = [[clase,edad,familia,sexo]]
prediction = lr.predict(input_data)
predict_probability = lr.predict_proba(input_data)


#Mostrar la predicción
if prediction == 1:
	st.subheader('Pasajero {} hubiera sobrevivido'.format(name))
	st.subheader('Pasajero {} hubiera sobrevivido con una probabilidad de {}%'.format(name , round(predict_probability[0][1]*100 , 3)))
else:
	st.subheader('Pasajero {} hubiera fallecido'.format(name))
	st.subheader('Pasajero {} hubiera fallecido con una probabilidad de {}%'.format(name, round(predict_probability[0][0]*100 , 3)))

