

titanic_model = pickle.load(open('../models/titanic_model.pickle', 'rb'))

#Predict using the model
#predigo el target y para los valores seteados, selecciono cualquiera para ver
input_data = (3,0,35,0,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = titanic_model.predict(input_data_reshaped)
#print(prediction)
if prediction[0]==0:
    print("Dead")
if prediction[0]==1:
    print("Alive")