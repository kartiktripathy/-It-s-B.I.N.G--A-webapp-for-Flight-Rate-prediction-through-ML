'''Flight Rate Prediction'''

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the train dataset
data_train = pd.read_excel('Data_Train.xlsx')
pd.set_option('display.max_columns', None)

#Data Preprocessing for the training data
data_train.dropna(inplace = True)
data_train.isnull().sum()
'''Removing the unnecessary columns'''
data_train.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
'''Encoding the Airline column'''
Airline = data_train[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)
'''Encoding the Date_of_Journey column'''
data_train["Journey_day"] = pd.to_datetime(data_train.Date_of_Journey, format="%d/%m/%Y").dt.day
data_train["Journey_month"] = pd.to_datetime(data_train["Date_of_Journey"], format = "%d/%m/%Y").dt.month
data_train.drop(["Date_of_Journey"], axis = 1, inplace = True)
'''Encoding the Source column'''
Source = data_train[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)
'''Encoding the Destination column'''
Destination = data_train[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first= True)
'''Encoding the Dep_Time column'''
data_train["Dep_hour"] = pd.to_datetime(data_train["Dep_Time"]).dt.hour
data_train["Dep_min"] = pd.to_datetime(data_train["Dep_Time"]).dt.minute
data_train.drop(["Dep_Time"], axis = 1, inplace = True)
'''Encoding the Arrival_Time column'''
data_train["Arrival_hour"] = pd.to_datetime(data_train.Arrival_Time).dt.hour
data_train["Arrival_min"] = pd.to_datetime(data_train.Arrival_Time).dt.minute
data_train.drop(["Arrival_Time"], axis = 1, inplace = True)
'''Encoding the Duration column'''
duration = list(data_train["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
data_train["Duration_hours"] = duration_hours
data_train["Duration_mins"] = duration_mins
data_train.drop(["Duration"], axis = 1, inplace = True)
'''Encoding the Total_Stops column'''
data_train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
'''Concatinating the dummy variables and removing the original columns'''
data_train = pd.concat([data_train, Airline, Source, Destination], axis = 1)
data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

#importing the test data
data_test = pd.read_excel('Test_set.xlsx')
pd.set_option('display.max_columns', None)

#Data Preprocessing for the test data
data_test.dropna(inplace = True)
data_test.isnull().sum()
'''Removing the unnecessary columns'''
data_test.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
'''Encoding the Airline column'''
Airline = data_test[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)
'''Encoding the Date_of_Journey column'''
data_test["Journey_day"] = pd.to_datetime(data_test.Date_of_Journey, format="%d/%m/%Y").dt.day
data_test["Journey_month"] = pd.to_datetime(data_test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
data_test.drop(["Date_of_Journey"], axis = 1, inplace = True)
'''Encoding the Source column'''
Source = data_test[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)
'''Encoding the Destination column'''
Destination = data_test[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first= True)
'''Encoding the Dep_Time column'''
data_test["Dep_hour"] = pd.to_datetime(data_test["Dep_Time"]).dt.hour
data_test["Dep_min"] = pd.to_datetime(data_test["Dep_Time"]).dt.minute
data_test.drop(["Dep_Time"], axis = 1, inplace = True)
'''Encoding the Arrival_Time column'''
data_test["Arrival_hour"] = pd.to_datetime(data_test.Arrival_Time).dt.hour
data_test["Arrival_min"] = pd.to_datetime(data_test.Arrival_Time).dt.minute
data_test.drop(["Arrival_Time"], axis = 1, inplace = True)
'''Encoding the Duration column'''
duration = list(data_test["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
data_test["Duration_hours"] = duration_hours
data_test["Duration_mins"] = duration_mins
data_test.drop(["Duration"], axis = 1, inplace = True)
'''Encoding the Total_Stops column'''
data_test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
'''Concatinating the dummy variables and removing the original columns'''
data_test = pd.concat([data_test, Airline, Source, Destination], axis = 1)
data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

#Feature Selection
X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
y = data_train.iloc[:,1].values

#Splitting the dataset into the training and test dataset
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300 , random_state=0 )
regressor.fit(X,y)

#Predicting the Test Set results
y_pred = regressor.predict(x_test)

#Applying GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = parameters,
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
'''the best accuracy u will get if u use these predicted best parameters'''
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

#Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
print("The R_2 Score is : ",r2_score(y_test, y_pred))


import pickle
pickle.dump(regressor , open('flight_rate_predictor.pkl','wb'))