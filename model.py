import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import datetime
print("import done")
data = pd.read_csv('london_energy.csv')
"""X = data.iloc[:, :-1].values
y = data.iloc[: ,2].values"""
print("stage 1 done")

data.head()
data['Date'] = pd.to_datetime(data['Date'])
data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
data['season'] = (data['Date'].dt.month%12 + 3)//3
data['Date'] = data['Date'].map(datetime.datetime.toordinal)

print("stage 2 done")


encoder = LabelEncoder()
encoded_col = encoder.fit_transform(data["House_number"])
data['House_number'] = encoded_col


X = data[['House_number', 'day_of_week', 'month', 'season' ,"Date"]]
y = data['Energy Demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions
'''y_pred = model.predict(X_test)
print(y_pred)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)'''

pickle.dump(model , open("model.pkl" , "wb"))



