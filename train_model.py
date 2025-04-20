import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# load data from the csv file
data = pd.read_csv('data/data.csv')

# my data and the output
X = data[['transaction_price', 'time_of_day', 'foreign_transaction', 'online_purchase']]
y = data['fraud']

# train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier() # classification ai model
model.fit(X_train, y_train)

# save the model
joblib.dump(model, 'model/fraudmodel.joblib')
