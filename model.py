import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv("winequality-red.csv", sep=';')

# Select only 5 features
selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol', 'pH']
X = data[selected_features]
y = data['quality']

model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

with open("svr_model.pkl", "wb") as f:
    pickle.dump(model, f)
