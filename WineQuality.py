import pandas as pd

data = pd.read_csv("winequalityN.csv")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# print(data.isnull().sum())
data = data.fillna(data.median())
# print(data.info())

data['quality'] = pd.cut(data['quality'],2, labels = ['1', '2'])

x = data.drop(["type", "citric acid", "alcohol", "pH", "density", "quality"], axis=1)
y = data["quality"]

# Feature Selection
best_features = SelectKBest(score_func = chi2, k = "all")
fit = best_features.fit(x, y)
data_scores = pd.DataFrame(fit.scores_)
data_columns = pd.DataFrame(x.columns)
features_scores = pd.concat([data_columns, data_scores], axis = 1)
features_scores.columns = ["Attributes", "Score"]
# print(features_scores)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.3)

dt=DecisionTreeClassifier(random_state=0)

dt.fit(x_train, y_train)
y_dt = dt.predict(x_test)

print("Score:", accuracy_score(y_test, y_dt))

#Score: 0.821025641025641



