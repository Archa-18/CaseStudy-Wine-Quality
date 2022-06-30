import pandas as pd

df = pd.read_csv("winequalityN.csv")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# print(df.isnull().sum())
df = df.fillna(df.median())
# print(df.info())

df['quality'] = pd.cut(df['quality'],2, labels = ['1', '2'])

x = df.drop(["type", "citric acid", "alcohol", "pH", "density", "quality"],
            axis=1)
y = df["quality"]

# Feature Selection
best_features = SelectKBest(score_func = chi2, k = "all")
fit = best_features.fit(x, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)
features_scores = pd.concat([df_columns, df_scores], axis = 1)
features_scores.columns = ["Attributes", "Score"]
# print(features_scores)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42,
                                                    test_size=0.3)
dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(x_train, y_train)
y_dtc = dtc.predict(x_test)

print("Score:", accuracy_score(y_test, y_dtc))


#Score: 0.821025641025641



