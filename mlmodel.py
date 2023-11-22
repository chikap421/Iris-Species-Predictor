import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
iris = pd.read_csv("/Users/chikamaduabuchi/Downloads/8836201-6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
print(iris.head())


y = iris['variety']
iris.drop(columns='variety', inplace=True)
x = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]


# Training the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
