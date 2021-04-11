import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection

# creating a data frame
df = pd.read_csv("dna.csv")
classes = df.loc[:, 'class']



# create X and Y datasets for training
X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])


# define a seed for reproducibility
seed = 1

# split the data into training and testing datasets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)

# define scoring method
scoring = 'accuracy'

# Define models to train
names = ["Nearest Neighbors",
         "Decision Tree", "Random Forest"]

classifiers = [

    KNeighborsClassifier(n_neighbors=14, weights='distance'),
    DecisionTreeClassifier(max_depth=7, criterion='entropy'),
]

models = zip(names, classifiers)

# evaluate each models in turn

results = []
names = []


for name, model in models:
    cv_results = model_selection.cross_val_score(model, X_train, y_train, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{0}: {1} ({2})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)
    print("Validating on test data set")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Validation Accuracy")
    print(round(accuracy_score(y_test, predictions), 2))
    print("Confusion Matrix/n")
    print(classification_report(y_test, predictions))


KNN_parameters = {
     "n_neighbors": range(1, 50),
     "weights": ["uniform", "distance"],
}
KNN_grid = GridSearchCV(KNeighborsRegressor(), KNN_parameters)
KNN_grid.fit(X_train, y_train)
print('kek', KNN_grid.best_params_)

DT_parameters = {"criterion": ['gini', 'entropy'],
                 "max_depth": range(1, 10),
                 # "min_samples_split": range(1, 10),
                 # "min_samples_leaf": range(1, 5),
}
DT_grid = GridSearchCV(DecisionTreeClassifier(), DT_parameters)
DT_grid.fit(X_train, y_train)
print(DT_grid.best_params_)

