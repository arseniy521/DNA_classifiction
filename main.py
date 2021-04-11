import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection

# creating a data frame
df = pd.read_csv("dna.csv")
classes = df.loc[:, 'class']

print(df)


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

    KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(max_depth=5),
]

models = zip(names, classifiers)

# evaluate each models in turn

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
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
