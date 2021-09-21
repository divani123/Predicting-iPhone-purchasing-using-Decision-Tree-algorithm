

# Step 1 - Load Data
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])


# Step 3 - Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4 - Fit the classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state=0)
classifier.fit(X_train, y_train)

# Step 5 - Predict
y_pred = classifier.predict(X_test)


# Step 6 - Evaluate the model performance
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score:",recall)


# Generating classification report
from sklearn.metrics import classification_report
print('classification report')
print(classification_report(y_test, y_pred))


# Classifier as an image
from sklearn import tree
plt.figure(figsize=(10,8))
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,8), dpi=1000)

tree.plot_tree(classifier,
               feature_names=X, class_names=['0','1'],
               filled=True);
fig.savefig('decision_tree.png')
plt.show()
print('Generated Image saved in the folder')

