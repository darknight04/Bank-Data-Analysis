#SOURCE CODE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading the Data
data = pd.read_csv("/Users/riya/Downloads/bank/bank-full.csv", sep=';', quotechar='"')

#Dataset Statistics
print(data.describe(include='all'))

#Visualization 1
plt.hist(data["age"], bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age in Bank Dataset: Riya Aggarwal")
plt.show()

#Visualization 2
job_counts = data["job"].value_counts()
plt.bar(job_counts.index, job_counts.values)
plt.xlabel("Job")
plt.ylabel("Count")
plt.title("Job Distribution in Bank Dataset: Riya Aggarwal")
plt.xticks(rotation=90)
plt.show()

#Visualization 3
sns.boxplot(x="campaign", y="duration", data=data)
plt.xlabel("Number of Contacts during Campaign")
plt.ylabel("Duration of Last Contact (seconds)")
plt.title("Relationship between Campaign and Duration in Bank Dataset: Riya Aggarwal")
plt.show()

#Visualization 4

job_counts = data['job'].value_counts()
labels = job_counts.index
sizes = job_counts.values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Jobs in Bank Dataset: Riya Aggarwal')
plt.show()


#PART 4
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#Split the dataset
X = data.drop("y", axis=1)
y = data["y"]

#onehot encoding
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Decision Tree Classifier
dt = DecisionTreeClassifier()
dt_scores = cross_val_score(dt, X_train, y_train, cv=5)
print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (dt_scores.mean(), dt_scores.std() * 2))
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)
dt_cm = confusion_matrix(y_test, dt_y_pred)
print("Decision Tree Confusion Matrix:")
print(dt_cm)

#KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("K-NN Accuracy: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_y_pred)
print("K-NN Confusion Matrix:")
print(knn_cm)

#Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_y_pred)
print("Random Forest Confusion Matrix:")
print(rf_cm)

#Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb_scores = cross_val_score(gnb, X_train, y_train, cv=5)
print("Gaussian Naive Bayes Accuracy: %0.2f (+/- %0.2f)" % (gnb_scores.mean(), gnb_scores.std() * 2))
gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)
gnb_cm = confusion_matrix(y_test, gnb_y_pred)
print("Gaussian Naive Bayes Confusion Matrix:")
print(gnb_cm)
