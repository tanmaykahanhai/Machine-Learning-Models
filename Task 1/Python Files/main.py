# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from math import sqrt
import pickle  # Import pickle as an alternative to joblib

# Step 2: Load the dataset
dataset = pd.read_csv('student_data.csv')

# Step 3: Perform data preprocessing
# Encoding categorical variables
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                       'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for column in categorical_columns:
    label_encoder = LabelEncoder()
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Step 4: Split the dataset into features (X) and target labels (y)
X = dataset.drop(columns=['G1', 'G2', 'G3'])  # Features
y = dataset['G3']  # Target variable (G3)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification using Random Forest with cross-validation (StratifiedKFold)
classification_model = RandomForestClassifier()

# Increase the number of splits for StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Use a larger number of splits
cv_scores = cross_val_score(classification_model, X, y, cv=cv, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Hyperparameter tuning (example: adjusting the number of trees)
tuned_classification_model = RandomForestClassifier(n_estimators=100)  # You can tune other hyperparameters as well
tuned_classification_model.fit(X_train, y_train)

# Save the trained model to a file using pickle
model_filename = 'random_forest_model.pkl'  # Choose a filename
with open(model_filename, 'wb') as model_file:
    pickle.dump(tuned_classification_model, model_file)
print(f"Trained model saved as {model_filename}")

y_train_pred = tuned_classification_model.predict(X_train)
y_test_pred = tuned_classification_model.predict(X_test)

# Evaluation for Classification
print("Tuned Classification Model Metrics:")
print("Train Set Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Test Set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print(f"Train Set Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}")
# Use zero_division=0 to avoid UndefinedMetricWarning
print(f"Train Set Precision: {precision_score(y_train, y_train_pred, average='weighted', zero_division=0):.2f}")
print(f"Test Set Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.2f}")
print(f"Train Set Recall: {recall_score(y_train, y_train_pred, average='weighted'):.2f}")
print(f"Test Set Recall: {recall_score(y_test, y_test_pred, average='weighted'):.2f}")

# Unsupervised K-means Clustering
# Determine the optimal number of clusters using the Elbow Method
inertia = []
silhouette_scores = []
for k in range(2, 11):  # You can adjust the range as needed
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the Elbow Method graph to find the optimal number of clusters
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)

# Plot the Silhouette Score graph
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Based on the Elbow Method and Silhouette Score, choose the optimal number of clusters (k)

# Fit K-means clustering with the chosen k
chosen_k = 4  # Replace with the optimal number of clusters
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
kmeans.fit(X)

# Add cluster labels to the original dataset
dataset['Cluster'] = kmeans.labels_

# ...

# Visualize the clustering results using the 'age' column
plt.scatter(X['age'], X['Fjob'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Fjob')
plt.legend()
plt.show()
