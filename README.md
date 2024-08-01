import sys # Importing the sys module, which provides access to some variables used or maintained by the interpreter.
# Asking the user to input a psychiatric disorder.
disorder = input("Enter a psychiatric disorder: ")
# Checking the user input for "Depression" and providing options for different machine learning methods.
if disorder == "Depression":
choice = input("Select a machine learning method: clustering (c) or random forests (rf): ")
# If the user inputs "c", providing information about the clustering method and
importing the Clus2 module.
if choice == "c":
print("Machine learning method: Clustering")
print("Omics data integrated: Genomics, metabolomics")
print("API: python clustering 1.3.0 (pip install python-clustering)\n")
import Clus2
# If the user inputs "rf", providing information about the random forests method and importing the ran_for module.
elif choice == "rf":
print("Machine learning method: Random forests")
print("Omics data integrated: Genomics, Transcriptomics, Epigenomics")
print("API: scikit-learn 1.2.2 (pip install scikit-learn)\n")
import ran_for
# If the user inputs anything else, printing an error message.
else:
print("Invalid input. Please select 'c' for clustering or 'rf' for random forests.")
# Checking the user input for "Schizophrenia" and providing information about the principal component analysis method.
elif disorder == "Schizophrenia":
print("Machine learning method: Principal component analysis")
print("Omics data integrated: Genomics, metabolomics")
print("API: scikit-learn 1.2.2 (pip install scikit-learn)")
import pca1
# Checking the user input for "Attention deficit hyperactivity disorder" and providing information about the linear models method.
elif disorder == "Attention deficit hyperactivity disorder":
print("Machine learning method: Linear models")
print("Omics data integrated: Genomics, epigenomics")
print("API: scikit-learn 1.2.2 (pip install scikit-learn)")
import lin_model
# Checking the user input for "Dementia" and providing information about the linear/logistic regression method.
elif disorder == "Dementia":
print("Machine learning method: Linear/logistic regression")
print("Omics data integrated: Genomics, transcriptomics, epigenomics, proteomics")
print("API: py4linear-regression 0.0.5 (pip install py4linear-regression)")
import linear_reg1
# If the user inputs anything else, printing an error message.
else:
print("Sorry, we don't have information for that psychiatric disorder.")

Modules:
1. Clus2
# Import necessary libraries
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Prompt user for number of samples, centers of the blobs, cluster standard deviation,
and random state
print("Enter the number of samples: ")
n_samples = int(input())
print("Enter the centers of the blobs (in the format 'x1,y1;x2,y2;x3,y3'): ")
centers_input = input()
centers = [list(map(float, center.split(","))) for center in centers_input.split(";")]
print("Enter the cluster standard deviation: ")
cluster_std = float(input())
print("Enter the random state: ")
random_state = int(input())
# Generate data using the make_blobs function and scale it using StandardScaler
X, labels_true = make_blobs(
n_samples=n_samples, centers=centers, cluster_std=cluster_std,
random_state=random_state
)
X = StandardScaler().fit_transform(X)
# Scatter plot of the generated data
plt.scatter(X[:, 0], X[:, 1])
plt.show()
# Prompt user for eps value and minimum number of samples for a cluster
eps = float(input("Enter eps value: "))
min_samples = int(input("Enter minimum number of samples for a cluster: "))
# Use the DBSCAN algorithm to cluster the data and get the labels
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_
# Get the number of clusters and noise points
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
# Print the estimated number of clusters and noise points
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# Calculate and print the homogeneity, completeness, v-measure, adjusted rand index,
adjusted mutual information, and silhouette coefficient
print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
print("Adjusted Mutual Information:"
f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
)
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
# Create a scatter plot of the clustered data with different colors representing different
clusters
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
if k == -1:
# Black used for noise.
col = [0, 0, 0, 1]
class_member_mask = labels == k
xy = X[class_member_mask & core_samples_mask]
plt.plot(
xy[:, 0],
xy[:, 1],
"o",
markerfacecolor=tuple(col),
markeredgecolor="k",
markersize=14,
)
xy = X[class_member_mask & ~core_samples_mask]
plt.plot(
xy[:, 0],
xy[:, 1],
"o",
markerfacecolor=tuple(col),
markeredgecolor="k",
markersize=6,
)
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show() # display the plot
2. ran_for
# Import required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Ask for user input to generate a random dataset
n_samples = int(input("Enter the number of samples: ")) # number of samples in the dataset
n_features = int(input("Enter the number of features: ")) # number of features in the dataset
n_informative = int(input("Enter the number of informative features: ")) # number of
informative features in the dataset
n_redundant = int(input("Enter the number of redundant features: ")) # number of redundant
features in the dataset
random_state = int(input("Enter the random state: ")) # random state for generating the
dataset
shuffle = False if input("Shuffle the data? (y/n)").lower() == "n" else True # whether to
shuffle the dataset or not
# Generate the dataset using make_classification function from sklearn
X, y = make_classification(n_samples=n_samples, n_features=n_features,
n_informative=n_informative, n_redundant=n_redundant,
random_state=random_state, shuffle=shuffle)
# Create a RandomForestClassifier model and fit it on the generated dataset
max_depth = int(input("Enter the maximum depth for the random forest classifier: ")) #
maximum depth of the random forest
clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
clf.fit(X, y)
# Get user input for new data and predict the class using the trained model
input_data = []
for i in range(n_features):
val = float(input(f"Enter value for feature {i+1}: "))
input_data.append(val)
prediction = clf.predict([input_data])
# Print the predicted class for the input data
print(f"The predicted class label for input data {input_data} is {prediction[0]}.")
3. pca1
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA
from scipy import sparse
# Load the digits dataset from sklearn
X, y = load_digits(return_X_y=True)
# Ask for user input for the number of principal components to extract and batch size
n_components = int(input("Enter the number of principal components to extract: ")) #
number of principal components to extract from the data
batch_size = int(input("Enter the batch size for the transformer: ")) # number of samples to
process at a time
transformer = IncrementalPCA(n_components=n_components, batch_size=batch_size) #
create an IncrementalPCA transformer with specified parameters
X_transformed = transformer.fit_transform(X) # fit the transformer on the data and
transform the data
# Plot the transformed data in a 2D scatter plot
plt.figure(figsize=(8, 6))
for digit in set(y):
# Plot the data points for each digit with a different color and label
plt.scatter(X_transformed[y == digit, 0], X_transformed[y == digit, 1], alpha=0.8,
label=digit)
plt.legend() # show the legend for the different digit labels
plt.title("PCA plot of the transformed data") # add a title to the plot
plt.show() # display the plot
4. lin_model
# Import required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
# Ask for user input for the number of data points and features
n_points = int(input("Enter the number of data points: "))
n_features = int(input("Enter the number of features: "))
# Create an empty array of shape (n_points, n_features) to store the input data
X = np.empty((n_points, n_features))
for i in range(n_points):
print("Enter the features for data point", i+1)
for j in range(n_features):
# Ask for user input for each feature of each data point
X[i, j] = float(input("Feature " + str(j+1) + ": "))
# Create an empty array of shape (n_points,) to store the target labels
Y = np.empty(n_points, dtype=int)
for i in range(n_points):
# Ask for user input for the target label of each data point
Y[i] = int(input("Enter the target label for data point " + str(i+1) + ": "))
# Create a pipeline with a StandardScaler and a SGDClassifier
clf = make_pipeline(StandardScaler(),
SGDClassifier(max_iter=1000, tol=1e-3))
# Fit the pipeline on the input data and target labels
clf.fit(X, Y)
# Ask for user input for the features of the data point to predict
x_pred = np.empty(n_features)
print("Enter the features for the data point to predict:")
for j in range(n_features):
x_pred[j] = float(input("Feature " + str(j+1) + ": "))
# Predict the label of the input data point using the pipeline
prediction = clf.predict([x_pred])
print("The predicted label for the input data point is:", prediction[0])
5. linear_reg1
# Import required libraries
from py4linear_regression.regression import linear_regression
# Prompt the user to enter the training data as a list of 2D points.
print("Enter the training data as a list of 2D points. For example: [[0,0],[0,1],[1,0],[1,1]]")
x_train = eval(input("x_train = "))
print("Enter the corresponding target values. For example: [0,1,2,3]")
t_train = eval(input("t_train = "))
# Prompt the user to enter the learning rate and number of iterations
learning_rate = float(input("Enter the learning rate for the linear regression model: "))
num_iterations = int(input("Enter the number of iterations for the linear regression model: "))
# Train the linear regression model
classifier = linear_regression()
classifier.learn(x_train, t_train, learning_rate, num_iterations)
# Prompt the user to enter the test data
print("Enter the test data as a list of 2D points. For example: [[0.01,0.99],[0.99,0.01]]")
x_test = eval(input("x_test = "))
# Use the trained model to make predictions on the test data
y = classifier.predict(x_test)
print("The predicted target values for the test data are:", y)

Results:
A. Clus2
Enter a psychiatric disorder: Depression
Select a machine learning method: clustering (c) or random forests (rf): c
Machine learning method: Clustering
Omics data integrated: Genomics, metabolomics
API: python clustering 1.3.0 (pip install python-clustering)
Enter the number of samples:
750
Enter the centers of the blobs (in the format 'x1,y1;x2,y2;x3,y3'):
1,1;-1,-1;1,-1
Enter the cluster standard deviation:
0.4
Enter the random state:
0
￼
Enter eps value: 0.3
Enter minimum number of samples for a cluster: 10
Estimated number of clusters: 3
Estimated number of noise points: 18
Homogeneity: 0.953
Completeness: 0.883
V-measure: 0.917
Adjusted Rand Index: 0.952
Adjusted Mutual Information: 0.916
Silhouette Coefficient: 0.626
￼
B. ran_for
Enter a psychiatric disorder: Depression
Select a machine learning method: clustering (c) or random forests (rf): rf
Machine learning method: Random forests
Omics data integrated: Genomics, Transcriptomics, Epigenomics
API: scikit-learn 1.2.2 (pip install scikit-learn)
Enter the number of samples: 1000
Enter the number of features: 4
Enter the number of informative features: 2
Enter the number of redundant features: 0
Enter the random state: 0
Shuffle the data? (y/n)n
Enter the maximum depth for the random forest classifier: 2
Enter value for feature 1: 0
Enter value for feature 2: 2
Enter value for feature 3: 4
Enter value for feature 4: 6
The predicted class label for input data [0.0, 2.0, 4.0, 6.0] is 1.

C. pca1
Enter a psychiatric disorder: Schizophrenia
Machine learning method: Principal component analysis
Omics data integrated: Genomics, metabolomics
API: scikit-learn 1.2.2 (pip install scikit-learn)
Enter the number of principal components to extract: 3
Enter the batch size for the transformer: 100

D. lin_model
Enter a psychiatric disorder: Attention deficit hyperactivity disorder
Machine learning method: Linear models
Omics data integrated: Genomics, epigenomics
API: scikit-learn 1.2.2 (pip install scikit-learn)
Enter the number of data points: 4
Enter the number of features: 2
Enter the features for data point 1
Feature 1: -1
Feature 2: -1
Enter the features for data point 2
Feature 1: -2
Feature 2: -1
Enter the features for data point 3
Feature 1: 1
Feature 2: 1
Enter the features for data point 4
Feature 1: 2
Feature 2: 1
Enter the target label for data point 1: 1
Enter the target label for data point 2: 1
Enter the target label for data point 3: 2
Enter the target label for data point 4: 2
Enter the features for the data point to predict:
Feature 1: -0.8
Feature 2: -1
The predicted label for the input data point is: 1

E. linear_reg1
Enter a psychiatric disorder: Dementia
Machine learning method: Linear/logistic regression
Omics data integrated: Genomics, transcriptomics, epigenomics, proteomics
API: py4linear-regression 0.0.5 (pip install py4linear-regression)
Enter the training data as a list of 2D points. For example: [[0,0],[0,1],[1,0],[1,1]]
x_train = [0,0],[0,1],[1,0],[1,1]
Enter the corresponding target values. For example: [0,1,2,3]
t_train = [0,1,2,3]
Enter the learning rate for the linear regression model: 0.01
Enter the number of iterations for the linear regression model: 100
Iter : 1 Loss : 3.065792046069714
Iter : 2 Loss : 2.7404565729264605
Iter : 3 Loss : 2.4554782163162194
Iter : 4 Loss : 2.203825566783748
Iter : 5 Loss : 1.9810042822723417
Iter : 6 Loss : 1.783766381893942
...
Enter the test data as a list of 2D points. For example: [[0.01,0.99],[0.99,0.01]]
x_test = [0.01,0.99],[0.99,0.01]
The predicted target values for the test data are: [1.14859268 2.024535 ]
