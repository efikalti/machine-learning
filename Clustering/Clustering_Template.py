# =============================================================================
# HOMEWORK 7 - CLUSTERING
# CLUSTERING ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering, KMeans

# Load breast cancer dataset from sklearn
breastCancer = datasets.load_breast_cancer(return_X_y=False)

# Get samples from the data, and keep only the features that you wish
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Scale data
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Model parameters
n_init = [5, 10, 25]
affinities = ['nearest_neighbors', 'rbf']
n_neighbors = [5, 10, 20]

# Run Spectral Clustering
for n in n_init:
    for affinity in affinities:
        if affinity == 'nearest_neighbors':
            for neighbors in n_neighbors:
                # Create Spectral Clustering model
                sc = SpectralClustering(n_clusters=2, affinity=affinity, n_init=n, assign_labels='discretize', n_neighbors=neighbors, random_state=0)

                # Train model
                sc.fit(x_train)

                # predict test data
                y_predicted = sc.fit_predict(x_test)

                # Calculate silhouette_score
                result = metrics.silhouette_score(x_test, y_predicted)

                print('# =============================================================================')
                print('Cluster:                             SpectralClustering')
                print('Kmeans runs:                         ' + str(n))
                print('Affinity:                            ' + affinity)
                print('Number of neighbors:                 ' + str(neighbors))
                print('Silhouette score:                    ' + str(result))
        else:
            # Create Spectral Clustering model
            sc = SpectralClustering(n_clusters=2, affinity=affinity, n_init=n, assign_labels='discretize', random_state=0)

            # Train model
            sc.fit(x_train)

            # predict test data
            y_predicted = sc.fit_predict(x_test)

            # Calculate silhouette_score
            result = metrics.silhouette_score(x_test, y_predicted)

            print('# =============================================================================')
            print('Cluster:                             SpectralClustering')
            print('Kmeans runs:                         ' + str(n))
            print('Affinity:                            ' + affinity)
            print('Silhouette score:                    ' + str(result))

# Run KMeans
for n in n_init:
    km = KMeans(n_clusters=2, n_init=n, random_state=0)

    # Train model
    km.fit(x_train)

    # predict test data
    y_predicted = km.fit_predict(x_test)

    # Calculate silhouette_score
    result = metrics.silhouette_score(x_test, y_predicted)

    print('# =============================================================================')
    print('Cluster:                             KMeans')
    print('Kmeans runs:                         ' + str(n))
    print('Silhouette score:                    ' + str(result))
