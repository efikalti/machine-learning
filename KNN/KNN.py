# =============================================================================
# HOMEWORK 3 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot

# Read datas from csv file
data = pd.read_csv("diabetes.csv")

# Get number or rows and columns
rows = data.shape[0]
cols = data.shape[1]

# Separate data variables
X = data.iloc[0:rows , 0:cols-1]

# Separate class variable
Y = data.iloc[0:rows, cols-1]

# Scale data
scaler = MinMaxScaler()
scaler.fit(X)
X_rescaled = scaler.transform(X)

# Separate train and test data from original dataset
x_train, x_test, y_train, y_test = train_test_split(X_rescaled, Y, random_state=0)

# Initialize variables for number of neighbors
k = 200

# Array of neighbor values
neighbors = np.arange(1, k+1)

# Set metric average
average = 'macro'

# Set algorithm available parameters
weights = ['distance', 'uniform']
p_values = [1, 2, 4]

# Run the classification algorithm 'k' times (k = number of neighbors)
print('K Nearest Neighbors')
print()

for weight in weights:
    # Test each weight parameter
    for p_value in p_values:
        # Test each power parameter

        # Initial best score variables
        best_F1 = None
        bestF1_ind = None

        # Initialize arrays for holding values for recall, precision and F1 scores.
        prec = np.arange(1, k+1, dtype = np.float64)
        rec = np.arange(1, k+1, dtype = np.float64)
        f1 = np.arange(1, k+1, dtype = np.float64)

        for n in range(1, k+1):
            # Create model
            model = KNeighborsClassifier(n_neighbors=n, weights=weight, metric="minkowski", p=p_value)

            # Train model
            model.fit(x_train, y_train)

            # Predict for test data
            y_predicted = model.predict(x_test)

            # Calculate precision, recall, fscore with micro average
            result = metrics.precision_recall_fscore_support(y_test, y_predicted, average=average)

            # Set metric scores
            prec[n-1] = result[0]
            rec[n-1] = result[1]
            f1_score = result[2]
            f1[n-1] = f1_score

            # Compare current f1 and best f1
            if best_F1 is None and bestF1_ind is None:
                best_F1 = f1_score
                bestF1_ind = n
            else:
                if best_F1 < f1_score:
                    best_F1 = f1_score
                    bestF1_ind = n

        # Print final results with best f1 and best number of neighbors for this iteration
        print('# =============================================================================')
        print('Weight:                             ' + str(weight))
        print('Minkowski power:                     ' + str(p_value))
        print('Best F1:                            ' + str(best_F1))
        print('Number of neighbors with best F1:   ' + str(bestF1_ind))
        print('Precision:                          ' + str(prec[bestF1_ind - 1]))
        print('Recall:                             ' + str(rec[bestF1_ind - 1]))
        print('Average:                            ' + str(average))

        # Create plot data and save in png

        # Set data for plot
        df = pd.DataFrame({'Number of Neighbors': neighbors, 'Precision': prec, 'Recall': rec, 'F1': f1 })

        # Plot data
        pyplot.plot('Number of Neighbors', 'Precision', data=df, color='red')
        pyplot.plot('Number of Neighbors', 'Recall', data=df, color='blue')
        pyplot.plot('Number of Neighbors', 'F1', data=df, color='green')
        pyplot.legend()
        pyplot.title(label='KNN for weight = ' + str(weight) + ', Minkowski power = ' + str(p_value))
        pyplot.xlabel('Number of Neighbors')
        pyplot.ylabel('Metrics')
        pyplot.savefig('results/KNN_diabetes_plot_' + str(weight) + '_' + str(p_value) + '.png')
        pyplot.clf()
