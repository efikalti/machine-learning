# =============================================================================
# HOMEWORK 1 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'cross_validation' package, which will help test our model
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, ensemble
from sklearn.model_selection import train_test_split


# Classifier variables
max_depths = [3, 4, 5, 6, 7, 8]
criterions = ['gini', 'entropy']
estimators=[25, 50, 75, 100]
average='weighted'


# Load breast cancer dataset from sklearn
breastCancer = datasets.load_breast_cancer(return_X_y=False)


# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily with a large number of features
numberOfFeatures = 8
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Iterate through every criterion
for criterion in criterions:
    # Iterate through every depth
    for depth in max_depths:
        # Iterate through number of estimators
        for estimator in estimators:
            # Set random forest clasifier with this critetion, depth and number of estimators
            model = ensemble.RandomForestClassifier(n_estimators=estimator, criterion=criterion, max_depth=depth)

            # Train model with the train set
            predictor = model.fit(x_train, y_train)

            # Predict with the trained model on the test set
            y_predicted = predictor.predict(x_test)

            # Calculate precision, recall, fscore with micro average
            result = metrics.precision_recall_fscore_support(y_test, y_predicted, average=average)

            # Print the results of the trained model
            print('# =============================================================================')
            print('Number of features:      ' + str(numberOfFeatures))
            print('Criterion:               ' + str(criterion))
            print('Max tree depth:          ' + str(depth))
            print('Number of estimators:    ' + str(estimator))
            print('Average:                 ' + str(average))
            print('Results:')
            print('Precision:               ' + str(result[0]))
            print('Recall:                  ' + str(result[1]))
            print('Fscore:                  ' + str(result[2]))
            print()
