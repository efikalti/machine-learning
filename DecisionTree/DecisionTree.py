# =============================================================================
# HOMEWORK 1 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# IMPORTS
import os
os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin/"


from sklearn import datasets, tree, metrics
from sklearn.model_selection import train_test_split
import graphviz

# Classifier variables
max_depths = [3, 4, 5, 6, 7, 8]
criterions = ['gini', 'entropy']
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
        # Only one available average = micro

        # Set decision tree clasifier with this critetion and depth
        model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=depth)

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
        print('Average:                 ' + str(average))
        print('Results:')
        print('Precision:               ' + str(result[0]))
        print('Recall:                  ' + str(result[1]))
        print('Fscore:                  ' + str(result[2]))
        print()


        # Export trained model graph
        dot_data = tree.export_graphviz(model, feature_names = breastCancer.feature_names[:numberOfFeatures], class_names = breastCancer.target_names)

        # Export the graph into a PDF file located within the same folder as this script.
        graph = graphviz.Source(dot_data)
        graph.render("weighted_results/breastCancerTreePlot_" + str(criterion) + "_" + str(depth))
