# =============================================================================
# HOMEWORK 6 - Support Vector Machines
# SUPPORT VECTOR MACHINE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def main():

    # Read datas from csv file
    data = pd.read_csv("creditcard.csv")

    # Get number or rows and columns
    rows = data.shape[0]
    cols = data.shape[1]

    # From myData object, read included features and target variable
    X = data.iloc[1:rows , 0:cols-1].values
    Y = data.iloc[1:rows, cols-1].values


    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)


    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    # =============================================================================

    kernels=['poly', 'rbf', 'sigmoid']
    C = [0.1, 1, 10, 100]
    gammas = [0.01, 0.1, 1, 5]
    degrees = [2, 3, 5]

    results = pd.DataFrame(columns=['Kernel', 'C', 'Gamma', 'Degree', 'Precision', 'Recall', 'F1', 'Normal', 'Fraud'])

    # Create and test svm classifiers
    for kernel in kernels:
        for c in C:
            if kernel == 'poly':
                for degree in degrees:
                    model = svm.SVC(kernel=kernel, C=c, degree=degree)
                    model.fit(x_train, y_train)
                    results = predict(model=model, x_test=x_test, y_test=y_test, kernel=kernel, C=c, degree=degree, results=results)
            else:
                for gamma in gammas:
                    model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
                    model.fit(x_train, y_train)
                    results = predict(model=model, x_test=x_test, y_test=y_test, kernel=kernel, C=c, gamma=gamma, results=results)
    # Save results to csv
    results.to_csv('results.csv')


"""
Calculate and add SVM results
"""
def predict(model, x_test, y_test, kernel, C=1.0, gamma='-', degree='-', results=None):
    y_predicted = model.predict(x_test)
    confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
    result = metrics.precision_recall_fscore_support(y_test, y_predicted, average='macro')
    results = results.append({ 'Kernel': kernel, 'C': C, 'Gamma': gamma, 'Degree': degree,
                  'Recall': float("%0.2f"%result[1]), 'Precision':  float("%0.2f"%result[0]), 'F1': float("%0.2f"%result[2]),
                  'Normal': confusionMatrix[0], 'Fraud': confusionMatrix[1]}, ignore_index=True)
    return results


if __name__ == "__main__":
    main()
