
import pandas as pd

import re

from sklearn import metrics
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter


xls_writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')


def main(clean_data=False):

    # Read datas from csv file
    if clean_data is True:
        # Read original data
        data = pd.read_csv("hate_tweets.csv")
        col_start = 1
    else:
        # Read transformed data
        data = pd.read_csv("clean_hate_tweets.csv")
        col_start = 5

    # Get number for rows and columns
    rows = data.shape[0]
    cols = data.shape[1]

    # Separate class and data variable
    if clean_data is True:
        Y = data.iloc[0:rows , col_start:cols-1].values
        X = data.iloc[0:rows, cols-1].values
        # Clean and write them to new csv
        write_csv(X, Y)
    else:
        Y = data.iloc[0:rows , col_start:cols-1].values
        Y = Y.flatten().tolist()
        X = data.iloc[0:rows, cols-1].values.tolist()

        data_info = pd.DataFrame(columns=['Total Samples', 'Hate Speech', 'Offensive Language', 'Neither'])
        hs = 0
        ol = 0
        ne = 0
        for i in range(0, len(X)):
            if Y[i] == 0:
                hs += 1
            elif Y[i] == 1:
                ol += 1
            else:
                ne += 1

        data_info = data_info.append({ 'Total Samples': rows, 'Hate Speech': hs, 'Offensive Language': ol, 'Neither': ne}, ignore_index=True)
        data_info.to_excel(xls_writer, sheet_name='Dataset info')

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0)

        # Train and predict using Multinomial Naive Bayes algorithm
        NB(x_train, y_train, x_test, y_test)


# Naive Bayes
def NB(x_train, y_train, x_test, y_test):

    # Create pipeline
    textTokenizer = text.TfidfVectorizer()
    clf = MultinomialNB()
    pipeline_model = pipeline.Pipeline([('TextTokenizer', textTokenizer), ('NaiveBayes', clf)])
    fit_prior_values = [True, False]

    # Initialize results dataframe
    results = pd.DataFrame(columns=['Uniform Probabilites', 'Alpha', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Naive Bayes
    for alpha in range(1, 10, 2):
        # Set alpha
        for fit_prior in fit_prior_values:
            # Set parameters
            pipeline_model.set_params(NaiveBayes__alpha=alpha, NaiveBayes__fit_prior=fit_prior)

            # Train model
            pipeline_model.fit(x_train, y_train)

            # Predict on test data
            y_predicted = pipeline_model.predict(x_test)

            # Compute metrics for precision, recall, fscore
            result = metrics.precision_recall_fscore_support(y_test, y_predicted, average='macro')
            accuracy = metrics.accuracy_score(y_test, y_predicted)

            # Set metric scores
            precision = result[0]
            recall = result[1]
            f1 = result[2]

            # If prior is false, a uniform prior is used
            prior = "True"
            if fit_prior is True:
                prior = "False"

            results = results.append({ 'Uniform Probabilites': prior, 'Alpha': alpha, 'Accuracy': float("%0.3f"%accuracy),
                          'Precision': float("%0.3f"%precision), 'Recall':  float("%0.3f"%recall), 'F1': float("%0.3f"%f1)}, ignore_index=True)


            plt.figure(figsize=[13, 6])
            # Calculate confusion matrix
            confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
            ax = sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="OrRd", cbar=False)
            # Plot confusion matrix
            fitStr = ''
            if fit_prior is True:
                fitStr = '_t'
            else:
                fitStr = '_f'
            plt.title('Multinomial NB - Confusion matrix (a = %.1f) [Acc= %.2f, Prec = %.2f, Rec = %.2f, F1 = %.2f] with fit_prior= %s' % (alpha, accuracy, precision, recall, f1, str(fit_prior)))
            plt.xlabel('True output')
            plt.ylabel('Predicted output')
            plt.savefig('results/c_m_' + str(alpha) + fitStr + '.png')
            plt.clf()

    results.to_excel(xls_writer, sheet_name='Multinomial Naive Bayes')
    # Write all the results
    xls_writer.save()


# Remove substring and special characters
def clean_string(string):

    # Remove RT tags
    pattern = '%s(.*?)%s' % (re.escape("RT"), re.escape(":"))
    s = re.sub(pattern, "", string)
    pattern = '%s(.*?)%s' % (re.escape("@"), re.escape(" "))
    s = re.sub(pattern, "", s)

    # Remove &amp; substrings
    s = re.sub(r"&amp;", "", s)

    # Remove emoticons
    pattern = '%s(.*?)%s' % (re.escape("&#"), re.escape(";"))
    s = re.sub(pattern, " ", s)

    # Remove links
    s = re.sub(r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", " ", s)

    # Remove special characters
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s)

    # Remove duplicate spaces
    s = re.sub(' +', ' ', s)

    # Remove trailing and duplicate spaces
    s = s.strip()

    # Check for empty strings
    if len(s) == 0:
        return "", False

    return s, True


# Clean X and write new data to csv
def write_csv(X, Y):
    valid = []
    for i in range(0, len(X)):
        X[i], valid_string = clean_string(X[i])
        # Add valid id to list of ids
        if valid_string is True:
            valid.append(i)

    results = pd.DataFrame(columns=['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet'])
    for index in range(0, len(valid)):
        i = valid[index]
        results = results.append({ 'count': Y[i][0], 'hate_speech': Y[i][1],
                          'offensive_language': Y[i][2],
                          'neither': Y[i][3], 'class':  Y[i][4],
                          'tweet': X[i]}, ignore_index=True)
    # Write all the results
    results.to_csv('clean_hate_tweets.csv', sep=',', index=True)


if __name__ == "__main__":
    main()
