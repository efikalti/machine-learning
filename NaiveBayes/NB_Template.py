# =============================================================================
# HOMEWORK 4 - BAYESIAN LEARNING
# NAIVE BAYES ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

from sklearn import datasets
from sklearn import pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load text data.
textData = datasets.fetch_20newsgroups()

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(textData.data, textData.target, random_state = 0)

print(x_train.shape)

# Create pipeline
textTokenizer = text.TfidfVectorizer(x_train)
clf = MultinomialNB()
pipeline_model = pipeline.Pipeline([('TextTokenizer', textTokenizer), ('NaiveBayes', clf)])
fit_prior_values = [True, False]

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

        # Set metric scores
        precision = result[0]
        recall = result[1]
        f1 = result[2]

        print("Class prior probabilites: " + str(fit_prior))
        print("Alpha: %f" % alpha)
        print("Recall: %f" % recall)
        print("Precision: %f" % precision)
        print("F1: %f" % f1)
        print()


        plt.figure(figsize=[13, 6])
        # Calculate confusion matrix
        confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
        ax = sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="OrRd", cbar=False,
                         xticklabels=textData.target_names, yticklabels=textData.target_names)
        # Plot confusion matrix
        fitStr = ''
        if fit_prior is True:
            fitStr = '_t'
        else:
            fitStr = '_f'
        plt.title('Multinomial NB - Confusion matrix (a = %.1f) [Prec = %.5f, Rec = %.5f, F1 = %.5f] with fit_prior= %s' % (alpha, precision, recall, f1, str(fit_prior)))
        plt.xlabel('True output')
        plt.ylabel('Predicted output')
        plt.savefig('results/c_m_' + str(alpha) + fitStr + '.png')
        plt.clf()
