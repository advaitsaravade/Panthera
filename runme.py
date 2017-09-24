import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset = pandas.read_csv('creditcard.csv') #or use a url

print("\n\nFinished loading dataset...\n\nStarting dataset analysis...\n\n");

#print(dataset.head(20))
#print(dataset.groupby('Class').size())

# print out the dimensions of the data in the form (rows, columns(aka fields))
# print(dataset.shape)
# print out the first 20 rows of the data
# print(dataset.head(20))
# print out some stats about the data (like count, mean, the min and max values as well as some percentiles)
# print(dataset.describe())

# okay you got my attention. Can we do some custom analytics?

# print(dataset.groupby('Class').size())

# Woah. Can we visualize data now?
# Like a graph bro?

# dataset.hist()
# plt.show()

# Awesome. Is there any way to find correlations between all of the columns?

# scatter_matrix(dataset)
# plt.show()
array = dataset.values
X = array[:,0:30]
Y = array[:,30]

validation_size = 0.2
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
# SGDClassifier should theoretically work best, but it needs more processing power. We just had an old macbook DAMMIT
#models.append(('SGD', SGDClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
prevMean = 0
counter = 0
for name, model in models:
    counter += 1
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) #msg is useful for debugging
    if (cv_results.mean() > prevMean):
        if name == 'LR':
            algoUse = 1
        elif name == 'LDA':
            algoUse = 2
        else:
            algoUse = 3 #KNN and non-linear
        prevMean = cv_results.mean()
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) #msg is useful for debugging
    print(str(((counter*100)/3))+'% of pre-analysis computations complete...\n\n')
# Compare Algorithms
# Make predictions on validation dataset
algo = LinearDiscriminantAnalysis()
algoName = ''
algoType = ''
if algoUse == 1:
    algo = LogisticRegression()
    algoName = 'Logistic Regression'
    algoType = 'Linear'
elif algoUse == 2:
    knn = LinearDiscriminantAnalysis()
    algoName = 'Linear Distriminant Analysis'
    algoType = 'Linear'
elif algoUse == 3:
    knn = KNeighborsClassifier()
    algoName = 'K Neighbor Classifier'
    algoType = 'Non-Linear'
algo = LinearDiscriminantAnalysis()
algo.fit(X_train, Y_train)
predictions = algo.predict(X_validation)
matrix = classification_report(Y_validation, predictions)
print('\nThis data set was identified as best suited for the '+algoName+' algorithm. Using this \033[92m'+algoType+' algorithm, Panthera had a '+str((accuracy_score(Y_validation, predictions)*100))+'% accuracy\033[0m in correctly identifying credit card transactions as either legitimate or fraudulent in this entire dataset.\n\n')
print('Final statistics on data:\n\n')
target_names = ['Legitimate Transaction', 'Fraudulent Transaction']
print(classification_report(Y_validation, predictions,target_names=target_names))