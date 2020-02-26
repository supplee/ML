# modeling.py - evaluate candidate predictive models

# Load libraries
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path = "./iris.csv"
debug = 1
plot_descriptive_stats = 0
variableNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(path,names=variableNames)

if debug:
	# shape
	print(dataset.shape)

	# 20 line header for debug
	print(dataset.head(20))

	# Descriptive statistics
	print(dataset.describe())

	# Class distribution
	print(dataset.groupby('class').size())

if plot_descriptive_stats:
	## PLOTTING EXAMPLES -- disabled as these are shown in desciptive.py ##

	# box and whisker plots
	dataset.plot(kind='box', subplots=True, layout=(1,4), sharex=False, sharey=True)
	pyplot.show()

	# histogram
	dataset.hist()
	pyplot.show()

	# scatter plot matrix
	scatter_matrix(dataset)

learningArray = dataset.values
print(learningArray.shape) # Check - should be 150 x 5

## MODEL EVALUATION THROUGH K-FOLD CROSS-VALIDATION ##
X = learningArray[:,0:4] # independent - multivariate data for learning
y = learningArray[:,4] # dependent - species classification

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluation each model through stratified K fold cross-validation
bestModel = []

print('Model accuracy through stratified 10-fold cross-validation')
for trial in range(10):
	results = []
	names = []
	meanAccuracy = []
	X_train, X_validation, Y_train, Y_validation = train_test_split(X,y,test_size=0.20)

	for name,model in models:
		kfold = StratifiedKFold(n_splits=10, shuffle=True)
		cv_results = cross_val_score(model,X_train,Y_train,cv=kfold, scoring='accuracy')
		results.append(cv_results)
		names.append(name)
		meanAccuracy.append(cv_results.mean())
		#print('%s: %s' % (name, cv_results))
		print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	mostAccurate = max(meanAccuracy)
	index = meanAccuracy.index(mostAccurate)
	bestModel.append(names[index])

tableData = DataFrame(bestModel, columns = ['model'])
freqTable = tableData['model'].value_counts()
freqTable.plot(kind='bar',title='number of trials')
pyplot.show() # We see that linear discriminant analysis typically leads to best results

