# descriptive.py - show descriptive statistics of the data set

# Load libraries
from pandas import read_csv
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

## PLOTTING EXAMPLES ##

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(1,4), sharex=False, sharey=True)
pyplot.show()

# histogram
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)


