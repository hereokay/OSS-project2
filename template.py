#PLEASE WRITE THE GITHUB URL BELOW!
#
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.svm import SVC


def load_dataset(dataset_path):
	data_df = pd.read_csv(dataset_path)
	return data_df


def dataset_stat(dataset_df):
	return data_df.shape[1]-1, data_df.groupby("target").size()[0], data_df.groupby("target").size()[1]


def split_dataset(dataset_df, testset_size):
	# To-Do: Implement this function
	data_data = data_df.drop(columns="target", axis=1)
	data_target = dataset_df['target']
	a, b, c, d = train_test_split(data_data, data_target, test_size=testset_size)
	return a, b, c, d


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	pipe = make_pipeline(
		StandardScaler(),
		tree.DecisionTreeClassifier()
	)

	pipe.fit(x_train, y_train)
	acc = accuracy_score(y_test, pipe.predict(x_test))
	prec = precision_score(y_test, pipe.predict(x_test))
	recall = recall_score(y_test, pipe.predict(x_test))
	return acc, prec, recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
	pipe = make_pipeline(
		StandardScaler(),
		RandomForestClassifier()
	)

	pipe.fit(x_train, y_train)
	acc = accuracy_score(y_test, pipe.predict(x_test))
	prec = precision_score(y_test, pipe.predict(x_test))
	recall = recall_score(y_test, pipe.predict(x_test))
	return acc, prec, recall


def svm_train_test(x_train, x_test, y_train, y_test):
	pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)

	pipe.fit(x_train, y_train)
	acc = accuracy_score(y_test, pipe.predict(x_test))
	prec = precision_score(y_test, pipe.predict(x_test))
	recall = recall_score(y_test, pipe.predict(x_test))
	return acc, prec, recall


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print("Accuracy: ", acc)
	print("Precision: ", prec)
	print("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print("Number of features: ", n_feats)
	print("Number of class 0 data entries: ", n_class0)
	print("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)