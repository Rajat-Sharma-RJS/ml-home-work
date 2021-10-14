import random as rd
import numpy
import matplotlib.pyplot as plt
import pandas
from math import sqrt

def expected_value(features_matrix, final_mat):
	exp_value = numpy.dot(features_matrix, final_mat)
	exp_value = 1/(1 + numpy.exp(-exp_value))
	return exp_value

def feature_derivative(errors, feature):
	derivative = numpy.sum(errors*feature)
	return derivative


def miniBGDA(features_matrix, output, alpha, epsilon, max_iter, initial_weights, size, reg):
	converge = False
	weights = initial_weights
	m = features_matrix.shape[0]
	itr = 0

	J = (abs(expected_value(features_matrix, weights) - output)).sum()
	arr = []
	half = int(size/2)
	while not converge:
		k = rd.randint(half, (m-half))
		prediction = expected_value(features_matrix[(k-half):(k+half)], weights)
		errors = prediction - output[(k-half):(k+half)]

		grad = numpy.zeros(len(weights))

		for i in range(len(weights)):
			derivative = feature_derivative(errors, features_matrix[(k-half):(k+half), i])
			grad[i] = (1.0/m)*derivative

		temp = numpy.zeros(len(weights))

		for i in range(len(weights)):
			temp[i] = (1 - (alpha*reg)/m)*weights[i] - alpha*grad[i]

		for i in range(len(weights)):
			weights[i] = temp[i]

		e = (abs(expected_value(features_matrix[(k-half):(k+half)], weights) - output[k])).sum()
		arr.append(e)
		if abs(J-e) <= epsilon:
			print("Converged, iteration :", itr)
			converge = True

		J = e
		itr += 1
		#print(J)

		if itr == max_iter:
			print("Max limit exceeded !")
			converge = True
	vec = numpy.array(arr)
	return(weights, vec)

def get_np_data(data_frame, features, output):
	data_frame['constant'] = 1 #adding a constant column
	features = ['constant'] + features #adding constant feature
	#selecting the columns related to features
	df_features = data_frame[features]
	#converting it to numpy matrix
	features_matrix = df_features.to_numpy()
	#now the output
	out = data_frame[output]
	out_arr = out.to_numpy()
	return (features_matrix, out_arr)


def scaling(features_matrix):
	m = features_matrix.shape[0]
	for i in range(len(features_matrix[0])) :
		if i == 0:
			continue
		feature = features_matrix[:, i]
		avg = feature.mean()
		vari = sqrt((1/m)*((feature - avg)**2).sum())
		features_matrix[:, i] = (feature - avg)/vari

	return features_matrix

def compute(exp_output_sc, test_output):
	true_positive, false_positive, false_negative, true_negative = [], [], [], []
	for i in range(len(test_output)):
		if exp_output_sc[i] > 0.5 and test_output[i] == 1:
			true_positive.append(exp_output_sc[i])
		elif exp_output_sc[i] > 0.5 and test_output[i] == 0:
			false_positive.append(exp_output_sc[i])
		elif exp_output_sc[i] <= 0.5 and test_output[i] == 1:
			false_negative.append(exp_output_sc[i])
		else:
			true_negative.append(exp_output_sc[i])
	true_positive, false_positive = numpy.array(true_positive), numpy.array(false_positive)
	false_negative, true_negative = numpy.array(false_negative), numpy.array(true_negative)
	return (true_positive, false_positive, false_negative, true_negative)

df = pandas.read_csv("Cleveland_heart_disease_data.csv")

train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]
figure, axis = plt.subplots(1, 1)

#modeling data
features_matrix, output = get_np_data(train, ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'num')
test_features, test_output = get_np_data(test, ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'num')

alpha = 1e-3
epsilon = 4e-9
max_iter = 2e6
size = 30 #mini batch
reg = 0.01

print("Mini Batch GDA ! Hence results may vary")
print("Adding regularization term of 0.01")
#after scaling
print("----------------------------------------------------------------------------")
print("After scaling")
features_matrix_sc = features_matrix
features_matrix_sc = scaling(features_matrix_sc)
initial_weights_sc = numpy.ones(len(features_matrix_sc[0]))
weights_sc, vec_sc = miniBGDA(features_matrix_sc, output, alpha, epsilon, max_iter, initial_weights_sc, size, reg)
print("weights :", weights_sc)

test_features_sc = test_features
test_features_sc = scaling(test_features_sc)

exp_output_sc = expected_value(test_features_sc, weights_sc)
err_sc = abs(exp_output_sc - test_output)
correct_sc = len(err_sc[err_sc <= 0.5])
acc_sc = correct_sc*100/len(err_sc)
print("Test accuracy(threshold = 0.5) :", round(acc_sc, 2),"%")

true_positive, false_positive, false_negative, true_negative = compute(exp_output_sc, test_output)

axis.plot(range(len(true_positive)), true_positive, 'b+', range(len(false_positive)), false_positive, 'r+', range(len(false_negative)), false_negative, 'b.', range(len(true_negative)), true_negative, 'r.')
axis.set_title("Logistic Classifier")
axis.set_xlabel('true_positive(blue +) false_positive(red +) false_negative(blue .) true_negative(red .)')
axis.set_ylabel('Prediction')

#plotting
figure.set_size_inches(8, 8)
plt.show()