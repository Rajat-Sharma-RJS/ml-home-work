import numpy
import matplotlib.pyplot as plt
import pandas
from math import sqrt

def expected_value(features_matrix, final_mat):
	exp_value = numpy.matmul(features_matrix, final_mat)
	exp_value = 1/(1 + numpy.exp(-exp_value))
	return exp_value

def feature_derivative(errors, feature):
	derivative = numpy.sum(errors*feature)
	return derivative


def batchGDA(features_matrix, output, alpha, epsilon, max_iter, initial_weights):
	converge = False
	weights = initial_weights
	m = features_matrix.shape[0]
	itr = 0

	J = (abs(expected_value(features_matrix, weights) - output)).sum()
	arr = []
	while not converge:
		prediction = expected_value(features_matrix, weights)
		errors = prediction - output

		grad = numpy.zeros(len(weights))

		for i in range(len(weights)):
			derivative = feature_derivative(errors, features_matrix[:, i])
			grad[i] = (1.0/m)*derivative

		temp = numpy.zeros(len(weights))

		for i in range(len(weights)):
			temp[i] = weights[i] - alpha*grad[i]

		for i in range(len(weights)):
			weights[i] = temp[i]

		e = (abs(expected_value(features_matrix, weights) - output)).sum()
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


df = pandas.read_csv("Academic_data.csv")
train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]
figure, axis = plt.subplots(1, 2)

#modeling data
features_matrix, output = get_np_data(train, ['score1', 'score2'], 'label')
test_features, test_output = get_np_data(test, ['score1', 'score2'], 'label')

alpha = 6e-8
epsilon = 4e-8
max_iter = 2e6

# predictor
initial_weights = numpy.zeros(len(features_matrix[0]))
weights, vec = batchGDA(features_matrix, output, alpha, epsilon, max_iter, initial_weights)
print("weights :", weights)

#prediction on testing data

exp_output = expected_value(test_features, weights)
err = abs(exp_output - test_output)
correct = len(err[err <= 0.5])
acc = correct*100/len(err)
print("Test accuracy :", round(acc, 2),"%")

axis[0].plot(range(1000), (vec[len(vec)-1000:]/vec[len(vec)-1])*10, '-')
axis[0].set_title("Without Feature Scaling")
axis[0].set_xlabel('iteration')
axis[0].set_ylabel('Error')

#after scaling
print("----------------------------------------------------------------------------")
print("After scaling")
features_matrix_sc = features_matrix
features_matrix_sc = scaling(features_matrix_sc)
initial_weights_sc = numpy.ones(len(features_matrix_sc[0]))
weights_sc, vec_sc = batchGDA(features_matrix_sc, output, alpha, epsilon, max_iter, initial_weights_sc)
print("weights :", weights_sc)

test_features_sc = test_features
test_features_sc = scaling(test_features_sc)

exp_output_sc = expected_value(test_features_sc, weights_sc)
err_sc = abs(exp_output_sc - test_output)
correct_sc = len(err_sc[err_sc <= 0.5])
acc_sc = correct_sc*100/len(err_sc)
print("Test accuracy :", round(acc_sc, 2),"%")

axis[1].plot(range(1000), (vec_sc[len(vec_sc)-1000:]/vec_sc[len(vec_sc)-1])*10, '-')
axis[1].set_title("With Feature Scaling")
axis[1].set_xlabel('iteration')
axis[1].set_ylabel('Error')

#plotting
figure.set_size_inches(10, 6)
plt.show()