import pandas
import numpy
import matplotlib.pyplot as plt
from math import sqrt
import random as rd

def expected_value(features_matrix, final_mat):
	exp_value = numpy.dot(features_matrix, final_mat)
	return exp_value

def feature_derivative(errors, feature):
	derivative = (errors*feature)
	return derivative


def stochasticGDA(features_matrix, output, alpha, epsilon, max_iter):
	converge = False
	weights = numpy.ones(len(features_matrix[0]))
	weights[0], weights[1], weights[3] = 57500, 12900, 15700
	m = features_matrix.shape[0]
	itr = 0

	J = ((expected_value(features_matrix, weights) - output)**2).sum()
	arr = []
	while not converge:
		k = rd.randint(0, (m-1))
		prediction = expected_value(features_matrix[k], weights)
		errors = prediction - output[k]

		grad = numpy.zeros(len(weights))

		for i in range(len(weights)):
			derivative = feature_derivative(errors, features_matrix[k, i])
			grad[i] = (1.0/m)*derivative

		temp = numpy.zeros(len(weights))

		for i in range(len(weights)):
			temp[i] = weights[i] - alpha*grad[i]

		for i in range(len(weights)):
			weights[i] = temp[i]

		e = ((expected_value(features_matrix[k], weights) - output[k])**2)
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

df = pandas.read_csv("Housing_Price_Data.csv")
train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]

#modeling data
features_matrix, output = get_np_data(train, ['lotsize', 'bedrooms', 'bathrms'], 'price')
alpha = 5e-5
epsilon = 1e-12
max_iter = 2e6

# features scaling
print("After Scaling")
features_matrix = scaling(features_matrix)

# predictor
print("Stochastic ! Hence results may vary")
weights, vec = stochasticGDA(features_matrix, output, alpha, epsilon, max_iter)
print("weights :", weights)

#prediction on testing data
test_features, test_output = get_np_data(test, ['lotsize', 'bedrooms', 'bathrms'], 'price')
test_features = scaling(test_features) #scaling
exp_output = expected_value(test_features, weights)
rss = ((test_output - exp_output)**2).sum()
print("RSS error(inverse of accuracy) :", round(rss))

#plotting
plt.plot(range(1000), (vec[len(vec)-1000:]/vec[len(vec)-1]), '-')
plt.xlabel('iteration')
plt.ylabel('Error')
plt.show()
