import pandas
import numpy
import matplotlib.pyplot as plt
from math import sqrt
import random as rd

def expected_value(features_matrix, final_mat):
	exp_value = numpy.matmul(features_matrix, final_mat)
	return exp_value

def feature_derivative(errors, feature):
	derivative = numpy.sum(errors*feature)
	return derivative


def miniBGDA(features_matrix, output, alpha, epsilon, max_iter, size, reg = 0.0):
	converge = False
	weights = numpy.ones(len(features_matrix[0]))
	weights[0], weights[2], weights[3] = -4900, 5500, 19000
	m = features_matrix.shape[0]
	itr = 0

	J = ((expected_value(features_matrix, weights) - output)**2).sum()
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

		e = ((expected_value(features_matrix[(k-half):(k+half)], weights) - output[(k-half):(k+half)])**2).sum()
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


df = pandas.read_csv("Housing_Price_Data.csv")
train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]

#modeling data
features_matrix, output = get_np_data(train, ['lotsize', 'bedrooms', 'bathrms'], 'price')
alpha = 4e-7
epsilon = 0.11
max_iter = 2e6
size = 30 #batch

figure, axis = plt.subplots(2, 2)
# predictor
print("Mini Batch GDA ! Hence results may vary")
print("Without Regularization")
weights, vec = miniBGDA(features_matrix, output, alpha, epsilon, max_iter, size)
print("weights :", weights)

#prediction on testing data
test_features, test_output = get_np_data(test, ['lotsize', 'bedrooms', 'bathrms'], 'price')

exp_output = expected_value(test_features, weights)
rss1 = ((test_output - exp_output)**2).sum()
print("RSS error(inverse of accuracy) :", round(rss1))
print("------------------------------------------------------------------------------------")

#setting
axis[0][0].plot(range(1000), (vec[len(vec)-1000:]/vec[len(vec)-1]), '-')
axis[0][0].set_title("lambda = 0.0")
axis[0][0].set_xlabel("iteration")
axis[0][0].set_ylabel("Error")

print("With Regularization of 0.1")
weights, vec = miniBGDA(features_matrix, output, alpha, epsilon, max_iter, size, 0.1)
print("weights :", weights)

#prediction on testing data
exp_output = expected_value(test_features, weights)
rss2 = ((test_output - exp_output)**2).sum()
print("RSS error(inverse of accuracy) :", round(rss2))
print("------------------------------------------------------------------------------------")

#setting
axis[0][1].plot(range(1000), (vec[len(vec)-1000:]/vec[len(vec)-1]), '-')
axis[0][1].set_title("lambda = 0.1")
axis[0][1].set_xlabel("iteration")
axis[0][1].set_ylabel("Error")

print("With Regularization of 0.01")
weights, vec = miniBGDA(features_matrix, output, alpha, epsilon, max_iter, size, 0.01)
print("weights :", weights)

#prediction on testing data
exp_output = expected_value(test_features, weights)
rss3 = ((test_output - exp_output)**2).sum()
print("RSS error(inverse of accuracy) :", round(rss3))

#setting
axis[1][0].plot(range(1000), (vec[len(vec)-1000:]/vec[len(vec)-1]), '-')
axis[1][0].set_title("lambda = 0.01")
axis[1][0].set_xlabel("iteration")
axis[1][0].set_ylabel("Error")

#scaling errors
y = numpy.array([round(rss1), round(rss3), round(rss2)])
x = numpy.array(["0.0", "0.01", "0.1"])

axis[1][1].bar(x, ((y+7e8)-min(y)))
axis[1][1].set_title("RSS v/s lambda")
axis[1][1].set_xlabel("lambda")
axis[1][1].set_ylabel("Residual Sum of Square of errors (RSS)")
#plotting
figure.set_size_inches(10, 10)
plt.show()
