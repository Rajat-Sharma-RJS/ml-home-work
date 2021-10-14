import pandas
import numpy
import matplotlib.pyplot as plt
from math import sqrt

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

def expected_value(features_matrix, final_mat):
	exp_value = numpy.matmul(features_matrix, final_mat)
	return exp_value

def compute(feature, given, tau):
	vec = feature - given
	d = numpy.dot(vec, vec)
	val = -(d/(2*tau*tau))
	return numpy.exp(val)

def lwr(features_matrix, output, given, tau):
	#taking transpose
	features_transpose = numpy.transpose(features_matrix)
	#weight matrix
	weight = numpy.identity(len(features_matrix))
	#updating
	for i in range(len(features_matrix)):
		weight[i, i] = compute(features_matrix[i], given, tau)

	product_of2 = numpy.matmul(features_transpose, weight)
	product_comb = numpy.matmul(product_of2, features_matrix)
	#print(product_comb)
	#taking inverse
	inverse_of2 = numpy.linalg.inv(product_comb)
	product_of3 = numpy.matmul(inverse_of2, features_transpose)
	product_weight = numpy.matmul(product_of3, weight)

	final_mat = numpy.matmul(product_weight, output)
	predict = expected_value(given, final_mat)
	return predict

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

def experiment(features_matrix, output, test_features, test_output, tau):
	prediction = numpy.zeros(len(test_output))

	for i in range(len(prediction)):
		prediction[i] = lwr(features_matrix, output, test_features[i], tau)

	rss = ((prediction - test_output)**2).sum()
	return (prediction, round(rss))

df = pandas.read_csv("Housing_Price_Data.csv")
train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]

features_matrix, output = get_np_data(train, ['lotsize', 'bedrooms', 'bathrms'], 'price')
#feature scaling is needed in this otherwise it will lead to singular matrix i.e error in code
features_matrix = scaling(features_matrix)

test_features, test_output = get_np_data(test, ['lotsize', 'bedrooms', 'bathrms'], 'price')
test_features = scaling(test_features)

#checking for different tau values
tau = 1
arr1, e1 = experiment(features_matrix, output, test_features, test_output, tau)
print("rss error (tau = 1):", e1)
tau = 5
arr2, e2 = experiment(features_matrix, output, test_features, test_output, tau)
print("rss error (tau = 5):", e2)
tau = 10
arr3, e3 = experiment(features_matrix, output, test_features, test_output, tau)
print("rss error (tau = 10):", e3)

x = numpy.array([i for i in range(len(test_output))])

figure, axis = plt.subplots(1, 2)

axis[0].plot(x, arr1, 'r-', x+3, arr2, 'g-', x, arr3, 'b-')
axis[0].set_title("Red (tau = 1), Green (tau = 5), Blue (tau = 10)")
axis[0].set_xlabel("x")
axis[0].set_ylabel("predicted values")

#calculating suitable tau
tau = [0.13]
for i in range(87):
	tau.append(tau[i]+0.01)

tau = numpy.array(tau)

error = numpy.zeros(len(tau))
j = 0
for i in range(len(tau)):
	arr, e = experiment(features_matrix, output, test_features, test_output, tau[i])
	error[i] = e
	if error[i] < error[j]:
		j = i

print("\nmin RSS error :", round(error[j]), ", for tau :", round(tau[j], 2))

axis[1].plot(tau, error, 'r-')
axis[1].set_title("Finding Best fit Tau")
axis[1].set_xlabel("tau")
axis[1].set_ylabel("Error")

figure.set_size_inches(15, 8)
plt.show()
