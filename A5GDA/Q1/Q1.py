import numpy
import pandas
from math import sqrt, pi

def choose(output, mu0, mu1):
	if output == 0:
		return mu0
	else:
		return mu1

def computePhi(idx, phi):
	if idx == 0:
		return (1 - phi)
	else:
		return phi

def computePXPY(feature, mu, sigma):
	mul = ((0.5/pi)**(len(feature)/2))*sqrt(1/abs(numpy.linalg.det(sigma)))

	vec = feature - mu

	mat = numpy.dot(vec, numpy.linalg.inv(sigma))
	mat = numpy.dot(mat, vec.T)

	return mul*numpy.exp(-mat*(1/2))

def computeParameters(features_matrix, output):
	m = features_matrix.shape[0]
	phi = (1/m)*(numpy.int64(output == 1)).sum()
	
	mu0 = (numpy.dot(numpy.transpose(numpy.int64(output == 0)), features_matrix))/((numpy.int64(output == 0)).sum())
	mu1 = (numpy.dot(numpy.transpose(numpy.int64(output == 1)), features_matrix))/((numpy.int64(output == 1)).sum())

	ent = None
	for i in range(m):
		temp = features_matrix[i] - choose(output[i], mu0, mu1)

		val = numpy.dot(temp.T, temp)
		if ent is None:
			ent = val
		else:
			ent += val

	sigma = (1/m)*ent
	
	return (phi, mu0, mu1, sigma)

def computeExpec(phi, mu0, mu1, sigma, test_features, test_output):
	exp_output_sc = numpy.zeros(len(test_output))
	m = len(test_output)

	for i in range(m):
		val1 = computePXPY(test_features[i], mu0, sigma)
		val2 = computePXPY(test_features[i], mu1, sigma)
		prob1 = val1*computePhi(0, phi)
		prob2 = val2*computePhi(1, phi)

		if prob1 > prob2:
			exp_output_sc[i] = 0
		else:
			exp_output_sc[i] = 1

	return exp_output_sc

def get_np_data(data_frame, features):
	df_features = data_frame[features]
	features_matrix = df_features.to_numpy()
	
	return (features_matrix)

def compute(exp_output_sc, test_output):
	true_positive, false_positive, false_negative, true_negative = 0, 0, 0, 0
	for i in range(len(test_output)):
		if exp_output_sc[i] == 1 and test_output[i] == 1:
			true_positive += 1
		elif exp_output_sc[i] == 1 and test_output[i] == 0:
			false_positive += 1
		elif exp_output_sc[i] == 0 and test_output[i] == 1:
			false_negative += 1
		else:
			true_negative += 1
	
	return (true_positive, false_positive, false_negative, true_negative)

df = pandas.read_csv("data_set.csv")
df['average_of2'] = (df['test1'] + df['test2'])/2
df['geometric_mean'] = (abs(df['test1']*df['test2']))**(1/2)
df['avg_ag'] = (df['average_of2'] + df['geometric_mean'])/2

combined_data = get_np_data(df, ['test1', 'test2', 'average_of2', 'geometric_mean', 'avg_ag', 'label'])
length = len(combined_data)

train, test = combined_data[:round(length*0.7)], combined_data[round(length*0.7):]
features_matrix, output = train[:,:5], train[:,5:]
test_features, test_output = test[:,:5], test[:,5:]
numpy.set_printoptions(formatter={'float_kind':'{:f}'.format})
print("Gaussian discriminant analysis model")

phi, mu0, mu1, sigma = computeParameters(features_matrix, output)
print("phi :", phi)
print("mu0 :", mu0)
print("mu1 :", mu1)
print("sigma :")
print(sigma)

exp_output_sc = computeExpec(phi, mu0, mu1, sigma, test_features, test_output)
true_positive, false_positive, false_negative, true_negative = compute(exp_output_sc, test_output)

test_accuracy = ((true_positive + true_negative)*100)/len(exp_output_sc)
print("Test Accuracy :", round(test_accuracy, 2))
print("true_positive :", true_positive, "  false_positive :", false_positive)
print("false_negative :", false_negative, "  true_negative :", true_negative)

