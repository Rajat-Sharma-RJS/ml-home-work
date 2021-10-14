import pandas
import numpy
import matplotlib.pyplot as plt

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

def regularize_fit(features_matrix, output, reg):
	#taking transpose
	features_transpose = numpy.transpose(features_matrix)
	product_of2 = numpy.matmul(features_transpose, features_matrix)
	#adding regularization constant
	regular = reg*numpy.identity(len(product_of2))
	#removing effect on constant parameter
	regular[0, 0] = 0
	#taking inverse
	inverse_of2 = numpy.linalg.inv((product_of2+regular))
	product_of3 = numpy.matmul(inverse_of2, features_transpose)

	final_mat = numpy.matmul(product_of3, output)
	
	return final_mat

def expected_value(features_matrix, final_mat):
	exp_value = numpy.matmul(features_matrix, final_mat)
	return exp_value

df = pandas.read_csv("Housing_Price_Data.csv")
train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]

#modeling data
features_matrix, output = get_np_data(train, ['lotsize', 'bedrooms', 'bathrms'], 'price')
#without regularization
print("Without Regularization")
final_mat = regularize_fit(features_matrix, output, 0)
w0, w1, w2, w3 = final_mat[0], final_mat[1], final_mat[2], final_mat[3]
print("w0 :", w0, ", w1 :", w1, ", w2 :", w2, ", w3 :", w3)
#computing error
test_features, test_output = get_np_data(test, ['lotsize', 'bedrooms', 'bathrms'], 'price')
exp_output = expected_value(test_features, final_mat)
rss1 = ((test_output-exp_output)*(test_output-exp_output)).sum()
print("RSS error(inverse of accuracy) :", round(rss1))

#with regularization
print("With Regularization constant of 0.1")
final_mat = regularize_fit(features_matrix, output, 0.1)
w0, w1, w2, w3 = final_mat[0], final_mat[1], final_mat[2], final_mat[3]
print("w0 :", w0, ", w1 :", w1, ", w2 :", w2, ", w3 :", w3)
#computing error
exp_output = expected_value(test_features, final_mat)
rss2 = ((test_output-exp_output)*(test_output-exp_output)).sum()
print("RSS error(inverse of accuracy) :", round(rss2))

#with regularization
print("With Regularization constant of 0.01")
final_mat = regularize_fit(features_matrix, output, 0.01)
w0, w1, w2, w3 = final_mat[0], final_mat[1], final_mat[2], final_mat[3]
print("w0 :", w0, ", w1 :", w1, ", w2 :", w2, ", w3 :", w3)
#computing error
exp_output = expected_value(test_features, final_mat)
rss3 = ((test_output-exp_output)*(test_output-exp_output)).sum()
print("RSS error(inverse of accuracy) :", round(rss3))

#with regularization
print("With Regularization constant of 0.001")
final_mat = regularize_fit(features_matrix, output, 0.001)
w0, w1, w2, w3 = final_mat[0], final_mat[1], final_mat[2], final_mat[3]
print("w0 :", w0, ", w1 :", w1, ", w2 :", w2, ", w3 :", w3)
#computing error
exp_output = expected_value(test_features, final_mat)
rss4 = ((test_output-exp_output)*(test_output-exp_output)).sum()
print("RSS error(inverse of accuracy) :", round(rss4))

#histogram
x = numpy.array([0, 0.001, 0.01, 0.1])
y = numpy.array([round(rss1), round(rss4), round(rss3), round(rss2)])
#plotting graph
plt.bar(x, y-round(rss1), color ='darkgreen', width = 0.005)
  
plt.xlabel("lambda")
plt.ylabel("Residual Sum of Squares of errors (as compared to lambda = 0)")
plt.title("RSS v/s Regularization constant")
plt.show()