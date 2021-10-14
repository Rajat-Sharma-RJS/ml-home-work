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

def quadratic_fit(features_matrix, output):
	features_transpose = numpy.transpose(features_matrix)
	product_of2 = numpy.matmul(features_transpose, features_matrix)

	inverse_of2 = numpy.linalg.inv(product_of2)
	product_of3 = numpy.matmul(inverse_of2, features_transpose)

	final_mat = numpy.matmul(product_of3, output)
	
	w0, w1, w2 = final_mat[0], final_mat[1], final_mat[2]
	print("w0 :", w0, ", w1 :", w1, ", w2 :", w2)
	return (w0, w1, w2)

def expected_value(inp, w0, w1, w2):
	exp_value = w0 + w1*inp + w2*inp*inp
	return exp_value

df = pandas.read_csv("india_covid19_may.csv")
df['Days'] = 0
for i in range(31):
	df['Days'][i] = i

df['Days_sq'] = df['Days']*df['Days']

train, test = df[df['Date_reported'] <= '2020-05-22'], df[df['Date_reported'] > '2020-05-22']

#training model
features_matrix, train_out = get_np_data(train, ['Days', 'Days_sq'], 'Deaths')
print("after training with 22 days data -----")
w0, w1, w2 = quadratic_fit(features_matrix, train_out)

#testing model
test_out = test['Deaths'].to_numpy()
test_in = test['Days'].to_numpy()

test_predict = expected_value(test_in, w0, w1, w2)
rss = ((test_out - test_predict)*(test_out - test_predict)).sum()
print("after testing the model on 9 days data ----")
print("residual sum of squares of errors(measure of accuracy) :", round(rss))

#plotting prediction on test data
fig,a =  plt.subplots(1,2)
a[0].plot(test_in, test_out, '.', test_in, test_predict, '-')
a[0].set_title('Training data : 22 & testing on : 9')

#taking complete data for training
features_matrix, comp_out = get_np_data(df, ['Days', 'Days_sq'], 'Deaths')
print("Taking complete data for training and printing the new parameters -----")
w0, w1, w2 = quadratic_fit(features_matrix, comp_out)
print("Deaths on April 20, 2020 :", round(expected_value(-11, w0, w1, w2)), " but the actaul is : 36", ", error :", round(36-expected_value(-11, w0, w1, w2)))
print("Deaths on June 10th , 2020 :", round(expected_value(40, w0, w1, w2)), " but the actaul is : 279", ", error :", round(279-expected_value(40, w0, w1, w2)))
inp = df['Days'].to_numpy()
#predicting on training data
a[1].plot(inp, comp_out, '.', inp, expected_value(inp, w0, w1, w2), '-')
a[1].set_title('Training data : 31')
plt.show()
