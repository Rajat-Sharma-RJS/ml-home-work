import pandas
import numpy
import matplotlib.pyplot as plt

def get_np_data(data_frame, output):
	out = data_frame[output]
	out_arr = out.to_numpy()
	return out_arr

def linear_fit(inp, output):
	avg_input, avg_output = inp.mean(), output.mean()
	sq_input = inp*inp
	product_of2 = inp*output

	avg_sq, avg_product = sq_input.mean(), product_of2.mean()
	slope = (avg_product - (avg_input*avg_output))/(avg_sq - (avg_input*avg_input))
	intercept = avg_output - slope*avg_input
	print("intercept :", intercept, ", slope :", slope)
	return (intercept, slope)

def expected_value(inp, intercept, slope):
	exp_value = intercept + slope*inp
	return exp_value

df = pandas.read_csv("india_covid19_may.csv")

train, test = df[df['Date_reported'] <= '2020-05-22'], df[df['Date_reported'] > '2020-05-22']
train_days, test_days = numpy.zeros(22), numpy.zeros(9)

for i in range(22):
	train_days[i] = i
for i in range(9):
	test_days[i] = i+22

#training model
train_out = get_np_data(train, 'Deaths')
print("after training with 22 days data -----")
intercept, slope = linear_fit(train_days, train_out)

#testing model
test_out = get_np_data(test, 'Deaths')
test_predict = expected_value(test_days, intercept, slope)
rss = ((test_out - test_predict)*(test_out - test_predict)).sum()
print("after testing the model on 9 days data ----")
print("residual sum of squares of errors(measure of accuracy) :", round(rss))

#plotting prediction on test data
fig,a =  plt.subplots(1,2)
a[0].plot(test_days, test_out, '.', test_days, test_predict, '-')
a[0].set_title('Training data : 22 & testing on : 9')

#taking complete data for training
comp_out = get_np_data(df, 'Deaths')
comp_days = numpy.zeros(31)
for i in range(31):
	comp_days[i] = i
print("Taking complete data for training and printing the new parameters -----")
intercept, slope = linear_fit(comp_days, comp_out)
print("Deaths on April 20, 2020 :", round(expected_value(-11, intercept, slope)), " but the actaul is : 36", ", error :", round((36-expected_value(-11, intercept, slope))))
print("Deaths on June 10th , 2020 :", round(expected_value(40, intercept, slope)), " but the actaul is : 279", ", error :", round((279-expected_value(40, intercept, slope))))
#predicting on training data
a[1].plot(comp_days, comp_out, '.', comp_days, expected_value(comp_days, intercept, slope), '-')
a[1].set_title('Training data : 31')

plt.show()
