import numpy
import matplotlib.pyplot as plt

def simple_linear_regression(input, output):
	avg_input, avg_output = input.mean(), output.mean()
	print("avg_input :", avg_input, ", avg_output :", avg_output)
	sq_input = input*input
	product_of2 = input*output

	avg_sq, avg_product = sq_input.mean(), product_of2.mean()
	print("avg_sq :", avg_sq, ", avg_product :", avg_product)
	slope = (avg_product - (avg_input*avg_output))/(avg_sq - (avg_input*avg_input))
	intercept = avg_output - slope*avg_input
	print("intercept :", intercept, ", slope :", slope)
	return (intercept, slope)

def expected_value(input, intercept, slope):
	exp_value = intercept + slope*input
	return exp_value

output = numpy.array([61.2, 58.3, 67.1, 69.2, 68.9, 83.5, 89.1, 80, 92.3, 93, 97])
input = numpy.array([2004, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017])
intercept, slope = simple_linear_regression(input, output)

exp_out = expected_value(input, intercept, slope)
plt.plot(input, output, '.', input, exp_out, '-')

print("Expected revenue in 2019 is :", expected_value(2019, intercept, slope))
error = output - exp_out
print("error :", error)
rss = error*error
sum = rss.sum()
print("rss :", sum)
plt.show()
