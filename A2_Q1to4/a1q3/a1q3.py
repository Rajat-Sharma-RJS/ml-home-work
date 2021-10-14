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

output = numpy.array([61.2, 49.5, 37.5, 28.4, 19.2, 10.1])
inp = numpy.array([54.3, 61.8, 72.4, 88.7, 118.6, 194])
input = 1.0/inp
intercept, slope = simple_linear_regression(input, output)

exp_out = expected_value(input, intercept, slope)
plt.plot(input, output, '.', input, exp_out, '-')

print("Expected P when V=100 is :", expected_value(0.01, intercept, slope))
error = output - exp_out
print("error :", error)
rss = error*error
sum = rss.sum()
print("rss :", sum)
plt.show()
