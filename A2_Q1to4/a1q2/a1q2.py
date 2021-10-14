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

ml = numpy.array([75, 80, 93, 65, 87, 71, 98, 68, 84, 77])
hur = numpy.array([82, 78, 86, 72, 91, 80, 95, 72, 89, 74])
print("Taking ML as independent variable ---")
intercept, slope = simple_linear_regression(ml, hur)

exp_out = expected_value(ml, intercept, slope)
plt.plot(ml, hur, '.', ml, exp_out, '-')

print("Expected HUR for 96 in ML is :", expected_value(96, intercept, slope))
error = hur - exp_out
print("error :", error)
rss = error*error
sum = rss.sum()
print("rss :", sum)
plt.show()

print("Taking HUR as independent variable ---")
intercept, slope = simple_linear_regression(hur, ml)

exp_out = expected_value(hur, intercept, slope)
plt.plot(hur, ml, '.', hur, exp_out, '-')

print("Expected ML for 95 in HUR is :", expected_value(95, intercept, slope))
error = ml - exp_out
print("error :", error)
rss = error*error
sum = rss.sum()
print("rss :", sum)
plt.show()
