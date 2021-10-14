import numpy
import matplotlib.pyplot as plt

def simple_linear_regression(input, output):
	avg_input, avg_output = input.mean(), output.mean()
	sq_input = input*input
	avg_sq = sq_input.mean()
	print("A(y): ", avg_output, ", A(x): ", avg_input, ", A(x^2): ", avg_sq)
	#computing p
	product_of2 = input*output
	avg_product = product_of2.mean()
	p = avg_product - avg_output*avg_input
	#computing q
	q = avg_sq - avg_input*avg_input
	#computing r
	cube_input = input*input*input
	avg_cube = cube_input.mean()
	r = avg_cube - avg_sq*avg_input
	#computing s
	product_of3 = output*input*input
	avg_of3 = product_of3.mean()
	s = avg_of3 - avg_output*avg_sq
	#computing t
	quad_input = input*input*input*input
	avg_quad = quad_input.mean()
	t = avg_quad - avg_sq*avg_sq
	#computing w2
	w2 = (q*s - r*p)/(q*t - r*r)
	#computing w1
	w1 = (p - w2*r)/q
	#computing w0
	w0 = avg_output - w1*avg_input - w2*avg_sq
	print("p: ",p,", q: ",q,", r: ",r,", s: ",s,", t: ",t)
	print("w0: ",w0,", w1: ",w1,", w2: ",w2)
	return (w0, w1, w2)

def expected_value(input, w0, w1, w2):
	exp_value = w0 + w1*input + w2*input*input
	return exp_value

output = numpy.array([2.4, 2.1, 3.2, 5.6, 9.3, 14.6, 21.9])
input = numpy.array([0, 1, 2, 3, 4, 5, 6])
w0, w1, w2 = simple_linear_regression(input, output)

exp_out = expected_value(input, w0, w1, w2)
plt.plot(input, output, '.', input, exp_out, '-')

error = output - exp_out
print("error :", error)
rss = error*error
sum = rss.sum()
print("rss :", sum)
plt.show()
