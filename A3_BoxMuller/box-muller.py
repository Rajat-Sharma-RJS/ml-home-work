import numpy
import matplotlib.pyplot as graph
from math import pi

def boxMuller(sd, itv):
	#random generator
	interval = numpy.random.RandomState(sd)
	#uniform distribution
	unif1, unif2 = interval.uniform(size = itv), interval.uniform(size = itv)
	#taking logarithm
	log = -numpy.log(unif1)
	#computing theta
	radian = 2*pi*unif2
	#taking root
	root = numpy.sqrt(2*log)
	#calculating x and y
	x = root*numpy.cos(radian)
	y = root*numpy.sin(radian)
	return (x, y, unif1, unif2)

#getting values
x, y, unif1, unif2 = boxMuller(65, 875)
#plotting values
plot, axis = graph.subplots(2, 2)
axis[0][0].hist(unif1, color = "darkorange")
axis[0][0].set_title('Uniform Distribution 1')
axis[0][1].hist(unif2, color = "yellowgreen")
axis[0][1].set_title('Uniform Distribution 2')

axis[1][0].hist(x, color = "chocolate")
axis[1][0].set_title('Normal Distribution 1')
axis[1][1].hist(y, color = "magenta")
axis[1][1].set_title('Normal Distribution 2')

plot.set_size_inches(10, 8)
graph.show()