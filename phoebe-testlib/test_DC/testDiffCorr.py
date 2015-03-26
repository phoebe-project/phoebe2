from phoebe.algorithms import _DiffCorr
import numpy as np

#Take the next line out to get non-repeated random numbers
np.random.seed(1111)



class System:
	def __init__(self,x,y,z):
		self.x = x
		self.y = y
		self.z = z
	def area(self):
		return self.x * self.y
	def perimeter(self):
		return 2 * self.x + 2 * self.y




def function(x, system):
	t = np.arange(0.0,1.0,0.100,dtype=np.float64)
	value = np.arange(len(t),dtype=np.float64)
	value = 3*x[0]*x[2]*(t**4) + 3*x[0]*x[1]*(t**2) - 4*x[1]*x[2]*(t**3)+ 3*x[2]*t
	return value


def d_function_dx(x, system):
	t = np.arange(0.0,1.0,0.100,np.float64)
	value = np.arange(len(t),dtype=np.float64)
	value = 3*x[2]*(t**4) + 3*x[1]*(t**2)
	return value

def d_function_dy(x, system):
	t = np.arange(0.0,1.0,0.100,np.float64)
	value = np.arange(len(t),dtype=np.float64)
	value = 3*x[0]*(t**2) - 4*x[2]*(t**3)
	return value



def d_function_dz(x, system):
	t = np.arange(0.0,1.0,0.100,np.float64)
	value = np.arange(len(t),dtype=np.float64)
	value = 3*x[0]*(t**4) - 4*x[1]*(t**3)+ 3*t
	return value




def main():

	#cheap way of faking enum
	#derivative type
	NUMERICAL = 0
	ANALYTICAL = 1
	NONE = 2

	# stopping criteria type
	MIN_DX = 0
	MIN_DELTA_DX = 1
	MIN_CHI2 = 2
	MIN_DELTA_CHI2 = 3

	# system is to mimic passing a python class to the lightcurve function
	system = System(3.000,3.000,4.000)
	nParams = 3;
	nDataPoints = 10;
	stoppingCriteriaType = MIN_DX
	#stopValue = 0.0001
	stopValue = 0.00001
	maxIterations = 30

	# /* initial guess for starting point */

	initialGuess = np.array([8.000, 4.000, 4.000],dtype=np.float64)
	# to test derivative = NONE must have that parameter at correct value...
	#initialGuess = np.array([8.000, 4.000, 7.000])

	derivativeType = np.array([ANALYTICAL, ANALYTICAL, ANALYTICAL], dtype=np.int32)
	#derivativeType = np.array([NUMERICAL, NUMERICAL, NUMERICAL], dtype=np.int32)
	#derivativeType = np.array([NUMERICAL, NUMERICAL, NONE])

	"""
	/*diffCorr->derivativeType[0] = NUMERICAL;
	diffCorr->derivativeType[1] = NUMERICAL;
	diffCorr->derivativeType[2] = NUMERICAL;*/
	"""

	"""
	/*diffCorr->derivativeType[0] = NUMERICAL;
	diffCorr->derivativeType[1] = ANALYTICAL;
	diffCorr->derivativeType[2] = NUMERICAL;*/
	"""

	"""
	/* To test None the initial guess must be correct for these values */
	/*diffCorr->derivativeType[0] = NONE;
	diffCorr->derivativeType[1] = NONE;
	diffCorr->derivativeType[2] = NUMERICAL;*/
	"""



	# /* This is for comparison to see how algorithum did and to make simulated data */
	solution = np.array([8.00, 5.00, 7.00],dtype=np.float64)




	# fill the simulated data
	data = np.empty([nDataPoints],dtype=np.float64)
	
	data = function(solution, system) + \
			0.0200*np.random.random_sample()


	print("data = ", data)
	

	# set the derivative functions as a list
	dFunc_dx = [d_function_dx, d_function_dy, d_function_dz]
	lFunc = function

	#/* Call DiffCorr to obtain the solution */
	error = 0;
	parameterSolution = np.empty([nParams],dtype=np.float64)
	print("About to call... derrivativeType = ",derivativeType)
	parameterSolution = _DiffCorr.DiffCorr(nParams,  \
				nDataPoints, \
				maxIterations, \
				stoppingCriteriaType, \
				stopValue, \
				data, \
				initialGuess, \
				derivativeType, \
				dFunc_dx, \
				lFunc, \
				system)


	print("I am back")
	print("  Solution = ", parameterSolution)

	sigma2 = 0
	print("\n\n\n For Comparision....\n")
	print("data = ",data)
	print("function = ",function(parameterSolution, system))
	diff = data - function(parameterSolution, system)
	print("diff = ",diff)
	sigma2 = np.dot(diff,diff)

	print("sigma2 = ",sigma2 )
	print("sigma = ", np.sqrt(np.dot(sigma2,sigma2)/(1.0*(nDataPoints-1.0))))




if __name__ == '__main__':
    main()
