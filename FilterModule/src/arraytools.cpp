#include "arraytools.h"

// constructor
arraytools::arraytools()
{
	// nop
}

// destructor
arraytools::~arraytools()
{
	// nop
}

/*
 Take a slice along axis 'axis' from 'a'.
 
 Parameters
 ----------
 a : numpy.ndarray
     The array to be sliced.
 start, stop, step : int or None
     The slice parameters.
 axis : int, optional
     The axis of `a` to be sliced.
 
 Examples
 --------
 >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 >>> axis_slice(a, start=0, stop=1, axis=1)
 array([[1],
        [4],
        [7]])
 >>> axis_slice(a, start=1, axis=0)
 array([[4, 5, 6],
        [7, 8, 9]])
 
 Notes
 -----
 The keyword arguments start, stop and step are used by calling
 slice(start, stop, step). This implies axis_slice() does not
 handle its arguments the exactly the same as indexing. To select
 a single index k, for example, use
     axis_slice(a, start=k, stop=k+1)
 In this case, the length of the axis 'axis' in the result will
 be 1; the trivial dimension is not removed. (Use numpy.squeeze()
 to remove trivial axes.)
 */
template<>
double arraytools::axis_slice<double>(
	const Eigen::VectorXd& a,
	int start,
	int stop,
	int step,
	int axis)
{
	// return value
	double ret = 0.0;

	return ret;
}

/*
 Take a slice along axis 'axis' from 'a'.

 Parameters
 ----------
 a : numpy.ndarray
	 The array to be sliced.
 start, stop, step : int or None
	 The slice parameters.
 axis : int, optional
	 The axis of `a` to be sliced.

 Examples
 --------
 >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 >>> axis_slice(a, start=0, stop=1, axis=1)
 array([[1],
		[4],
		[7]])
 >>> axis_slice(a, start=1, axis=0)
 array([[4, 5, 6],
		[7, 8, 9]])

 Notes
 -----
 The keyword arguments start, stop and step are used by calling
 slice(start, stop, step). This implies axis_slice() does not
 handle its arguments the exactly the same as indexing. To select
 a single index k, for example, use
	 axis_slice(a, start=k, stop=k+1)
 In this case, the length of the axis 'axis' in the result will
 be 1; the trivial dimension is not removed. (Use numpy.squeeze()
 to remove trivial axes.)
 */
template<>
Eigen::VectorXd& arraytools::axis_slice<Eigen::VectorXd&>(
	const Eigen::VectorXd& a,
	int start,
	int stop,
	int step,
	int axis)
{
	// return value
	Eigen::VectorXd ret;

	return ret;
}

/*
 Reverse the 1-D slices of `a` along axis `axis`.
 Returns axis_slice(a, step=-1, axis=axis).
 */
Eigen::VectorXd& arraytools::axis_reverse(
	const Eigen::VectorXd& a,
	int axis)
{
	// return value
	Eigen::VectorXd ret;

	return ret;
}

/*
 Odd extension at the boundaries of an array
 
 Generate a new ndarray by making an odd extension of `x` along an axis.
 
 Parameters
 ----------
 x : ndarray
     The array to be extended.
 n : int
     The number of elements by which to extend `x` at each end of the axis.
 axis : int, optional
     The axis along which to extend `x`. Default is -1.
 
 Examples
 --------
 >>> from scipy.signal._arraytools import odd_ext
 >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
 >>> odd_ext(a, 2)
 array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],
        [-4, -1,  0,  1,  4,  9, 16, 23, 28]])
 
 Odd extension is a "180 degree rotation" at the endpoints of the original
 array:
 
 >>> t = np.linspace(0, 1.5, 100)
 >>> a = 0.9 * np.sin(2 * np.pi * t**2)
 >>> b = odd_ext(a, 40)
 >>> import matplotlib.pyplot as plt
 >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='odd extension')
 >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
 >>> plt.legend(loc='best')
 >>> plt.show()
 */
Eigen::VectorXd& arraytools::odd_ext(
	const Eigen::VectorXd& x,
	int n,
	int axis)
{
	// return value
	Eigen::VectorXd ret;

	return ret;
}
