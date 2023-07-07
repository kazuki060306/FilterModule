#include "signaltools.h"

// constructor
signaltools::signaltools()
	: m_arraytools(NULL)
	, m_sigtools(NULL)
{
	m_arraytools = new arraytools();
	m_sigtools = new sigtools();
}

// destructor
signaltools::~signaltools()
{
	delete m_arraytools;
	delete m_sigtools;
}

/*
 Apply a digital filter forward and backward to a signal.
 
 This function applies a linear digital filter twice, once forward and
 once backwards.The combined filter has zero phase and a filter order
 twice that of the original.
 
 The function provides options for handling the edges of the signal.
 
 The function `sosfiltfilt` (and filter design using ``output='sos'``)
 should be preferred over `filtfilt` for most filtering tasks, as
 second - order sections have fewer numerical problems.
 
 Parameters
 ----------
 b : (N, ) array_like
 The numerator coefficient vector of the filter.
 a : (N, ) array_like
 The denominator coefficient vector of the filter.If ``a[0]``
 is not 1, then both `a` and `b` are normalized by ``a[0]``.
 x : array_like
 The array of data to be filtered.
 axis : int, optional
 The axis of `x` to which the filter is applied.
 Default is - 1.
 padtype : str or None, optional
 Must be 'odd', 'even', 'constant', or None.This determines the
 type of extension to use for the padded signal to which the filter
 is applied.If `padtype` is None, no padding is used.The default
 is 'odd'.
 padlen : int or None, optional
 The number of elements by which to extend `x` at both ends of
 `axis` before applying the filter.This value must be less than
 ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
 The default value is ``3 * max(len(a), len(b))``.
 method : str, optional
 Determines the method for handling the edges of the signal, either
 "pad" or "gust".When `method` is "pad", the signal is padded; the
 type of padding is determined by `padtype` and `padlen`, and `irlen`
 is ignored.When `method` is "gust", Gustafsson's method is used,
 and `padtype` and `padlen` are ignored.
 irlen : int or None, optional
 When `method` is "gust", `irlen` specifies the length of the
 impulse response of the filter.If `irlen` is None, no part
 of the impulse response is ignored.For a long signal, specifying
 `irlen` can significantly improve the performance of the filter.
 
 Returns
 ------ -
 y : ndarray
 The filtered output with the same shape as `x`.
 
 See Also
 --------
 sosfiltfilt, lfilter_zi, lfilter, lfiltic, savgol_filter, sosfilt
 
 Notes
 ---- -
 When `method` is "pad", the function pads the data along the given axis
 in one of three ways : odd, even or constant.The odd and even extensions
 have the corresponding symmetry about the end point of the data.The
 constant extension extends the data with the values at the end points.On
 both the forward and backward passes, the initial condition of the
 filter is found by using `lfilter_zi` and scaling it by the end point of
 the extended data.
 
 When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
 conditions are chosen for the forward and backward passes so that the
 forward - backward filter gives the same result as the backward - forward
 filter.
 
 The option to use Gustaffson's method was added in scipy version 0.16.0.
 
 References
 ----------
 ..[1] F.Gustaffson, "Determining the initial states in forward-backward
 filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
 1996.
 
 Examples
 --------
 The examples will use several functions from `scipy.signal`.
 
 >> > import numpy as np
 >> > from scipy import signal
 >> > import matplotlib.pyplot as plt
 
 First we create a one second signal that is the sum of two pure sine
 waves, with frequencies 5 Hz and 250 Hz, sampled at 2000 Hz.
 
 >> > t = np.linspace(0, 1.0, 2001)
 >> > xlow = np.sin(2 * np.pi * 5 * t)
 >> > xhigh = np.sin(2 * np.pi * 250 * t)
 >> > x = xlow + xhigh
 
 Now create a lowpass Butterworth filter with a cutoff of 0.125 times
 the Nyquist frequency, or 125 Hz, and apply it to ``x`` with `filtfilt`.
 The result should be approximately ``xlow``, with no phase shift.
 
 >> > b, a = signal.butter(8, 0.125)
 >> > y = signal.filtfilt(b, a, x, padlen = 150)
 >> > np.abs(y - xlow).max()
 9.1086182074789912e-06
 
 We get a fairly clean result for this artificial example because
 the odd extension is exact, and with the moderately long padding,
 the filter's transients have dissipated by the time the actual data
 is reached.In general, transient effects at the edges are
 unavoidable.
 
 The following example demonstrates the option ``method="gust"``.
 
 First, create a filter.
 
 >> > b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
 
 `sig` is a random input signal to be filtered.
 
 >> > rng = np.random.default_rng()
 >> > n = 60
 >> > sig = rng.standard_normal(n) * *3 + 3 * rng.standard_normal(n).cumsum()
 
 Apply `filtfilt` to `sig`, once using the Gustafsson method, and
 once using padding, and plot the results for comparison.
 
 >> > fgust = signal.filtfilt(b, a, sig, method = "gust")
 >> > fpad = signal.filtfilt(b, a, sig, padlen = 50)
 >> > plt.plot(sig, 'k-', label = 'input')
 >> > plt.plot(fgust, 'b-', linewidth = 4, label = 'gust')
 >> > plt.plot(fpad, 'c-', linewidth = 1.5, label = 'pad')
 >> > plt.legend(loc = 'best')
 >> > plt.show()
 
 The `irlen` argument can be used to improve the performance
 of Gustafsson's method.
 
 Estimate the impulse response length of the filter.
 
 >> > z, p, k = signal.tf2zpk(b, a)
 >> > eps = 1e-9
 >> > r = np.max(np.abs(p))
 >> > approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
 >> > approx_impulse_len
 137
 
 Apply the filter to a longer signal, with and without the `irlen`
 argument.The difference between `y1` and `y2` is small.For long
 signals, using `irlen` gives a significant performance improvement.
 
 >> > x = rng.standard_normal(4000)
 >> > y1 = signal.filtfilt(b, a, x, method = 'gust')
 >> > y2 = signal.filtfilt(b, a, x, method = 'gust', irlen = approx_impulse_len)
 >> > print(np.max(np.abs(y1 - y2)))
 2.875334415008979e-10
 */
Eigen::VectorXd& signaltools::filtfilt(
	const Eigen::VectorXd& b,
	const Eigen::VectorXd& a,
	const Eigen::VectorXd& x,
	int axis,
	PADTYPE padtype,
	int padlen,
	METHOD method,
	int irlen)
{
	// return value
	Eigen::VectorXd y;

	if (method != pad || method != gust)
	{
		return y;
	}

	if (method == gust)
	{
		// unsupported case
		//y = _filtfilt_gust(b, a, x, axis = axis, irlen = irlen);
	}
	// method == pad
	else
	{

		// local valiable
		int edge = 0;
		Eigen::VectorXd ext;
		Eigen::VectorXd zi;
		double x0 = 0.0;
		double y0 = 0.0;
		Eigen::VectorXd zi_x0;
		Eigen::VectorXd zi_y0;
		size_t len_zi = 0;

		edge = _validate_pad(padtype, padlen, x, axis, std::max(a.size(), b.size()), ext);

		// Get the steady state of the filter's step response.
		zi = lfilter_zi(b, a);

		// Reshape zi and create x0 so that zi * x0 broadcasts
		// to the correct value for the 'zi' keyword argument
		// to lfilter.
		//zi_shape = [1] * x.ndim
		//zi_shape[axis] = zi.size
		//zi = np.reshape(zi, zi_shape)
		x0 = m_arraytools->axis_slice<double>(ext, NONE, 1, NONE, axis);

		len_zi = zi.size();

		// Forward filter.
		zi_x0.setZero(len_zi);
		for (size_t i =0; i < zi.size(); ++i)
		{
			zi_x0(i) = zi(i) * x0;
		}
		y = lfilter(b, a, ext, axis, zi_x0);

		// Backward filter.
		// Create y0 so zi * y0 broadcasts appropriately.
		y0 = m_arraytools->axis_slice<double>(y, -1, NONE, NONE, axis = axis);
		for (size_t i = 0; i < zi.size(); ++i)
		{
			zi_y0[i] = zi[i] * y0;
		}
		y = lfilter(b, a, m_arraytools->axis_reverse(y, axis), axis, zi_y0);

		// Reverse y.
		y = m_arraytools->axis_reverse(y, axis);

		if (edge > 0)
		{
			// Slice the actual signal from the extended signal.
			y = m_arraytools->axis_slice<Eigen::VectorXd&>(y, edge, -edge, NONE, axis);
		}
	}

	return y;
}

/*
 Helper to validate padding for filtfilt
 */
int signaltools::_validate_pad(
	PADTYPE padtype,
	int padlen,
	const Eigen::VectorXd& x,
	int axis,
	int ntaps,
	Eigen::VectorXd& ext)
{
	// return value
	int edge = 0;

	if (padtype != even || padtype != odd || padtype != constant)
	{
		return edge;
	}
	//if (padlen == None)
	//{
	//	# Original padding; preserved for backwards compatibility.
	//	edge = ntaps * 3;
	//}
 //   else
	//{
	//	edge = padlen
	//}

	// x's 'axis' dimension must be bigger than edge.
	//if (x.shape[axis] <= edge)
	//{
	//	return edge;
	//}

	if (edge > 0)
	{
		// Make an extension of length `edge` at each
		// end of the input array.
		if (padtype == even)
		{
			// unsupported case
			///ext = even_ext(x, edge, axis);
		}
		else if (padtype == odd)
		{
			ext = m_arraytools->odd_ext(x, edge, axis);
		}
		else
		{
			// unsupported case
			//ext = const_ext(x, edge, axis);
		}
	}
	else
	{
		ext = x;
	}

	return edge;
}

/*
 Construct initial conditions for lfilter for step response steady-state.
 
 Compute an initial state `zi` for the `lfilter` function that corresponds
 to the steady state of the step response.
 
 A typical use of this function is to set the initial state so that the
 output of the filter starts at the same value as the first element of
 the signal to be filtered.
 
 Parameters
 ----------
 b, a : array_like (1-D)
 	The IIR filter coefficients. See `lfilter` for more
 	information.
 
 Returns
 -------
 zi : 1-D ndarray
 	The initial state for the filter.
 
 See Also
 --------
 lfilter, lfiltic, filtfilt
 
 Notes
 -----
 A linear filter with order m has a state space representation (A, B, C, D),
 for which the output y of the filter can be expressed as::
 
 	z(n+1) = A*z(n) + B*x(n)
 	y(n)   = C*z(n) + D*x(n)
 
 where z(n) is a vector of length m, A has shape (m, m), B has shape
 (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
 a scalar).  lfilter_zi solves::
 
 	zi = A*zi + B
 
 In other words, it finds the initial condition for which the response
 to an input of all ones is a constant.
 
 Given the filter coefficients `a` and `b`, the state space matrices
 for the transposed direct form II implementation of the linear filter,
 which is the implementation used by scipy.signal.lfilter, are::
 
 	A = scipy.linalg.companion(a).T
 	B = b[1:] - a[1:]*b[0]
 
 assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
 divided by a[0].
 
 Examples
 --------
 The following code creates a lowpass Butterworth filter. Then it
 applies that filter to an array whose values are all 1.0; the
 output is also all 1.0, as expected for a lowpass filter.  If the
 `zi` argument of `lfilter` had not been given, the output would have
 shown the transient signal.
 
 >>> from numpy import array, ones
 >>> from scipy.signal import lfilter, lfilter_zi, butter
 >>> b, a = butter(5, 0.25)
 >>> zi = lfilter_zi(b, a)
 >>> y, zo = lfilter(b, a, ones(10), zi=zi)
 >>> y
 array([1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
 
 Another example:
 
 >>> x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
 >>> y, zf = lfilter(b, a, x, zi=zi*x[0])
 >>> y
 array([ 0.5       ,  0.5       ,  0.5       ,  0.49836039,  0.48610528,
 	0.44399389,  0.35505241])
 
 Note that the `zi` argument to `lfilter` was computed using
 `lfilter_zi` and scaled by `x[0]`.  Then the output `y` has no
 transient until the input drops from 0.5 to 0.0.
 */
Eigen::VectorXd& signaltools::lfilter_zi(
	Eigen::VectorXd b,
	Eigen::VectorXd a)
{
	// FIXME: Can this function be replaced with an appropriate
	// use of lfiltic ? For example, when b, a = butter(N, Wn),
	//    lfiltic(b, a, y = numpy.ones_like(a), x = numpy.ones_like(b)).
	//
	//
	// We could use scipy.signal.normalize, but it uses warnings in
	// cases where a ValueError is more appropriate, and it allows
	// b to be 2D.

	// retrun value
	Eigen::VectorXd zi;

	//b = np.atleast_1d(b)
	//if (b.ndim != 1)
	//{
	//		raise ValueError("Numerator b must be 1-D.")
	//}
	//a = np.atleast_1d(a)
	//f (a.ndim != 1)
	//{
	//		raise ValueError("Denominator a must be 1-D.")
	//}
	
	size_t len_a = a.size();
	while (len_a > 1 && a(0) == 0.0)
	{
		for (size_t i = 1; i < len_a; ++i)
		{
			Eigen::VectorXd _a = a(Eigen::seq(1, len_a - 1));
			a.resize(len_a - 1);
			a = _a;
		}
	}
	if (a.size() < 1)
	{
		// There must be at least one nonzero `a` coefficient.
		return zi;
	}

	// local variable
	int n = 0;
	Eigen::MatrixXd IminusA;
	Eigen::VectorXd B;

	if (a(0) != 1.0)
	{
		// Normalize the coefficients so a(0) == 1.
		for (size_t i = 1; i < b.size(); ++i)
		{
			b(i) /= a(0);
		}
		for (size_t i = 1; i < a.size(); ++i)
		{
			a(i) /= a(0);
		}
	}

	n = std::max(a.size(), b.size());

	// Pad a or b with zeros so they are the same length.
	if (a.size() < n)
	{
		Eigen::VectorXd zero_vec;
		zero_vec.resize(n - a.size());
		a << zero_vec;
	}
	else if (b.size() < n)
	{
		Eigen::VectorXd zero_vec;
		zero_vec.resize(n - b.size());
		b << zero_vec;
	}

	Eigen::internal::companion<double, Eigen::Dynamic> companion(a);
	IminusA = Eigen::MatrixXd::Identity(n - 1, n - 1) - companion.denseMatrix().transpose();
	B = b(Eigen::seq(1, b.size() - 1)) - a(Eigen::seq(1, a.size() - 1)) *b(0);
	// Solve zi = A*zi + B
	zi = IminusA.fullPivLu().solve(B);

	// For future reference : we could also use the following
	// explicit formulas to solve the linear system :
	//
	// zi = np.zeros(n - 1)
	// zi[0] = B.sum() / IminusA[:, 0].sum()
	// asum = 1.0
	// csum = 0.0
	// for k in range(1, n - 1) :
	//     asum += a[k]
	//     csum += b[k] - a[k] * b[0]
	//     zi[k] = asum * zi[0] - csum

	return zi;
}

/*
 Filter data along one-dimension with an IIR or FIR filter.
 
 Filter a data sequence, `x`, using a digital filter.  This works for many
 fundamental data types (including Object type).  The filter is a direct
 form II transposed implementation of the standard difference equation
 (see Notes).
 
 The function `sosfilt` (and filter design using ``output='sos'``) should be
 preferred over `lfilter` for most filtering tasks, as second-order sections
 have fewer numerical problems.
 
 Parameters
 ----------
 b : array_like
 	The numerator coefficient vector in a 1-D sequence.
 a : array_like
 	The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
 	is not 1, then both `a` and `b` are normalized by ``a[0]``.
 x : array_like
 	An N-dimensional input array.
 axis : int, optional
 	The axis of the input data array along which to apply the
 	linear filter. The filter is applied to each subarray along
 	this axis.  Default is -1.
 zi : array_like, optional
 	Initial conditions for the filter delays.  It is a vector
 	(or array of vectors for an N-dimensional input) of length
 	``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
 	initial rest is assumed.  See `lfiltic` for more information.
 
 Returns
 -------
 y : array
 	The output of the digital filter.
 zf : array, optional
 	If `zi` is None, this is not returned, otherwise, `zf` holds the
 	final filter delay values.
 
 See Also
 --------
 lfiltic : Construct initial conditions for `lfilter`.
 lfilter_zi : Compute initial state (steady state of step response) for
 			 `lfilter`.
 filtfilt : A forward-backward filter, to obtain a filter with zero phase.
 savgol_filter : A Savitzky-Golay filter.
 sosfilt: Filter data using cascaded second-order sections.
 sosfiltfilt: A forward-backward filter using second-order sections.
 
 Notes
 -----
 The filter function is implemented as a direct II transposed structure.
 This means that the filter implements::
 
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
 						 - a[1]*y[n-1] - ... - a[N]*y[n-N]
 
 where `M` is the degree of the numerator, `N` is the degree of the
 denominator, and `n` is the sample number.  It is implemented using
 the following difference equations (assuming M = N)::
 
 	 a[0]*y[n] = b[0] * x[n]               + d[0][n-1]
 	   d[0][n] = b[1] * x[n] - a[1] * y[n] + d[1][n-1]
 	   d[1][n] = b[2] * x[n] - a[2] * y[n] + d[2][n-1]
 	 ...
 	 d[N-2][n] = b[N-1]*x[n] - a[N-1]*y[n] + d[N-1][n-1]
 	 d[N-1][n] = b[N] * x[n] - a[N] * y[n]
 
 where `d` are the state variables.
 
 The rational transfer function describing this filter in the
 z-transform domain is::
 
 						 -1              -M
 			 b[0] + b[1]z  + ... + b[M] z
 	 Y(z) = -------------------------------- X(z)
 						 -1              -N
 			 a[0] + a[1]z  + ... + a[N] z
 
 Examples
 --------
 Generate a noisy signal to be filtered:
 
 >>> import numpy as np
 >>> from scipy import signal
 >>> import matplotlib.pyplot as plt
 >>> rng = np.random.default_rng()
 >>> t = np.linspace(-1, 1, 201)
 >>> x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
 ...      0.1*np.sin(2*np.pi*1.25*t + 1) +
 ...      0.18*np.cos(2*np.pi*3.85*t))
 >>> xn = x + rng.standard_normal(len(t)) * 0.08
 
 Create an order 3 lowpass butterworth filter:
 
 >>> b, a = signal.butter(3, 0.05)
 
 Apply the filter to xn.  Use lfilter_zi to choose the initial condition of
 the filter:
 
 >>> zi = signal.lfilter_zi(b, a)
 >>> z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
 
 Apply the filter again, to have a result filtered at an order the same as
 filtfilt:
 
 >>> z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
 
 Use filtfilt to apply the filter:
 
 >>> y = signal.filtfilt(b, a, xn)
 
 Plot the original signal and the various filtered versions:
 
 >>> plt.figure
 >>> plt.plot(t, xn, 'b', alpha=0.75)
 >>> plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
 >>> plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
 ...             'filtfilt'), loc='best')
 >>> plt.grid(True)
 >>> plt.show()
 */
Eigen::VectorXd& signaltools::lfilter(
	const Eigen::VectorXd& b,
	const Eigen::VectorXd& a,
	const Eigen::VectorXd& x,
	int axis,
	const Eigen::VectorXd& zi)
{
	// return value
	Eigen::VectorXd y;

	// a = np.atleast_1d(a)

	if (a.size() == 1)
	{
		// unsupported case
		return y;
	}
	else
	{
		if (zi.size() == 0)
		{
			y = m_sigtools->linear_filter(b, a, x, axis, zi);
		}
		else
		{
			y = m_sigtools->linear_filter(b, a, x, axis, zi);
		}
	}

	return y;
}