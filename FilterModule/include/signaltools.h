#ifndef SIGNALTOOLS_H
#define SIGNALTOOLS_H

#include <Eigen/Core>
#include <Eigen/Polynomials>
#include "arraytools.h"
#include "sigtools.h"

class signaltools
{
public:

private:

	enum PADTYPE
	{
		odd,
		even,
		constant,
	};

	enum METHOD
	{
		pad,
		gust,
	};

	arraytools	*m_arraytools;
	sigtools	*m_sigtools;

public:

	Eigen::VectorXd& filtfilt(
		const Eigen::VectorXd& b,
		const Eigen::VectorXd& a,
		const Eigen::VectorXd& x,
		int axis = -1,
		PADTYPE padtype = odd,
		int padlen = -1,
		METHOD method = pad,
		int irlen = -1);

private:

	// constructor
	signaltools();

	// destructor
	~signaltools();

	int _validate_pad(
		PADTYPE padtype,
		int padlen,
		const Eigen::VectorXd& x,
		int axis,
		int ntaps,
		Eigen::VectorXd& ext);

	Eigen::VectorXd& lfilter_zi(
		Eigen::VectorXd b,
		Eigen::VectorXd a);

	Eigen::VectorXd& lfilter(
		const Eigen::VectorXd& b,
		const Eigen::VectorXd& a,
		const Eigen::VectorXd& x,
		int axis,
		const Eigen::VectorXd& zi);
};
#endif	// SIGNALTOOLS_H
