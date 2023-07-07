#include "sigtools.h"

// constructor
sigtools::sigtools()
{
	// nop
}

// destructor
sigtools::~sigtools()
{
	// nop
}

Eigen::VectorXd& sigtools::linear_filter(
	const Eigen::VectorXd& b,
	const Eigen::VectorXd& a,
	const Eigen::VectorXd& x,
	int axis,
	const Eigen::VectorXd& zi)
{
	// return value
	Eigen::VectorXd ret;

	return ret;
}
