#ifndef SIGTOOLS_H
#define SIGTOOLS_H

#include <Eigen/Core>

class sigtools
{
public:

	// constructor
	sigtools();

	// destructor
	~sigtools();

	Eigen::VectorXd& linear_filter(
		const Eigen::VectorXd& b,
		const Eigen::VectorXd& a,
		const Eigen::VectorXd& x,
		int axis,
		const Eigen::VectorXd& zi);
};
#endif	// SIGTOOLS_H
