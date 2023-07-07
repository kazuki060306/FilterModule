#ifndef ARRATOOLS_H
#define ARRATOOLS_H

#include <Eigen/Core>
#include <cmath>

#define NONE NAN

/*
 Functions for acting on a axis of an array.
 */
class arraytools
{
public:

	// constructor
	arraytools();

	// destructor
	~arraytools();

	template<typename T>
	T axis_slice(
		const Eigen::VectorXd& a,
		int start,
		int stop,
		int step,
		int axis);

	Eigen::VectorXd& axis_reverse(
		const Eigen::VectorXd& a,
		int axis);

	Eigen::VectorXd& odd_ext(
		const Eigen::VectorXd& x,
		int n,
		int axis);
};
#endif	// ARRATOOLS_H
