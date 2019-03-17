#ifndef STAN_MATH_PRIM_FUN_REP_ROW_VECTOR_HPP
#define STAN_MATH_PRIM_FUN_REP_ROW_VECTOR_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/err/check_nonnegative.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

template <typename T>
inline Eigen::Matrix<typename boost::math::tools::promote_args<T>::type, 1,
                     Eigen::Dynamic>
rep_row_vector(const T& x, int m) {
  check_nonnegative("rep_row_vector", "m", m);
  return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type, 1,
                       Eigen::Dynamic>::Constant(m, x);
}

}  // namespace math
}  // namespace stan

#endif
