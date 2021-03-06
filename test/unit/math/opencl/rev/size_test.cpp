#ifdef STAN_OPENCL
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(MathMatrixCL, size_rev) {
  using stan::math::matrix_cl;
  using stan::math::size;
  using stan::math::to_matrix_cl;
  using stan::math::var_value;
  Eigen::Matrix<double, -1, -1> m0(0, 0);
  var_value<matrix_cl<double>> m0_cl = to_matrix_cl(m0);
  EXPECT_EQ(0, size(m0_cl));

  Eigen::Matrix<double, -1, -1> m1(0, 5);
  var_value<matrix_cl<double>> m1_cl = to_matrix_cl(m1);
  EXPECT_EQ(0, size(m1_cl));

  Eigen::Matrix<double, -1, -1> m2(5, 0);
  var_value<matrix_cl<double>> m2_cl = to_matrix_cl(m2);
  EXPECT_EQ(0, size(m2_cl));

  Eigen::Matrix<double, -1, -1> m3(5, 4);
  var_value<matrix_cl<double>> m3_cl = to_matrix_cl(m3);
  EXPECT_EQ(20, size(m3_cl));

  Eigen::Matrix<double, -1, -1> m4(3, 2);
  var_value<matrix_cl<double>> m4_cl = to_matrix_cl(m4);
  EXPECT_EQ(6, size(m4_cl));
}

#endif
