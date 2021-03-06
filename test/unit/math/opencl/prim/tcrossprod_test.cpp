#ifdef STAN_OPENCL
#include <stan/math/opencl/prim.hpp>
#include <stan/math/prim/fun/tcrossprod.hpp>
#include <gtest/gtest.h>

void test_tcrossprod(const stan::math::matrix_d& x) {
  using stan::math::tcrossprod;
  stan::math::matrix_cl<double> x_cl(x);
  stan::math::matrix_d y_cl = from_matrix_cl(tcrossprod(x_cl));
  stan::math::matrix_d y_correct = tcrossprod(x);
  EXPECT_EQ(y_correct.rows(), y_cl.rows());
  EXPECT_EQ(y_correct.cols(), y_cl.cols());
  for (int m = 0; m < y_correct.rows(); ++m)
    for (int n = 0; n < y_correct.cols(); ++n)
      EXPECT_FLOAT_EQ(y_cl(m, n), y_correct(m, n));
}

TEST(MathMatrixCL, tcrossprod) {
  stan::math::matrix_d x;
  test_tcrossprod(x);

  x = stan::math::matrix_d(1, 1);
  x << 3.0;
  test_tcrossprod(x);

  x = stan::math::matrix_d(2, 2);
  x << 1.0, 0.0, 2.0, 3.0;
  test_tcrossprod(x);

  x = stan::math::matrix_d(3, 3);
  x << 1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0;
  test_tcrossprod(x);

  x = Eigen::MatrixXd::Random(100, 100);
  test_tcrossprod(x);
}
#endif
