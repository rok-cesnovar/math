#include <test/unit/math/test_ad.hpp>
#include <limits>

namespace lb_constrain_test {
template <typename T1, typename T2>
void expect_matvar(const T1& x, const T2& lb) {
  auto f1 = [](const auto& x, const auto& lb) {
    return stan::math::lb_constrain(x, lb);
  };
  auto f2 = [](const auto& x, const auto& lb) {
    stan::return_type_t<decltype(x), decltype(lb)> lp = 0;
    return stan::math::lb_constrain(x, lb, lp);
  };
  auto f3 = [](const auto& x, const auto& lb) {
    stan::return_type_t<decltype(x), decltype(lb)> lp = 0;
    stan::math::lb_constrain(x, lb, lp);
    return lp;
  };
  auto f4 = [](const auto& x, const auto& lb) {
    stan::return_type_t<decltype(x), decltype(lb)> lp = 0;
    auto xx = stan::math::lb_constrain(x, lb, lp);
    return stan::math::add(lp, stan::math::sum(xx));
  };

  stan::test::expect_ad_matvar(f1, x, lb);
  stan::test::expect_ad_matvar(f2, x, lb);
  stan::test::expect_ad_matvar(f3, x, lb);
  stan::test::expect_ad_matvar(f4, x, lb);
}
}  // namespace lb_constrain_test

TEST(mathMixMatFun, lb_matvar_constrain) {
  using stan::scalar_type_t;
  using stan::math::lb_constrain;
  using stan::math::promote_scalar_t;
  Eigen::MatrixXd A(2, 2);
  A << 5.0, 2.0, 0.0, 0.005;
  Eigen::MatrixXd lbm(2, 2);
  lbm << 7.0, 5.0, 0.0, 0.0005;
  lb_constrain_test::expect_matvar(A, lbm);
  double lbd = 6.0;
  lb_constrain_test::expect_matvar(A, lbd);
}

TEST(mathMixMatFun, lb_matvar_constrain_neg_inf) {
  Eigen::MatrixXd A(2, 2);
  A << 5.0, 2.0, 4.0, -2.0;
  Eigen::MatrixXd lbm(2, 2);
  lbm << 7.0, 5.0, stan::math::NEGATIVE_INFTY, 100.0;
  lb_constrain_test::expect_matvar(A, lbm);
  lb_constrain_test::expect_matvar(A, stan::math::NEGATIVE_INFTY);
}
