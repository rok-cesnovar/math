
#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <string>
#include <test/unit/math/prim/prob/vector_rng_test_helper.hpp>
#include <test/unit/math/prim/prob/VectorIntRNGTestRig.hpp>
#include <limits>
#include <vector>

TEST(ProbDistributionsNegBinomial, error_check) {
  using std::log;

  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::neg_binomial_2_log_rng(6, 2, rng));
  EXPECT_NO_THROW(stan::math::neg_binomial_2_log_rng(-0.5, 1, rng));
  EXPECT_NO_THROW(stan::math::neg_binomial_2_log_rng(log(1e8), 1, rng));

  EXPECT_THROW(stan::math::neg_binomial_2_log_rng(0, -2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::neg_binomial_2_log_rng(6, -2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::neg_binomial_2_log_rng(-6, -0.1, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::neg_binomial_2_log_rng(
                   stan::math::positive_infinity(), 2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::neg_binomial_2_log_rng(
                   stan::math::positive_infinity(), 6, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::neg_binomial_2_log_rng(
                   2, stan::math::positive_infinity(), rng),
               std::domain_error);

  std::string error_msg;

  error_msg
      = "neg_binomial_2_log_rng: Exponential "
        "of the log-location parameter divided by the precision "
        "parameter is inf, but must be finite!";
  try {
    stan::math::neg_binomial_2_log_rng(log(1e300), 1e-300, rng);
    FAIL() << "neg_binomial_2_log_rng should have thrown" << std::endl;
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(error_msg) == std::string::npos)
      FAIL() << "Error message is different than expected" << std::endl
             << "EXPECTED: " << error_msg << std::endl
             << "FOUND: " << e.what() << std::endl;
    SUCCEED();
  }

  error_msg
      = "neg_binomial_2_log_rng: Random number that "
        "came from gamma distribution is";
  try {
    stan::math::neg_binomial_2_log_rng(log(1e10), 1e20, rng);
    FAIL() << "neg_binomial_2_log_rng should have thrown" << std::endl;
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(error_msg) == std::string::npos)
      FAIL() << "Error message is different than expected" << std::endl
             << "EXPECTED: " << error_msg << std::endl
             << "FOUND: " << e.what() << std::endl;
    SUCCEED();
  }
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = stan::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<> dist(1.1,
                                                     1.1 / (1.1 + exp(2.4)));
  boost::math::chi_squared mydist(K - 1);

  int loc[K - 1];
  for (int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin[K];
  double expect[K];
  for (int i = 0; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K - 1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::math::neg_binomial_2_log_rng(2.4, 1.1, rng);
    int i = 0;
    while (i < K - 1 && a > loc[i])
      ++i;
    ++bin[i];
    count++;
  }

  double chi = 0;

  for (int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest2) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = stan::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<> dist(0.6,
                                                     0.6 / (0.6 + exp(2.4)));
  boost::math::chi_squared mydist(K - 1);

  int loc[K - 1];
  for (int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin[K];
  double expect[K];
  for (int i = 0; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K - 1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::math::neg_binomial_2_log_rng(2.4, 0.6, rng);
    int i = 0;
    while (i < K - 1 && a > loc[i])
      ++i;
    ++bin[i];
    count++;
  }

  double chi = 0;

  for (int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest3) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = stan::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<> dist(121,
                                                     121 / (121 + exp(2.4)));
  boost::math::chi_squared mydist(K - 1);

  int loc[K - 1];
  for (int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin[K];
  double expect[K];
  for (int i = 0; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K - 1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::math::neg_binomial_2_log_rng(2.4, 121, rng);
    int i = 0;
    while (i < K - 1 && a > loc[i])
      ++i;
    ++bin[i];
    count++;
  }

  double chi = 0;

  for (int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbNegBinomial2, log_matches_lpmf) {
  double y = 0.8;
  double mu = 1.1;
  double phi = 2.3;

  EXPECT_FLOAT_EQ((stan::math::neg_binomial_2_lpmf(y, mu, phi)),
                  (stan::math::neg_binomial_2_log(y, mu, phi)));
  EXPECT_FLOAT_EQ((stan::math::neg_binomial_2_lpmf<true>(y, mu, phi)),
                  (stan::math::neg_binomial_2_log<true>(y, mu, phi)));
  EXPECT_FLOAT_EQ((stan::math::neg_binomial_2_lpmf<false>(y, mu, phi)),
                  (stan::math::neg_binomial_2_log<false>(y, mu, phi)));
  EXPECT_FLOAT_EQ(
      (stan::math::neg_binomial_2_lpmf<true, double, double, double>(y, mu,
                                                                     phi)),
      (stan::math::neg_binomial_2_log<true, double, double, double>(y, mu,
                                                                    phi)));
  EXPECT_FLOAT_EQ(
      (stan::math::neg_binomial_2_lpmf<false, double, double, double>(y, mu,
                                                                      phi)),
      (stan::math::neg_binomial_2_log<false, double, double, double>(y, mu,
                                                                     phi)));
  EXPECT_FLOAT_EQ(
      (stan::math::neg_binomial_2_lpmf<double, double, double>(y, mu, phi)),
      (stan::math::neg_binomial_2_log<double, double, double>(y, mu, phi)));
}

class NegativeBinomial2LogTestRig : public VectorIntRNGTestRig {
 public:
  NegativeBinomial2LogTestRig()
      : VectorIntRNGTestRig(10000, 10, {0, 1, 2, 3, 4, 5, 6},
                            {-0.5, 0.0, 0.1, 1.7}, {-2, 0, 1, 2}, {}, {},
                            {0.1, 1.1, 4.99}, {1, 2, 3}, {-3.0, -2.0, 0.0},
                            {-3, -1, 0}) {}

  template <typename T1, typename T2, typename T3, typename T_rng>
  auto generate_samples(const T1& eta, const T2& phi, const T3&,
                        T_rng& rng) const {
    return stan::math::neg_binomial_2_log_rng(eta, phi, rng);
  }

  template <typename T1>
  double pmf(int y, T1 eta, double phi, double) const {
    return std::exp(stan::math::neg_binomial_2_log_lpmf(y, eta, phi));
  }
};

TEST(ProbDistributionsNegativeBinomial2Log_mat, errorCheck) {
  check_dist_throws_all_types(NegativeBinomial2LogTestRig());
}

TEST(ProbDistributionsNegativeBinomial2Log_mat, distributionCheck) {
  check_counts_real_real(NegativeBinomial2LogTestRig());
}
