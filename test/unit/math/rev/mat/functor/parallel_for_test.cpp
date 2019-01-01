#include <stan/math/rev/mat.hpp>
#include <stan/math/parallel/for_each.hpp>
#include <stan/math/parallel/get_num_threads.hpp>
#include <stan/math/rev/core/nest_chainablestack.hpp>
#include <gtest/gtest.h>

#include <tbb/task_scheduler_init.h>

#include <boost/iterator/counting_iterator.hpp>

#include <stdexcept>
#include <vector>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using stan::math::value_of;
using stan::math::vector_d;
using stan::math::vector_v;
using std::vector;

// fun1(x, y) = (x^2 * y) + (3 * y^2)
struct fun1 {
  template <typename T>
  inline T operator()(const Matrix<T, Dynamic, 1>& x) const {
    return x(0) * x(0) * x(1) + 3.0 * x(1) * x(1);
  }
};

// test if base case is ok (move me)

static tbb::task_scheduler_init task_scheduler(
    stan::math::internal::get_num_threads());

TEST(Base, parallel_for) {
  // const int num_threads = stan::math::internal::get_num_threads();
  // tbb::task_scheduler_init task_scheduler(num_threads);
  const int num_jobs = 1000;
  typedef boost::counting_iterator<int> count_iter;

  vector_d x_ref_2(num_jobs);

  for (int i = 0; i < num_jobs; ++i) {
    x_ref_2(i) = 7 + i;
  }

  auto loop_fun = [&](int i) -> vector_d {
    fun1 f;
    vector_d res(1);
    vector_d iarg(2);
    iarg << 5, x_ref_2(i);
    res(0) = f(iarg);
    return res;
  };

  vector_d parallel_result = stan::math::parallel_for_each(
      count_iter(0), count_iter(num_jobs), loop_fun);

  for (int i = 0; i < num_jobs; ++i) {
    vector_d x_ref(2);
    x_ref << 5, x_ref_2(i);
    double fx_ref = parallel_result(i);
    EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) * x_ref(1) + 3 * x_ref(1) * x_ref(1),
                    fx_ref);
  }

  stan::math::recover_memory_global();
}

TEST(AgradAutoDiff, parallel_for) {
  // const int num_threads = stan::math::internal::get_num_threads();
  // tbb::task_scheduler_init task_scheduler(num_threads);
  typedef boost::counting_iterator<int> count_iter;

  const int num_jobs = 1000;
  fun1 f;

  stan::math::var fixed_arg = 5;

  vector_v x_ref_2(num_jobs);

  for (int i = 0; i < num_jobs; ++i) {
    x_ref_2(i) = 7 + i;
  }

  auto loop_fun = [&](int i) -> vector_v {
    vector_v res(1);
    vector_v iarg(2);
    iarg << fixed_arg, x_ref_2(i);
    res(0) = f(iarg);
    return res;
  };

  vector_v parallel_result = stan::math::parallel_for_each(
      count_iter(0), count_iter(num_jobs), loop_fun);

  for (int i = 0; i < num_jobs; ++i) {
    vector_d x_ref(2);
    x_ref << 5, value_of(x_ref_2(i));
    double fx_ref = value_of(parallel_result(i));
    EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) * x_ref(1) + 3 * x_ref(1) * x_ref(1),
                    fx_ref);
    vector<double> grad_fx_ref(2);
    stan::math::set_zero_all_adjoints_global();
    // stan::math::set_zero_all_adjoints();
    stan::math::grad(parallel_result(i).vi_);
    grad_fx_ref[0] = fixed_arg.adj();
    grad_fx_ref[1] = x_ref_2(i).adj();

    EXPECT_EQ(2, grad_fx_ref.size());
    EXPECT_FLOAT_EQ(2 * x_ref(0) * x_ref(1), grad_fx_ref[0]);
    EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) + 3 * 2 * x_ref(1), grad_fx_ref[1]);
  }

  stan::math::recover_memory_global();
}

TEST(AgradAutoDiff, parallel_for_each_scalar) {
  // const int num_threads = stan::math::internal::get_num_threads();
  // tbb::task_scheduler_init task_scheduler(num_threads);
  typedef boost::counting_iterator<int> count_iter;

  const int num_jobs = 1000;
  fun1 f;

  stan::math::var fixed_arg = 5;

  vector_v x_ref_2(num_jobs);

  for (int i = 0; i < num_jobs; ++i) {
    x_ref_2(i) = 7 + i;
  }

  auto loop_fun = [&](int i) -> stan::math::var {
    vector_v iarg(2);
    iarg << fixed_arg, x_ref_2(i);
    return f(iarg);
  };

  vector_v parallel_result = stan::math::parallel_for_each(
      count_iter(0), count_iter(num_jobs), loop_fun);

  for (int i = 0; i < num_jobs; ++i) {
    vector_d x_ref(2);
    x_ref << 5, value_of(x_ref_2(i));
    double fx_ref = value_of(parallel_result(i));
    EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) * x_ref(1) + 3 * x_ref(1) * x_ref(1),
                    fx_ref);
    vector<double> grad_fx_ref(2);
    stan::math::set_zero_all_adjoints_global();
    // stan::math::set_zero_all_adjoints();
    stan::math::grad(parallel_result(i).vi_);
    grad_fx_ref[0] = fixed_arg.adj();
    grad_fx_ref[1] = x_ref_2(i).adj();

    EXPECT_EQ(2, grad_fx_ref.size());
    EXPECT_FLOAT_EQ(2 * x_ref(0) * x_ref(1), grad_fx_ref[0]);
    EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) + 3 * 2 * x_ref(1), grad_fx_ref[1]);
  }

  stan::math::recover_memory_global();
}
/**/

// test threaded AD if enabled
/*
TEST(AgradAutoDiff, parallel_for_each) {
  using stan::math::ChainableStack;
  using stan::math::var;
  typedef boost::counting_iterator<int> count_iter;
  using std::size_t;

  const int num_threads = stan::math::internal::get_num_threads();
  tbb::task_scheduler_init task_scheduler(num_threads);

  fun1 f;
  const int num_jobs = 1000;

  vector<vector_v> x_ref_v(num_jobs);
  vector<vector_d> x_ref_d(num_jobs);

  for (int i = 0; i < num_jobs; ++i) {
    vector_v xi(2);
    xi << 5, 7 + i;
    x_ref_v[i] = xi;
    x_ref_d[i] = value_of(xi);
  }

  vector<var> fres(num_jobs);

  typedef ChainableStack::AutodiffStackStorage local_ad_stack_t;
  typedef std::vector<stan::math::vari*>::reverse_iterator rev_it_t;

  vector<bool> stack_is_local(num_jobs, false);
  vector<local_ad_stack_t*> stack_used(num_jobs, nullptr);
  vector<size_t> stack_starts(num_jobs);
  vector<size_t> stack_ends(num_jobs);

  std::thread::id parent_thread_id = std::this_thread::get_id();

  auto apply_f = [&](int i) -> void {
                   stack_is_local[i] = parent_thread_id ==
std::this_thread::get_id();
                   //local_ad_stack_t* local_instance = stack_is_local[i]
                   //                                   ? nullptr
                   //                                   :
&ChainableStack::instance(); local_ad_stack_t& local_instance =
ChainableStack::instance(); stack_starts[i] = local_instance.var_stack_.size();
                   fres[i] = f(x_ref_v[i]);
                   stack_ends[i] = local_instance.var_stack_.size();
                   stack_used[i] = &local_instance;
                   std::cout << "job i = " << i << "; stack size = " <<
stack_ends[i] - stack_starts[i] << std::endl;
                 };

#ifdef STAN_THREADS
  constexpr std_par::execution::parallel_unsequenced_policy exec_policy
      = std_par::execution::par_unseq;
#else
  constexpr std_par::execution::sequenced_policy exec_policy
      = std_par::execution::seq;
#endif

  std::cout << "Firing off parallel jobs" << std::endl;
  std_par::for_each(exec_policy, count_iter(0), count_iter(num_jobs), apply_f);
*/
/*
local_ad_stack_t& ad_tape = ChainableStack::instance();
std::for_each(ChainableStack::instance_.begin(),
              ChainableStack::instance_.end(),
              [&](local_ad_stack_t& local_instance) {
                if (&local_instance == &ad_tape) {
                  std::cout << "Found main tape!" << std::endl;
                  return;
                }
                ad_tape.var_stack_.insert(ad_tape.var_stack_.end(),
                                          local_instance.var_stack_.begin(),
                                          local_instance.var_stack_.end());
                local_instance.var_stack_.clear();
                //it_t begin = local_instance.var_stack_.rbegin();
                //it_t end = local_instance.var_stack_.rend();
                //for (it_t it = begin; it != end; ++it) {
                //  (*it)->chain();
                //}
              });
*/
/*
// now register all the AD tape pieces on the per-thread instances
// in the main AD tape
for (int i = 0; i < num_jobs; ++i) {
  if(!stack_is_local[i]) {
    std::cout << "non-local job i = " << i << "; stack size = " << stack_ends[i]
- stack_starts[i] << std::endl;
    stan::math::register_nested_chainablestack(*stack_used[i], stack_starts[i],
stack_ends[i]);
  }
}

for (int i = 0; i < num_jobs; ++i) {
  vector_d x_ref = x_ref_d[i];
  double fx_ref = value_of(fres[i]);
  vector<double> grad_fx_ref(2);
  //stan::math::set_zero_all_adjoints_global();
  //stan::math::set_zero_all_adjoints();
  //stan::math::grad_global(fres[i].vi_);
  stan::math::set_zero_all_adjoints();
  stan::math::grad(fres[i].vi_);
  grad_fx_ref[0] = x_ref_v[i](0).adj();
  grad_fx_ref[1] = x_ref_v[i](1).adj();

  EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) * x_ref(1) + 3 * x_ref(1) * x_ref(1),
                  fx_ref);
  EXPECT_EQ(2, grad_fx_ref.size());
  EXPECT_FLOAT_EQ(2 * x_ref(0) * x_ref(1), grad_fx_ref[0]);
  EXPECT_FLOAT_EQ(x_ref(0) * x_ref(0) + 3 * 2 * x_ref(1), grad_fx_ref[1]);
}
}

*/
