#ifndef STAN_MATH_REV_SCAL_FUNCTOR_REDUCE_SUM_HPP
#define STAN_MATH_REV_SCAL_FUNCTOR_REDUCE_SUM_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/functor.hpp>
#include <stan/math/rev/fun.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

#include <iostream>
#include <iterator>
#include <tuple>
#include <algorithm>
#include <vector>

namespace stan {
namespace math {

// we have to be able to tear down the nested context of a specified
// context
// absolete...luckily
static inline void recover_memory_nested(ChainableStack::AutodiffStackStorage* instance) {
  if (instance->nested_var_stack_sizes_.empty()) {
    throw std::logic_error(
        "empty_nested() must be false"
        " before calling recover_memory_nested()");
  }

  instance->var_stack_.resize(
      instance->nested_var_stack_sizes_.back());
  instance->nested_var_stack_sizes_.pop_back();

  instance->var_nochain_stack_.resize(
      instance->nested_var_nochain_stack_sizes_.back());
  instance->nested_var_nochain_stack_sizes_.pop_back();

  for (size_t i
       = instance->nested_var_alloc_stack_starts_.back();
       i < instance->var_alloc_stack_.size(); ++i) {
    delete instance->var_alloc_stack_[i];
  }
  instance->var_alloc_stack_.resize(
      instance->nested_var_alloc_stack_starts_.back());
  instance->nested_var_alloc_stack_starts_.pop_back();

  instance->memalloc_.recover_nested();
}


namespace internal {

//static int counter = 0;

/**
 * Var specialization of implimentation called by reduce `reduce_sum`.
 * @tparam ReduceFunction An type with a valid `operator()`
 * @tparam ReturnType The return type of the call to
 * `ReduceFunction::operator()`
 * @tparam Vec type of the first input argument with an `operator[]`
 * @tparam Args Parameter pack holding the types of the object send to
 * `ReduceFunction::operator()`
 */
template <typename ReduceFunction, typename ReturnType, typename Vec,
          typename... Args>
struct reduce_sum_impl<ReduceFunction, require_var_t<ReturnType>, ReturnType,
                       Vec, Args...> {

  /**
   * the local context can be used to store var's which are the roots
   * of a reverse mode autodiff call. The local context manages it's
   * own memory for the vari's on the heap. var's can be copied onto
   * it with the deep_copy function.
   */
  struct local_context {
    ChainableStack::AutodiffStackStorage instance_;

    struct local_vari : public vari {
      
      local_vari(ChainableStack::AutodiffStackStorage& stack, double x, bool stacked)
          : vari(stack, x, stacked) { }
      // todo: take care of memory pool / new operator
      //static inline void* operator new(size_t nbytes) {
      //  return ChainableStack::instance_->memalloc_.alloc(nbytes);
      //}
   };

    std::vector<local_vari> vari_store_;
    
    /**
     * Specialization of deep copy that returns references for arithmetic types.
     * @tparam Arith an arithmetic type.
     * @param arg For lvalue references this will be passed by reference.
     *  Otherwise it will be moved.
     */
    template <typename Arith,
              typename = require_arithmetic_t<scalar_type_t<Arith>>>
    inline decltype(auto) deep_copy(Arith&& arg) {
      return std::forward<Arith>(arg);
    }

    /**
     * Specialization to copy a single var
     * @param arg a var.
     */
    inline auto deep_copy(const var& arg) {
      vari_store_.emplace_back(instance_, arg.val(), false);
      return var(&vari_store_.back());
    }

    /**
     * Specialization to copy a standard vector of vars.
     * @tparam VarVec A standard vector holding vars.
     * @param arg A standard vector holding vars.
     */
    template <typename VarVec, require_std_vector_vt<is_var, VarVec>* = nullptr>
     inline auto deep_copy(VarVec&& arg) {
      std::vector<var> copy_vec(arg.size());
      for (size_t i = 0; i < arg.size(); ++i) {
        vari_store_.emplace_back(instance_, arg[i].val(), false);
        copy_vec[i] = &vari_store_.back();
      }
      return copy_vec;
    }

    /**
     * Specialization to copy a standard vector holding containers.
     * @tparam VarVec A standard vector holding containers of vars.
     * @param arg A standard vector holding containers of vars.
     */
    template <typename VarVec, require_std_vector_st<is_var, VarVec>* = nullptr,
              require_std_vector_vt<is_container, VarVec>* = nullptr>
     inline auto deep_copy(VarVec&& arg) {
      std::vector<value_type_t<VarVec>> copy_vec(arg.size());
      for (size_t i = 0; i < arg.size(); ++i) {
        copy_vec[i] = deep_copy(arg[i]);
      }
      return copy_vec;
    }

    /**
     * Specialization to copy an `Eigen` containers of vars.
     * @tparam EigT A type derived from `EigenBase` containing vars.
     * @param arg A container hollding var types.
     */
    template <typename EigT, require_eigen_vt<is_var, EigT>* = nullptr>
     inline auto deep_copy(EigT&& arg) {
      return arg
          .unaryExpr([&](auto&& x) {
                       vari_store_.emplace_back(instance_, x.val(), false);
                       return var(&vari_store_.back());
                     })
          .eval();
    }

  };


    /**
     * Accumulates adjoints for a single var within the reduce.
     * @tparam Pargs types of further arguments to accumulate over.
     * @param dest points to values holding adjoint accumulate.
     * @param x A var to accumulate the adjoint of.
     * @param args Further args to accumulate over.
     */
    template <typename... Pargs>
    static inline double* accumulate_adjoints(double* dest, const var& x,
                                       Pargs&&... args) {
      *dest += x.adj();
      return accumulate_adjoints(dest + 1, std::forward<Pargs>(args)...);
    }

    /**
     * Accumulates adjoints for a standard vector of vars within the reduce.
     * @tparam VarVec the type of a standard container holding vars.
     * @tparam Pargs types of further arguments to accumulate over.
     * @param dest points to values holding adjoint accumulate.
     * @param x A vector of vars to accumulate the adjoint of.
     * @param args Further args to accumulate over.
     */
    template <typename VarVec, require_std_vector_vt<is_var, VarVec>* = nullptr,
              typename... Pargs>
    static inline double* accumulate_adjoints(double* dest, VarVec&& x,
                                       Pargs&&... args) {
      for (auto&& x_iter : x) {
        *dest += x_iter.adj();
        ++dest;
      }
      return accumulate_adjoints(dest, std::forward<Pargs>(args)...);
    }

    /**
     * Accumulates adjoints for a standard vector of containers within the
     * reduce.
     * @tparam VecContainer the type of a standard container holding var
     * containers.
     * @tparam Pargs types of further arguments to accumulate over.
     * @param dest points to values holding adjoint accumulate.
     * @param x A vector of var containers to accumulate the adjoint of.
     * @param args Further args to accumulate over.
     */
    template <typename VecContainer,
              require_std_vector_st<is_var, VecContainer>* = nullptr,
              require_std_vector_vt<is_container, VecContainer>* = nullptr,
              typename... Pargs>
    static inline double* accumulate_adjoints(double* dest, VecContainer&& x,
                                       Pargs&&... args) {
      for (auto&& x_iter : x) {
        dest = accumulate_adjoints(dest, x_iter);
      }
      return accumulate_adjoints(dest, std::forward<Pargs>(args)...);
    }

    /**
     * Accumulates adjoints for an eigen container within the reduce.
     * @tparam EigT Type derived from `EigenBase` containing vars.
     * @tparam Pargs types of further arguments to accumulate over.
     * @param dest points to values holding adjoint accumulate.
     * @param x An eigen type holding vars to accumulate over.
     * @param args Further args to accumulate over.
     */
    template <typename EigT, require_eigen_vt<is_var, EigT>* = nullptr,
              typename... Pargs>
    static inline double* accumulate_adjoints(double* dest, EigT&& x,
                                       Pargs&&... args) {
      Eigen::Map<Eigen::MatrixXd>(dest, x.rows(), x.cols()) += x.adj();
      return accumulate_adjoints(dest + x.size(), std::forward<Pargs>(args)...);
    }

    /**
     * Specialization that is a no-op for Arithmetic types.
     * @tparam Arith A type satisfying `std::is_arithmetic`.
     * @tparam Pargs types of further arguments to accumulate over.
     * @param dest points to values holding adjoint accumulate.
     * @param x An object that is either arithmetic or a container of Arithmetic
     *  types.
     * @param args Further args to accumulate over.
     */
    template <typename Arith,
              require_arithmetic_t<scalar_type_t<Arith>>* = nullptr,
              typename... Pargs>
    static inline double* accumulate_adjoints(double* dest, Arith&& x,
                                       Pargs&&... args) {
      return accumulate_adjoints(dest, std::forward<Pargs>(args)...);
    }

    static inline double* accumulate_adjoints(double* x) { return x; }


  

  /*
    class local_args {
      //const nested_rev_autodiff nested_context_;
      // not sure what the type is of the apply below => this gives a
      // compiler error, but auto type is not allowed
      using args_tuple_copy_t = std::tuple<std::decay_t<Args>...>;
      //int num_instance;
      args_tuple_copy_t args_tuple_copy_;
      ChainableStack::AutodiffStackStorage* instance_;
      bool is_dirty_;

     public:
      template <typename... ArgsT>
      local_args(ArgsT&&... args) :
          //num_instance(counter++),
          instance_(ChainableStack::instance_),
          //args_tuple_copy_(std::tuple<decltype(deep_copy(args))...>(deep_copy(args)...)),
          is_dirty_(false)
      {
        start_nested();
        
        args_tuple_copy_ = std::tuple<decltype(deep_copy(args))...>(deep_copy(args)...);
        //std::cout << "creating shared copy " << num_instance << " in thread " << std::this_thread::get_id() << std::endl;
      }

      args_tuple_copy_t& get_clean_copy() {
        // for now just assume that we can zero out the adjoints
        // correctly with this call... but we have to loop over each
        // var stored in the tuple copy to make this safe
        if(is_dirty_) {
          ChainableStack::AutodiffStackStorage* local_instance = ChainableStack::instance_;
          ChainableStack::instance_ = instance_;
          set_zero_all_adjoints_nested();
          ChainableStack::instance_ = local_instance;
        }
        is_dirty_ = true;
        return args_tuple_copy_;
      }

      ~local_args() {
        //std::cout << "destroying shared copy " << num_instance << " in thread " << std::this_thread::get_id() << std::endl;
        ChainableStack::AutodiffStackStorage* local_instance = ChainableStack::instance_;
        ChainableStack::instance_ = instance_;
        recover_memory_nested();
        ChainableStack::instance_ = local_instance;
      }
    };
  */

  using args_tuple_copy_t = std::tuple<std::decay_t<Args>...>;

  struct local_args {
    local_context context_;
    args_tuple_copy_t args_tuple_copy_;

    template <typename... ArgsT>
    local_args(ArgsT&&... args) :
        context_(),
        args_tuple_copy_(std::tuple<decltype(context_.deep_copy(args))...>(context_.deep_copy(args)...))
    {}
    
  };


  using tls_args_tuple_t = tbb::enumerable_thread_specific< local_args >;
  
  /**
   * Internal object used in `tbb::parallel_reduce`
   * @note see link [here](https://tinyurl.com/vp7xw2t) for requirements.
   */
  struct recursive_reducer {
    size_t per_job_sliced_terms_;
    size_t num_shared_terms_;  // Number of terms shared across threads
    double* sliced_partials_;  // Points to adjoints of the partial calculations
    Vec vmapped_;
    std::ostream* msgs_;
    tls_args_tuple_t& tls_args_tuple_copy_;
    //std::tuple<Args...> args_tuple_;
    double sum_{0.0};
    //Eigen::VectorXd args_adjoints_{0};

    template <typename VecT>
    recursive_reducer(size_t per_job_sliced_terms, size_t num_shared_terms,
                      double* sliced_partials, VecT&& vmapped,
                      std::ostream* msgs, tls_args_tuple_t& tls_args_tuple_copy)
        : per_job_sliced_terms_(per_job_sliced_terms),
          num_shared_terms_(num_shared_terms),
          sliced_partials_(sliced_partials),
          vmapped_(std::forward<VecT>(vmapped)),
          msgs_(msgs),
          tls_args_tuple_copy_(tls_args_tuple_copy)
    {}

    recursive_reducer(recursive_reducer& other, tbb::split)
        : per_job_sliced_terms_(other.per_job_sliced_terms_),
          num_shared_terms_(other.num_shared_terms_),
          sliced_partials_(other.sliced_partials_),
          vmapped_(other.vmapped_),
          msgs_(other.msgs_),
          tls_args_tuple_copy_(other.tls_args_tuple_copy_) {}

    /**
     * Within thread reduction.
     */
    inline void operator()(const tbb::blocked_range<size_t>& r) {
      if (r.empty()) {
        return;
      }

      /*
      if (args_adjoints_.size() == 0) {
        args_adjoints_ = Eigen::VectorXd::Zero(this->num_shared_terms_);
      }
      */

      //local_args<decltype(args_tuple_)> local(args_tuple_);
      //local_args::args_tuple_copy_t& args_tuple_local_copy =
      //tls_args_tuple_.local().args_tuple_copy_;
      //auto& args_tuple_local_copy = tls_args_tuple_.local().get_clean_copy();
      // todo: zero adjoints of args_tuple_local as needed
      //args_tuple_copy_t& local_args_tuple_copy = args_tuple_copy_[tbb::this_task_arena::current_thread_index()];
      args_tuple_copy_t& local_args_tuple_copy = tls_args_tuple_copy_.local().args_tuple_copy_;
    
      const nested_rev_autodiff begin_nest;

      // create a deep copy of all var's so that these are not
      // linked to any outer AD tree

      local_context context;
      
      std::decay_t<Vec> local_sub_slice;
      local_sub_slice.reserve(r.size());
      for (int i = r.begin(); i < r.end(); ++i) {
        local_sub_slice.emplace_back(context.deep_copy(vmapped_[i]));
      }
      
      /*
      auto args_tuple_local_copy = apply(
          [&](auto&&... args) {
            return std::tuple<decltype(deep_copy(args))...>(deep_copy(args)...);
          },
          this->args_tuple_);
      */
      
      var sub_sum_v = apply(
          [&](auto&&... args) {
            return ReduceFunction()(r.begin(), r.end() - 1, local_sub_slice,
                                    this->msgs_, args...);
          },
          local_args_tuple_copy);
      //local.args_tuple_copy_);

      sub_sum_v.grad();
      sum_ += sub_sum_v.val();
      accumulate_adjoints(
          this->sliced_partials_ + r.begin() * per_job_sliced_terms_,
          local_sub_slice);

      /* we just keep accumulating the adjoints in the args_tuple_copy
      and collect from there at the very end
      apply(
          [&](auto&&... args) {
            return accumulate_adjoints(args_adjoints_.data(),
                                       std::forward<decltype(args)>(args)...);
          },
          std::forward<decltype(tls_args_tuple_.local().get_clean_copy())>(args_tuple_local_copy));
      */
      // SW: Why did we use a move here? I hope forward is also fine.
      //std::move(local.args_tuple_copy_));
      //std::move(args_tuple_local_copy));
    }

    /**
     * Join the sum of the reducers
     * @param child A sub-reducer that is aggregated to make the final reduced
     * value.
     * @note side effect of updating `sum_` and
     *  `args_adjoints_`
     */
    void join(const recursive_reducer& rhs) {
      this->sum_ += rhs.sum_;
      /*
      if (this->args_adjoints_.size() != 0 && rhs.args_adjoints_.size() != 0) {
        this->args_adjoints_ += rhs.args_adjoints_;
      } else if (this->args_adjoints_.size() == 0
                 && rhs.args_adjoints_.size() != 0) {
        this->args_adjoints_ = rhs.args_adjoints_;
      }
      */
    }
  };

  /**
   * Count the number of vars in an vector.
   * @tparam Vec type of standard container holding vars.
   * @tparam Pargs Types to be forwarded to recursive call of `count_var_impl`.
   * @param[in] count The current count of the number of vars.
   * @param[in] x A vector holding vars.
   * @param[in] args objects to be forwarded to recursive call of
   * `count_var_impl`
   */
  template <typename VecVar, require_std_vector_vt<is_var, VecVar>* = nullptr,
            typename... Pargs>
  inline size_t count_var_impl(size_t count, VecVar&& x,
                               Pargs&&... args) const {
    return count_var_impl(count + x.size(), std::forward<Pargs>(args)...);
  }

  /**
   * Count the number of vars in an vector holding containers.
   * @tparam A standard vector holding containers of vars.
   * @tparam Pargs Types to be forwarded to recursive call of `count_var_impl`.
   * @param[in] count The current count of the number of vars.
   * @param[in] x A vector holding containers of vars.
   * @param[in] args objects to be forwarded to recursive call of
   * `count_var_impl`
   */
  template <typename VecContainer,
            require_std_vector_st<is_var, VecContainer>* = nullptr,
            require_std_vector_vt<is_container, VecContainer>* = nullptr,
            typename... Pargs>
  inline size_t count_var_impl(size_t count, VecContainer&& x,
                               Pargs&&... args) const {
    for (auto&& x_iter : x) {
      count = count_var_impl(count, x_iter);
    }
    return count_var_impl(count, std::forward<Pargs>(args)...);
  }
  /**
   * Count the number of vars in an eigen container.
   * @tparam EigT A type derived from `EigenBase`.
   * @tparam Pargs Types to be forwarded to recursive call of `count_var_impl`.
   * @param[in] count The current count of the number of vars.
   * @param[in] x An Eigen container holding vars.
   * @param[in] args objects to be forwarded to recursive call of
   * `count_var_impl`
   */
  template <typename EigT, require_eigen_vt<is_var, EigT>* = nullptr,
            typename... Pargs>
  inline size_t count_var_impl(size_t count, EigT&& x, Pargs&&... args) const {
    return count_var_impl(count + x.size(), std::forward<Pargs>(args)...);
  }

  /**
   * Add a var to the count.
   * @tparam Pargs Types to be forwarded to recursive call of `count_var_impl`.
   * @param[in] count The current count of the number of vars.
   * @param[in] x A var.
   * @param[in] args objects to be forwarded to recursive call of
   * `count_var_impl`
   */
  template <typename... Pargs>
  inline size_t count_var_impl(size_t count, const var& x,
                               Pargs&&... args) const {
    return count_var_impl(count + 1, std::forward<Pargs>(args)...);
  }

  /**
   * Specialization that performs skips count for arithmetic types and
   * containers.
   * @tparam Arith An object that is either arithmetic or holds arithmetic
   * types.
   * @tparam Pargs Types to be forwarded to recursive call of `count_var_impl`.
   * @param[in] count The current count of the number of vars.
   * @param[in] x An arithmetic value or container.
   * @param[in] args objects to be forwarded to recursive call of
   * `count_var_impl`
   */
  template <typename Arith,
            require_arithmetic_t<scalar_type_t<Arith>>* = nullptr,
            typename... Pargs>
  inline size_t count_var_impl(size_t count, Arith& x, Pargs&&... args) const {
    return count_var_impl(count, std::forward<Pargs>(args)...);
  }

  /**
   * Specialization to end recursion
   */
  inline size_t count_var_impl(size_t count) const { return count; }

  /**
   * Count the number of scalars of type T in the input argument list
   *
   * @tparam Pargs Types of input arguments
   * @return Number of scalars of type T in input
   */
  template <typename... Pargs>
  inline size_t count_var(Pargs&&... args) const {
    return count_var_impl(0, std::forward<Pargs>(args)...);
  }

  /**
   * Store a reference to `vari` used in `reduce_sum`.
   * @tparam Pargs Types to be forwarded to recursive call of `save_varis`.
   * @param[in, out] dest Pointer to set of varis.
   * @param[in] x A `var` whose `vari` will be stores
   * @param[in] args Further arguments forwarded to recursive call.
   */
  template <typename... Pargs>
  inline vari** save_varis(vari** dest, const var& x, Pargs&&... args) const {
    *dest = x.vi_;
    return save_varis(dest + 1, std::forward<Pargs>(args)...);
  }

  /**
   * Store a reference to an std vector's `vari` used in `reduce_sum`.
   * @tparam VarVec Type of standard vector holding vars.
   * @tparam Pargs Types to be forwarded to recursive call of `save_varis`.
   * @param[in, out] dest Pointer to set of varis.
   * @param[in] x A standard vector whose vari's are stored.
   * @param[in] args Further arguments forwarded to recursive call.
   */
  template <typename VarVec, require_std_vector_vt<is_var, VarVec>* = nullptr,
            typename... Pargs>
  inline vari** save_varis(vari** dest, VarVec&& x, Pargs&&... args) const {
    using write_map = Eigen::Map<Eigen::Matrix<vari*, -1, 1>>;
    using read_map = Eigen::Map<const Eigen::Matrix<var, -1, 1>>;
    write_map(dest, x.size(), 1) = read_map(x.data(), x.size(), 1).vi();
    return save_varis(dest + x.size(), std::forward<Pargs>(args)...);
  }

  /**
   * Store a reference to an std vector of containers's `vari` used in
   * `reduce_sum`.
   * @tparam VecContainer Type of standard vector holding containers of vars.
   * @tparam Pargs Types to be forwarded to recursive call of `save_varis`.
   * @param[in, out] dest Pointer to set of varis.
   * @param[in] x A standard vector whose container's varis are stored.
   * @param[in] args Further arguments forwarded to recursive call.
   */
  template <typename VecContainer,
            require_std_vector_st<is_var, VecContainer>* = nullptr,
            require_std_vector_vt<is_container, VecContainer>* = nullptr,
            typename... Pargs>
  inline vari** save_varis(vari** dest, VecContainer&& x,
                           Pargs&&... args) const {
    for (size_t i = 0; i < x.size(); ++i) {
      dest = save_varis(dest, x[i]);
    }
    return save_varis(dest, std::forward<Pargs>(args)...);
  }

  /**
   * Store a reference to an Eigen types `vari` used in `reduce_sum`.
   * @tparam VarVec Type derived from `EigenBase` holding vars.
   * @tparam Pargs Types to be forwarded to recursive call of `save_varis`.
   * @param[in, out] dest Pointer to set of varis.
   * @param[in] x An `EigenBase` whose vari's are stored.
   * @param[in] args Further arguments forwarded to recursive call.
   */
  template <typename EigT, require_eigen_vt<is_var, EigT>* = nullptr,
            typename... Pargs>
  inline vari** save_varis(vari** dest, EigT&& x, Pargs&&... args) const {
    using mat_t = std::decay_t<EigT>;
    using write_map = Eigen::Map<Eigen::Matrix<vari*, mat_t::RowsAtCompileTime,
                                               mat_t::ColsAtCompileTime>>;
    write_map(dest, x.rows(), x.cols()) = x.vi();
    return save_varis(dest + x.size(), std::forward<Pargs>(args)...);
  }

  /**
   * Specialization to skip over arithmetic types and containers of arithmetic
   * types.
   * @tparam Arith either an arithmetic type or container holding arithmetic
   * types.
   * @tparam Pargs Types to be forwarded to recursive call of `save_varis`.
   * @param[in, out] dest Pointer to set of varis.
   * @param[in] x arithmetic or container of arithmetics.
   * @param[in] args Further arguments forwarded to recursive call.
   */
  template <typename Arith,
            require_arithmetic_t<scalar_type_t<Arith>>* = nullptr,
            typename... Pargs>
  inline vari** save_varis(vari** dest, Arith&& x, Pargs&&... args) const {
    return save_varis(dest, std::forward<Pargs>(args)...);
  }
  /**
   * Ends the recursion by returning the input `vari**`
   * @param dest a pointer to pointers to varis.
   * @return the `dest` vari* pointer
   */
  inline vari** save_varis(vari** dest) const { return dest; }

  template <typename OpVec, typename... OpArgs,
            typename = require_same_t<Vec, OpVec>,
            typename = require_all_t<std::is_same<Args, OpArgs>...>>
  inline var operator()(OpVec&& vmapped, std::size_t grainsize,
                        std::ostream* msgs, OpArgs&&... args) const {

    const std::size_t num_jobs = vmapped.size();

    if (num_jobs == 0) {
      return var(0.0);
    }

    const std::size_t per_job_sliced_terms = count_var(vmapped[0]);
    const std::size_t num_sliced_terms = num_jobs * per_job_sliced_terms;
    const std::size_t num_shared_terms = count_var(args...);

    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
        num_sliced_terms + num_shared_terms);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
        num_sliced_terms + num_shared_terms);

    for (size_t i = 0; i < (num_sliced_terms + num_shared_terms); ++i) {
      partials[i] = 0.0;
    }

    double sum = 0.0;

    {
      tls_args_tuple_t tls_args_tuple_copy(std::forward<OpArgs>(args)...);

      /*
      const std::size_t num_worker_threads = tbb::this_task_arena::max_concurrency();
      std::vector<args_tuple_copy_t> args_tuple_copy;
      args_tuple_copy.reserve(num_worker_threads);

      for(std::size_t i = 0; i != num_worker_threads; ++i)
        args_tuple_copy.emplace_back(context.deep_copy(args)...);
      */
      
      recursive_reducer worker(per_job_sliced_terms, num_shared_terms, partials,
                               vmapped, msgs, tls_args_tuple_copy);

#ifdef STAN_DETERMINISTIC
      tbb::inline_partitioner partitioner;
      tbb::parallel_deterministic_reduce(
          tbb::blocked_range<std::size_t>(0, num_jobs, grainsize), worker,
          partitioner);
#else
      tbb::parallel_reduce(
          tbb::blocked_range<std::size_t>(0, num_jobs, grainsize), worker);
#endif

      save_varis(varis, std::forward<OpVec>(vmapped));
      save_varis(varis + num_sliced_terms, std::forward<OpArgs>(args)...);

      /*
      for(std::size_t i = 0; i != num_worker_threads; ++i) {
        apply(
          [&](auto&&... args) {
            return accumulate_adjoints(partials + num_sliced_terms,
                                       std::forward<decltype(args)>(args)...);
          },
          std::move(args_tuple_copy[i]));
      }
      */
      tls_args_tuple_copy.combine_each(
          [&](local_args& args_context) {
            apply(
                [&](auto&&... args) {
                  return accumulate_adjoints(partials + num_sliced_terms,
                                             std::forward<OpArgs>(args)...);
                },
                std::move(args_context.args_tuple_copy_));
          });
  
      sum = worker.sum_;
    }
      
    return var(new precomputed_gradients_vari(
        sum, num_sliced_terms + num_shared_terms, varis, partials));
  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif
