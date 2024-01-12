//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_GinkgoSolver_DEF_HPP
#define _FROSCH_GinkgoSolver_DEF_HPP

#include <FROSch_GinkgoSolverTpetra_decl.hpp>

namespace FROSch {

using namespace Stratimikos;
using namespace Teuchos;
using namespace Thyra;
using namespace Xpetra;

namespace detail {

/**
 * Helper to check if an executor type can access the memory of an memory space
 *
 * @tparam MemorySpace  Type fulfilling the Kokkos MemorySpace concept.
 * @tparam ExecType  One of the Ginkgo executor types.
 */
template <typename MemorySpace, typename ExecType>
struct compatible_space
    : std::integral_constant<bool, Kokkos::has_shared_space ||
                                       Kokkos::has_shared_host_pinned_space> {};

template <>
struct compatible_space<Kokkos::HostSpace, gko::ReferenceExecutor>
    : std::true_type {};

template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::ReferenceExecutor> {
  // need manual implementation of std::integral_constant because,
  // while compiling for cuda, somehow bool is replaced by __nv_bool
  static constexpr bool value =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible;
};

#ifdef KOKKOS_ENABLE_OPENMP
template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::OmpExecutor>
    : compatible_space<MemorySpace, gko::ReferenceExecutor> {};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::CudaExecutor> {
  static constexpr bool value =
      Kokkos::SpaceAccessibility<Kokkos::Cuda, MemorySpace>::accessible;
};
#endif

#ifdef KOKKOS_ENABLE_HIP
template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::HipExecutor> {
  static constexpr bool value =
      Kokkos::SpaceAccessibility<Kokkos::HIP, MemorySpace>::accessible;
};
#endif

#ifdef KOKKOS_ENABLE_SYCL
template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::DpcppExecutor> {
  static constexpr bool value =
      Kokkos::SpaceAccessibility<Kokkos::Experimental::SYCL,
                                 MemorySpace>::accessible;
};
#endif

/**
 * Checks if the memory space is accessible by the executor
 *
 * @tparam MemorySpace  A Kokkos memory space type
 * @tparam ExecType  A Ginkgo executor type
 *
 * @return  true if the memory space is accessible by the executor
 */
template <typename MemorySpace, typename ExecType>
inline bool check_compatibility(MemorySpace, std::shared_ptr<const ExecType>) {
  return compatible_space<MemorySpace, ExecType>::value;
}

} // namespace detail

/**
 * Creates an Executor matching the Kokkos::DefaultHostExecutionSpace.
 *
 * If no kokkos host execution space is enabled, this throws an exception.
 *
 * @return  An executor of type either ReferenceExecutor or OmpExecutor.
 */
inline std::shared_ptr<gko::Executor> create_default_host_executor() {
  static std::mutex mutex{};
  std::lock_guard<std::mutex> guard(mutex);
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::Serial>) {
    static auto exec = gko::ReferenceExecutor::create();
    return exec;
  }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::OpenMP>) {
    static auto exec = gko::OmpExecutor::create();
    return exec;
  }
#endif
  GKO_NOT_IMPLEMENTED;
}

/**
 * Creates an Executor for a specific Kokkos ExecutionSpace.
 *
 * This function supports the following Kokkos ExecutionSpaces:
 * - Serial
 * - OpenMP
 * - Cuda
 * - HIP
 * - Experimental::SYCL
 * If none of these spaces are enabled, then this function throws an exception.
 * For Cuda, HIP, SYCL, the device-id used by Kokkos is passed to the Executor
 * constructor.
 *
 * @tparam ExecSpace  A supported Kokkos ExecutionSpace.
 *
 * @return  An executor matching the type of the ExecSpace.
 */
template <typename ExecSpace,
          typename MemorySpace = typename ExecSpace::memory_space>
inline std::shared_ptr<gko::Executor> create_executor(ExecSpace ex,
                                                      MemorySpace = {}) {
  static_assert(Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible);
  static std::mutex mutex{};
  std::lock_guard<std::mutex> guard(mutex);
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<ExecSpace, Kokkos::Serial>) {
    return gko::ReferenceExecutor::create();
  }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
    return gko::OmpExecutor::create();
  }
#endif
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
    if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
      return gko::CudaExecutor::create(
          Kokkos::device_id(), create_default_host_executor(),
          std::make_shared<gko::CudaAllocator>(), ex.cuda_stream());
    }
    if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
      return gko::CudaExecutor::create(
          Kokkos::device_id(), create_default_host_executor(),
          std::make_shared<gko::CudaUnifiedAllocator>(Kokkos::device_id()),
          ex.cuda_stream());
    }
    if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaHostPinnedSpace>) {
      return gko::CudaExecutor::create(
          Kokkos::device_id(), create_default_host_executor(),
          std::make_shared<gko::CudaHostAllocator>(Kokkos::device_id()),
          ex.cuda_stream());
    }
  }
#endif
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
      return gko::HipExecutor::create(
          Kokkos::device_id(), create_default_host_executor(),
          std::make_shared<gko::HipAllocator>(), ex.hip_stream());
    }
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPManagedSpace>) {
      return gko::HipExecutor::create(
          Kokkos::device_id(), create_default_host_executor(),
          std::make_shared<gko::HipUnifiedAllocator>(Kokkos::device_id()),
          ex.hip_stream());
    }
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPHostPinnedSpace>) {
      return gko::HipExecutor::create(
          Kokkos::device_id(), create_default_host_executor(),
          std::make_shared<gko::HipHostAllocator>(Kokkos::device_id()),
          ex.hip_stream());
    }
  }
#endif
#ifdef KOKKOS_ENABLE_SYCL
  if constexpr (std::is_same_v<ExecSpace, Kokkos::Experimental::SYCL>) {
    static_assert(
        std::is_same_v<MemorySpace, Kokkos::Experimental::SYCLSpace>,
        "Ginkgo doesn't support shared memory space allocation for SYCL");
    return gko::DpcppExecutor::create(Kokkos::device_id(),
                                      create_default_host_executor());
  }
#endif
  GKO_NOT_IMPLEMENTED;
}

/**
 * Creates an Executor matching the Kokkos::DefaultExecutionSpace.
 *
 * @return  An executor matching the type of Kokkos::DefaultExecutionSpace.
 */
inline std::shared_ptr<gko::Executor>
create_default_executor(Kokkos::DefaultExecutionSpace ex = {}) {
  return create_executor(std::move(ex));
}

template <class SC, class LO, class GO, class NO>
int GinkgoSolver<SC, LO, GO, NO>::initialize() {
  FROSCH_TIMER_START_SOLVER(initializeTime, "GinkgoSolver::initialize");
  this->IsInitialized_ = true;
  this->IsComputed_ = false;
  return 0;
}

template <class SC, class LO, class GO, class NO>
int GinkgoSolver<SC, LO, GO, NO>::compute() {
  FROSCH_TIMER_START_SOLVER(computeTime, "GinkgoSolver::compute");
  FROSCH_ASSERT(this->IsInitialized_,
                "FROSch::GinkgoSolver: !this->IsInitialized_");
  this->IsComputed_ = true;
  return 0;
}

template <class SC, class LO, class GO, class NO>
void GinkgoSolver<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                         ETransp mode, SC alpha,
                                         SC beta) const {
  FROSCH_TIMER_START_SOLVER(applyTime, "GinkgoSolver::apply");
  FROSCH_ASSERT(this->IsComputed_, "FROSch::GinkgoSolver: !this->IsComputed_.");

  // extract execution space of the vectors
  using execution_space = typename XMap::local_map_type::execution_space;
  using memory_space = typename XMap::local_map_type::memory_space;

  auto exec = create_executor(execution_space{}, memory_space{});
  auto host_exec = exec->get_master();

  std::cout << "Executor type: " << typeid(*exec).name() << std::endl;

  ArrayRCP<const SC> valuesx = x.getData(0);
  using Vec = gko::matrix::Dense<SC>;
  auto gko_x = Vec::create_const(
      host_exec, gko::dim<2>{valuesx.size(), 1},
      gko::make_const_array_view(host_exec, valuesx.size(), valuesx.get()), 1);

  ArrayRCP<SC> valuesy = y.getDataNonConst(0);
  auto gko_y = Vec::create(
      host_exec, gko::dim<2>{valuesy.size(), 1},
      gko::make_array_view(host_exec, valuesy.size(), valuesy.get()), 1);

  solver->apply(gko::initialize<Vec>({alpha}, exec), gko_x,
                gko::initialize<Vec>({beta}, exec), gko_y);
}

template <class SC, class LO, class GO, class NO>
int GinkgoSolver<SC, LO, GO, NO>::updateMatrix(ConstXMatrixPtr k,
                                               bool reuseInitialize) {
  FROSCH_ASSERT(false, "FROSch::GinkgoSolver: updateMatrix() is not "
                       "implemented for the GinkgoSolver yet.");
}

template <class SC, class LO, class GO, class NO>
GinkgoSolver<SC, LO, GO, NO>::GinkgoSolver(ConstXMatrixPtr k,
                                           ParameterListPtr parameterList,
                                           string description)
    : Solver<SC, LO, GO, NO>(k, parameterList, description) {
  FROSCH_TIMER_START_SOLVER(GinkgoSolverTime, "GinkgoSolver::GinkgoSolver");
  FROSCH_ASSERT(!this->K_.is_null(), "FROSch::GinkgoSolver: K_ is null.");

  const CrsMatrixWrap<SC, LO, GO, NO> &crsOp =
      dynamic_cast<const CrsMatrixWrap<SC, LO, GO, NO> &>(*this->K_);

  gko::matrix_assembly_data<SC, LO> mdK{
      gko::dim<2>{this->K_->getLocalNumRows(),
                  this->K_->getColMap()->getLocalNumElements()}};
  for (size_t i = 0; i < this->K_->getLocalNumRows(); i++) {
    ArrayView<const LO> indices;
    ArrayView<const SC> values;
    this->K_->getLocalRowView(i, indices, values);
    for (size_t j = 0; j < indices.size(); j++) {
      mdK.set_value(i, indices[j], values[j]);
    }
  }

  auto exec = gko::ReferenceExecutor::create();

  using Mtx = gko::matrix::Csr<SC, LO>;
  auto mK = gko::share(Mtx::create(exec));
  mK->read(mdK.get_ordered_data());

  using Cg = gko::solver::Cg<SC>;
  using Jac = gko::preconditioner::Jacobi<SC>;
  solver =
      Cg::build()
          .with_preconditioner(Jac::build().with_max_block_size(1u).on(exec))
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(1000u).on(exec))
          .on(exec)
          ->generate(mK);
}

} // namespace FROSch

#endif
