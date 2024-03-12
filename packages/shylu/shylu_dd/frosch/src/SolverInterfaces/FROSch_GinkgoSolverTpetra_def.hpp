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

#if GKO_VERSION_MAJOR * 1000 + GKO_VERSION_MINOR * 10 < 1080
namespace gko {
namespace ext {
namespace kokkos {
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
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::Serial>) {
    return gko::ReferenceExecutor::create();
  }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::OpenMP>) {
    return gko::OmpExecutor::create();
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

} // namespace kokkos
} // namespace ext
} // namespace gko
#endif

namespace FROSch {

using namespace Stratimikos;
using namespace Teuchos;
using namespace Thyra;
using namespace Xpetra;

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
  Teuchos::TimeMonitor::getStackedTimer()->start(
      "Ginkgo Solver - Compute");
  Teuchos::TimeMonitor::getStackedTimer()->start(
      "Ginkgo Solver - Compute - Copy to Host");
  auto host_k = this->K_->getLocalMatrixHost();
  Teuchos::TimeMonitor::getStackedTimer()->stop(
      "Ginkgo Solver - Compute - Copy to Host");

  Teuchos::TimeMonitor::getStackedTimer()->start(
      "Ginkgo Solver - Compute - Matrix Data");
  gko::matrix_data<SC, LO> mdK{
      gko::dim<2>{static_cast<gko::size_type>(host_k.numRows()),
                  static_cast<gko::size_type>(host_k.numCols())}};
  mdK.nonzeros.reserve(host_k.nnz());
  for (LO i = 0; i < host_k.numRows(); i++) {
    const auto &row = host_k.row(i);
    for (LO j = 0; j < row.length; j++) {
      mdK.nonzeros.emplace_back(i, row.colidx(j), row.value(j));
    }
  }
  Teuchos::TimeMonitor::getStackedTimer()->stop(
      "Ginkgo Solver - Compute - Matrix Data");

  Teuchos::TimeMonitor::getStackedTimer()->start(
      "Ginkgo Solver - Compute - Matrix Creation");
  using Mtx = gko::matrix::Csr<SC, LO>;
  mtx = Mtx::create(exec);
  mtx->read(std::move(mdK));
  if (perm_factory) {
    perm = gko::as<Permutation>(perm_factory->generate(mtx));
    mtx = mtx->permute(perm);
  }
  Teuchos::TimeMonitor::getStackedTimer()->stop(
      "Ginkgo Solver - Compute - Matrix Creation");

  Teuchos::TimeMonitor::getStackedTimer()->start(
      "Ginkgo Solver - Compute - Factorization");
  solver = solver_factory->generate(mtx);
  Teuchos::TimeMonitor::getStackedTimer()->stop(
      "Ginkgo Solver - Compute - Factorization");

  Teuchos::TimeMonitor::getStackedTimer()->stop(
      "Ginkgo Solver - Compute");

  this->IsComputed_ = true;
  return 0;
}

template <class SC, class LO, class GO, class NO>
void GinkgoSolver<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                         ETransp mode, SC alpha,
                                         SC beta) const {
  FROSCH_TIMER_START_SOLVER(applyTime, "GinkgoSolver::apply");
  FROSCH_ASSERT(this->IsComputed_, "FROSch::GinkgoSolver: !this->IsComputed_.");

  Teuchos::TimeMonitor::getStackedTimer()->start("Ginkgo Solver - Apply");

  auto tpetrax = dynamic_cast<const XTMultiVector &>(x);
  auto viewx = tpetrax.getDeviceLocalView(Access::ReadOnly);
  auto gko_x = Vec::create_const(
      exec, gko::dim<2>{viewx.extent(0), viewx.extent(1)},
      gko::make_const_array_view(exec, viewx.size(), viewx.data()), viewx.stride(0));
  auto in_ptr = gko_x.get();
  if (perm) {
    reordered_in.init_from(gko_x.get());
    gko_x->permute(perm, reordered_in.get(), gko::matrix::permute_mode::rows);
    in_ptr = reordered_in.get();
  }

  auto tpetray = dynamic_cast<const XTMultiVector &>(y);
  auto viewy = tpetray.getDeviceLocalView(Access::ReadWrite);
  auto gko_y =
      Vec::create(exec, gko::dim<2>{viewy.extent(0), viewy.extent(1)},
                  gko::make_array_view(exec, viewy.size(), viewy.data()),
                  viewy.stride(0));
  auto out_ptr = gko_y.get();
  if (perm) {
    reordered_out.init_from(gko_y.get());
    out_ptr = reordered_out.get();
  }

  solver->apply(gko::initialize<Vec>({alpha}, exec), in_ptr,
                gko::initialize<Vec>({beta}, exec), out_ptr);
  if (perm) {
    reordered_out->permute(perm, gko_y,
                           gko::matrix::permute_mode::inverse_rows);
  }

  exec->synchronize();
  Teuchos::TimeMonitor::getStackedTimer()->stop("Ginkgo Solver - Apply");
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
  auto gko_parameter_list = parameterList->sublist("GinkgoSolver");

  Teuchos::TimeMonitor::getStackedTimer()->start("Ginkgo Solver - Creation");

  // extract execution space of the vectors
  using execution_space = typename XMap::local_map_type::execution_space;
  using memory_space = typename XMap::local_map_type::memory_space;

  exec = gko::ext::kokkos::create_executor(execution_space{}, memory_space{});

  auto reordering_str = gko_parameter_list.get("Reordering", "none");
  if(reordering_str == "none") {
    perm_factory = nullptr;
  } else if (reordering_str == "rcm") {
    perm_factory = gko::experimental::reorder::Rcm<LO>::build().on(exec);
  } else if (reordering_str == "amd") {
    perm_factory = gko::experimental::reorder::Amd<LO>::build().on(exec);
  } else {
    FROSCH_ASSERT(false, "FROSch::GinkgoSolver: unknown Ginkgo reordering "
                         "requested: " +
                             reordering_str);
  }

  auto symbolic_type_str = gko_parameter_list.get("SymbolicType", "general");
  gko::experimental::factorization::symbolic_type symbolic_type;
  if (symbolic_type_str == "general") {
    symbolic_type = gko::experimental::factorization::symbolic_type::general;
  } else if (symbolic_type_str == "near_symmetric") {
    symbolic_type =
        gko::experimental::factorization::symbolic_type::near_symmetric;
  } else if (symbolic_type_str == "symmetric") {
    symbolic_type = gko::experimental::factorization::symbolic_type::symmetric;
  } else {
    FROSCH_ASSERT(false, "FROSch::GinkgoSolver: unknown Ginkgo symbolic "
                         "factorization type requested: " +
                             reordering_str);
  }

  using Lu = gko::experimental::factorization::Lu<SC, LO>;
  using Direct = gko::experimental::solver::Direct<SC, LO>;
  solver_factory = Direct::build()
               .with_factorization(
                   Lu::build().with_symbolic_algorithm(symbolic_type))
               .on(exec);

  exec->synchronize();
  Teuchos::TimeMonitor::getStackedTimer()->stop("Ginkgo Solver - Creation");
}

template <class SC, class LO, class GO, class NO>
GinkgoSolver<SC, LO, GO, NO>::~GinkgoSolver() {
  exec->clear_loggers();
}

} // namespace FROSch

#endif
