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

#include <FROSch_GinkgoSolver_decl.hpp>

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
  this->IsComputed_ = true;
  return 0;
}

template <class SC, class LO, class GO, class NO>
void GinkgoSolver<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                         ETransp mode, SC alpha,
                                         SC beta) const {
  FROSCH_TIMER_START_SOLVER(applyTime, "GinkgoSolver::apply");
  FROSCH_ASSERT(this->IsComputed_, "FROSch::GinkgoSolver: !this->IsComputed_.");

  auto exec = solver->get_executor();
  auto host_exec = exec->get_master();

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
              gko::stop::Iteration::build().with_max_iters(10u).on(exec))
          .on(exec)
          ->generate(mK);
}

} // namespace FROSch

#endif
