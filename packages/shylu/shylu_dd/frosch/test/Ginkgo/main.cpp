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
// Questions? Contact Alexander Heinlein (a.heinlein@tudelft.nl)
//
// ************************************************************************
//@HEADER

#include <ShyLU_DDFROSch_config.h>

#include <mpi.h>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_StackedTimer.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

// Stratimikos includes
#include <Stratimikos_FROSch_def.hpp>

#include <Tpetra_Core.hpp>

// Xpetra include
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>

// FROSCH thyra includes
#include "Thyra_FROSchLinearOp_def.hpp"
#include "Thyra_FROSchFactory_def.hpp"
#include <FROSch_Tools_def.hpp>

// Ginkgo
#ifdef HAVE_SHYLU_DDFROSCH_GINKGO
#include <ginkgo/ginkgo.hpp>
#endif



using UN    = unsigned;
using SC    = double;
using LO    = int;
using GO    = FROSch::DefaultGlobalOrdinal;
using NO    = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;
using namespace Thyra;

int main(int argc, char *argv[])
{
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();

    CommandLineProcessor My_CLP;

    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

    int M = 3;
    // My_CLP.setOption("M",&M,"H / h.");
    // int Dimension = 2;
    // string xmlFile = "ParameterList.xml";
    // My_CLP.setOption("PLIST",&xmlFile,"File name of the parameter list.");
    bool useepetra = false;
    My_CLP.setOption("USEEPETRA","USETPETRA",&useepetra,"Use Epetra infrastructure for the linear algebra.");

    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc,argv);
    if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }

    CommWorld->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Thyra Laplace Test"));
    TimeMonitor::setStackedTimer(stackedTimer);

    int N = 0;
    int color=1;
    N = (int) (pow(CommWorld->getSize(),1/2.) + 100*numeric_limits<double>::epsilon()); // 1/H
    if (CommWorld->getRank()<N*N) {
        color=0;
    }
    assert(N==1);

    UnderlyingLib xpetraLib = UseTpetra;
    if (useepetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }

    RCP<const Comm<int> > Comm = CommWorld->split(color,CommWorld->getRank());

    if (color==0) {

        Comm->barrier(); if (Comm->getRank()==0) cout << "##############################\n# Assembly System #\n##############################\n" << endl;

        // RCP<ParameterList> parameterList = getParametersFromXmlFile(xmlFile);
        ParameterList GaleriList;
        GaleriList.set("nx", GO(N*M));
        GaleriList.set("ny", GO(N*M));
        GaleriList.set("nz", GO(N*M));
        GaleriList.set("mx", GO(N));
        GaleriList.set("my", GO(N));
        GaleriList.set("mz", GO(N));

        RCP<Map<LO,GO,NO> > UniqueMap = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian2D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
        RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace2D",UniqueMap,GaleriList);
        RCP<Matrix<SC,LO,GO,NO> > K = Problem->BuildMatrix();

        RCP<MultiVector<SC,LO,GO,NO> > x = MultiVectorFactory<SC,LO,GO,NO>::Build(K->getMap(),1);
        RCP<MultiVector<SC,LO,GO,NO> > b = MultiVectorFactory<SC,LO,GO,NO>::Build(K->getMap(),1);

        x->putScalar(ScalarTraits<SC>::zero());
        b->putScalar(ScalarTraits<SC>::one());

        Comm->barrier(); if (Comm->getRank()==0) cout << "##############################\n# Solve System #\n##############################\n" << endl;

        #ifdef HAVE_SHYLU_DDFROSCH_GINKGO

        // Print ginkgo version
        std::clog << gko::version_info::get() << std::endl;

        // Convert matrix to ginkgo format
        gko::matrix_assembly_data<SC,LO> mdK{gko::dim<2>{K->getLocalNumRows(),K->getColMap()->getLocalNumElements()}};
        for (size_t i = 0; i < K->getLocalNumRows(); i++) {
            ArrayView<const LO> indices;
            ArrayView<const SC> values;
            K->getLocalRowView(i,indices,values);
            std::cout << i << " " << indices << " " << values << std::endl;
            for (size_t j = 0; j < indices.size(); j++) {
                mdK.set_value(i,indices[j],values[j]);
            }
        }
        auto exec = gko::ReferenceExecutor::create();
        using Mtx = gko::matrix::Csr<SC,LO>;
        auto mK = gko::share(Mtx::create(exec));
        mK->read(mdK.get_ordered_data());
        gko::write(std::cout, mK);

        // Convert right hand side vector to ginkgo format
        gko::matrix_assembly_data<SC,LO> mdb{gko::dim<2>{K->getLocalNumRows(),1}};
        ArrayRCP<SC> valuesb = b->getDataNonConst(0);
        using Vec = gko::matrix::Dense<SC>;
        auto mb = Vec::create(exec,gko::dim<2>{valuesb.size(),1},gko::make_array_view(exec,valuesb.size(),valuesb.get()),1);
        gko::write(std::cout, mb);

        using Cg = gko::solver::Cg<SC>;
        using Jac = gko::preconditioner::Jacobi<SC>;
        auto solver = Cg::build().with_preconditioner(Jac::build()
                                    .with_max_block_size(1u).on(exec))
                                    .with_criteria(gko::stop::Iteration::build()
                                        .with_max_iters(10u).on(exec))
                                        .on(exec)->generate(mK);

        // Convert solution vector to ginkgo format
        gko::matrix_assembly_data<SC,LO> mdx{gko::dim<2>{K->getLocalNumRows(),1}};
        ArrayRCP<SC> valuesx = x->getDataNonConst(0);
        auto mx = Vec::create(exec,gko::dim<2>{valuesx.size(),1},gko::make_array_view(exec,valuesx.size(),valuesx.get()),1);
        // auto mx = Vec::create(exec);
        mx->read(mdx.get_ordered_data());

        solver->apply(mb,mx);

        gko::write(std::cout, mx);
        RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); x->describe(*fancy,VERB_EXTREME);

        #endif

        Comm->barrier(); if (Comm->getRank()==0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }

    CommWorld->barrier();
    stackedTimer->stop("Thyra Laplace Test");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report(*out,CommWorld,options);

    return(EXIT_SUCCESS);

}
