#include <config.h>
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions

#include <dune/multiscale/common/main_init.hh>
#include <dune/multiscale/msfem/localsolution_proxy.hh>
#include <dune/multiscale/msfem/localproblems/localgridlist.hh>
#include <dune/multiscale/problems/base.hh>
#include <dune/multiscale/problems/selector.hh>
#include <dune/multiscale/common/grid_creation.hh>

#include <dune/multiscale/msfem/msfem_solver.hh>
#include <dune/multiscale/msfem/msfem_traits.hh>

#include "mlmc.hh"

class MsFemDifference : public MultiLevelMonteCarlo::Difference {
public:

  virtual ~MsFemDifference(){}
  /// Initialize level.
  /// \param global  global communicator
  /// \param local   communicator of processors involved in this solution
  virtual void init(MPI_Comm global, MPI_Comm local) {
    Dune::Multiscale::Problem::getMutableModelData().problem_init(global, local);
  }

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval() {
    using namespace Dune;
    auto grid = Multiscale::make_coarse_grid();
    typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType PartViewType;
    const Multiscale::CommonTraits::SpaceType coarseSpace(
          PartViewType::create(*grid, Multiscale::CommonTraits::st_gdt_grid_level));
    std::unique_ptr<Multiscale::LocalsolutionProxy> msfem_solution(nullptr);

    Multiscale::LocalGridList localgrid_list(coarseSpace);
    Multiscale::Elliptic_MsFEM_Solver().apply(coarseSpace, msfem_solution, localgrid_list);

    return -1.;
  }
};

int main(int argc, char** argv)
{
  try {
    Dune::Multiscale::init(argc, argv);

    MsFemDifference msfem_diff;
    auto value = msfem_diff.eval();
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
