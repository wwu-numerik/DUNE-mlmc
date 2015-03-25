#ifndef DUNE_MLMC_MSFEM_HH
#define DUNE_MLMC_MSFEM_HH

#include <dune/mlmc/mlmc.hh>

#include <dune/multiscale/common/traits.hh>
#include <dune/gdt/products/boundaryl2.hh>


namespace Dune {
namespace Multiscale {
class LocalsolutionProxy;
}
}


namespace MultiLevelMonteCarlo {

class MsCgFemDifference : public Difference {
public:
  virtual ~MsCgFemDifference() {}
  /// Initialize level.
  /// \param global  global communicator
  /// \param local   communicator of processors involved in this solution
  virtual void init(MPI_Comm global, MPI_Comm local);

  double compute_inflow_difference(const Dune::Multiscale::CommonTraits::GridType& coarse_grid,
                                   const Dune::Multiscale::LocalsolutionProxy &msfem_solution,
                                   const std::shared_ptr<Dune::Multiscale::CommonTraits::GridType> fine_grid = nullptr,
                                   const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType* fine_function = nullptr);

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval();
};

class MsFemSingleDifference : public MsCgFemDifference {
public:
  virtual ~MsFemSingleDifference() {}

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval();
};

void msfem_init(int argc, char** argv);

} // namespace MultiLevelMonteCarlo

#endif // DUNE_MLMC_MSFEM_HH
