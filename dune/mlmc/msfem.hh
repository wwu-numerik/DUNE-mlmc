#ifndef DUNE_MLMC_MSFEM_HH
#define DUNE_MLMC_MSFEM_HH

#include <dune/mlmc/mlmc.hh>

#include <dune/multiscale/common/traits.hh>
#include <dune/gdt/products/boundaryl2.hh>

namespace MultiLevelMonteCarlo {

class MsFemDifference : public Difference {
public:
  virtual ~MsFemDifference() {}
  /// Initialize level.
  /// \param global  global communicator
  /// \param local   communicator of processors involved in this solution
  virtual void init(MPI_Comm global, MPI_Comm local);

  double compute_inflow_difference(const Dune::Multiscale::CommonTraits::GridType& grid,
                                   const Dune::Multiscale::CommonTraits::DiscreteFunctionType& coarse_function,
                                   const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType& fine_function);

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval();
};

void msfem_init(int argc, char** argv);
}

#endif // DUNE_MLMC_MSFEM_HH
