// This file is part of the dune-mlmc project:
//   http://users.dune-project.org/projects/dune-mlmc
// Copyright Holders: Rene Milk
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_MLMC_MSFEM_HH
#define DUNE_MLMC_MSFEM_HH

#include <dune/mlmc/mlmc.hh>

#include <dune/multiscale/common/traits.hh>
#include <dune/gdt/products/boundaryl2.hh>
#include <dune/multiscale/problems/selector.hh>

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
  virtual void init(Dune::MPIHelper::MPICommunicator global, Dune::MPIHelper::MPICommunicator local);

  double
  compute_inflow_difference(const Dune::Multiscale::CommonTraits::GridType& coarse_grid,
                            Dune::Multiscale::LocalsolutionProxy& msfem_solution,
                            const std::shared_ptr<Dune::Multiscale::CommonTraits::GridType> fine_grid = nullptr,
                            const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType* fine_function = nullptr);

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval();

private:
  std::atomic<bool> init_called_{false};

protected:
  std::unique_ptr<DMP::ProblemContainer> problem_;
  Dune::MPIHelper::MPICommunicator local_comm_;
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
