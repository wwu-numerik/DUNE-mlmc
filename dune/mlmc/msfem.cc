// This file is part of the dune-mlmc project:
//   http://users.dune-project.org/projects/dune-mlmc
// Copyright Holders: Rene Milk
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include <config.h>

#include "msfem.hh"

#include <dune/multiscale/common/main_init.hh>
#include <dune/multiscale/msfem/localsolution_proxy.hh>
#include <dune/multiscale/msfem/localproblems/localgridlist.hh>
#include <dune/multiscale/problems/base.hh>
#include <dune/multiscale/common/grid_creation.hh>
#include <dune/multiscale/msfem/msfem_solver.hh>
#include <dune/multiscale/msfem/fem_solver.hh>
#include <dune/multiscale/msfem/msfem_traits.hh>
#include <dune/multiscale/common/heterogenous.hh>

#if DUNE_MULTISCALE_WITH_DUNE_FEM
#include <dune/fem/mpimanager.hh>
#endif

#include <dune/xt/common/ranges.hh>
#include <dune/stuff/grid/walker/functors.hh>
#include <dune/stuff/grid/walker/apply-on.hh>
#include <dune/xt/common/timings.hh>
#include <dune/xt/common/logging.hh>
#include <dune/xt/common/configuration.hh>
#include <dune/xt/common/signals.hh>
#include <dune/xt/common/memory.hh>

#include <dune/gdt/products/h1.hh>
#include <dune/gdt/products/weightedl2.hh>
#include <dune/gdt/products/boundaryl2.hh>

#include <boost/filesystem.hpp>
#include <tbb/task_scheduler_init.h>


double surface_flow_gdt(const Dune::Multiscale::CommonTraits::GridType &grid,
                    const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType& solution,
                        const DMP::ProblemContainer& problem) {
  using namespace Dune::Multiscale;
  const auto gv = grid.leafGridView();

  // Constants and types
  constexpr auto dim = CommonTraits::world_dim;
  typedef double REAL; //TODO read from input
  typedef typename Dune::FieldVector<REAL,dim> FV;   // point on cell
  typedef typename Dune::FieldMatrix<REAL,dim,dim> FM;   // point on cell
  typedef typename Dune::FieldMatrix<REAL,1,dim> Grad;   // point on cell
  typedef typename Dune::QuadratureRule<REAL,dim-1> QR;
  typedef typename Dune::QuadratureRules<REAL,dim-1> QRS;

  const auto& diffusion = problem.getDiffusion();

  // Quadrature rule
  auto iCell = gv.template begin< 0,Dune::Interior_Partition >();
  auto iFace = gv.ibegin(*iCell);
  const QR& rule = QRS::rule(iFace->geometry().type(),2); // TODO order as para

  // Loop over cells
  REAL localFlux(0);
  for(iCell = gv.template begin< 0,Dune::Interior_Partition >();
      iCell != gv.template end< 0,Dune::Interior_Partition >(); ++iCell) {
    // Loop over interfaces
    const auto local_solution = solution.local_function(*iCell);
    for(iFace = gv.ibegin(*iCell); iFace != gv.iend(*iCell); ++iFace) {
      if(iFace->boundary() && abs(iFace->geometry().center()[0]) < 1e-10) {
        double area = iFace->geometry().volume();
        // Loop over gauss points
        for(auto iGauss = rule.begin(); iGauss != rule.end(); ++iGauss) {
          FV pos = iFace->geometry().global(iGauss->position());
          Grad grad;
          FM diff;
          diffusion.evaluate(pos, diff);
          local_solution->jacobian(pos, grad);
          localFlux -= iGauss->weight() * area * diff[0][0] * grad[0][0];
        }
      }
    }
  }
  localFlux = grid.comm().sum(localFlux);
  return localFlux;
}

void MultiLevelMonteCarlo::MsCgFemDifference::init(Dune::MPIHelper::MPICommunicator global, Dune::MPIHelper::MPICommunicator local) {
  // inits perm field only, no create()
  local_comm_ = local;
  if(init_called_)
    return;

  problem_ = Dune::XT::Common::make_unique<DMP::ProblemContainer>(global, local, DXTC_CONFIG);
  assert(problem_);
  init_called_ = true;
}

double MultiLevelMonteCarlo::MsCgFemDifference::compute_inflow_difference(const Dune::Multiscale::CommonTraits::GridType& coarse_grid,
                                                                          Dune::Multiscale::LocalsolutionProxy &msfem_solution,
                                                                          const std::shared_ptr<Dune::Multiscale::CommonTraits::GridType> fine_grid,
                                                                          const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType* fine_function) {
  using namespace Dune;
  typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
      PartViewType;
  const Multiscale::CommonTraits::SpaceType coarse_space(PartViewType::create(
                                                         coarse_grid, Multiscale::CommonTraits::st_gdt_grid_level));
  Multiscale::CommonTraits::DiscreteFunctionType projected_msfem_solution(
        coarse_space, "MsFEM_Solution");
  Multiscale::MsFEMProjection::project(
        msfem_solution, projected_msfem_solution);
  const auto coarse_flow = surface_flow_gdt(coarse_grid, projected_msfem_solution, *problem_);

  if(fine_function && fine_grid) {
    const auto fine_flow = surface_flow_gdt(*fine_grid, *fine_function, *problem_);

    //fine_function->visualize("fine_sol");
    //projected_msfem_solution.visualize("proj_msfem_sol");

    return fine_flow - coarse_flow;
  }

  return coarse_flow;
}

double MultiLevelMonteCarlo::MsCgFemDifference::eval() {
  using namespace Dune;
  Dune::XT::Common::OutputScopedTiming tm("mlmc.difference_cg-msfem", DXTC_LOG_INFO_0);
  assert(init_called_);
  assert(problem_);
  auto coarse_grid = Multiscale::make_coarse_grid(*problem_, local_comm_);
  // create() new perm field
  problem_->getMutableModelData().prepare_new_evaluation(*problem_);
  auto fine_grid = Multiscale::make_fine_grid(*problem_, coarse_grid, true,local_comm_);

  typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
      PartViewType;
  const Multiscale::CommonTraits::SpaceType coarse_space(PartViewType::create(
                                                           *coarse_grid, Multiscale::CommonTraits::st_gdt_grid_level));


  std::unique_ptr<Multiscale::LocalsolutionProxy> msfem_solution(nullptr);

  Multiscale::LocalGridList localgrid_list(*problem_, coarse_space);
  Dune::XT::Common::timings().start("mlmc.difference_cg-msfem.msfem-solve");
  Multiscale::Elliptic_MsFEM_Solver().apply(*problem_, coarse_space, msfem_solution,
                                            localgrid_list);
  Dune::XT::Common::timings().stop("mlmc.difference_cg-msfem.msfem-solve");

  Dune::XT::Common::timings().start("mlmc.difference_cg-msfem.cgfem-solve");
  Multiscale::Elliptic_FEM_Solver fem(*problem_, fine_grid);
  const auto &fine_fem_solution = fem.solve();
  Dune::XT::Common::timings().stop("mlmc.difference_cg-msfem.cgfem-solve");

  Dune::XT::Common::OutputScopedTiming tmd("mlmc.difference_cg-msfem.compute_inflow_difference", DXTC_LOG_INFO_0);
  return compute_inflow_difference(*coarse_grid, *msfem_solution,
                                   fine_grid, &fine_fem_solution);
}

double MultiLevelMonteCarlo::MsFemSingleDifference::eval() {
  using namespace Dune;
  Dune::XT::Common::OutputScopedTiming tm("mlmc.single_msfem", DXTC_LOG_INFO_0);
  assert(problem_);
//  assert(init_called_);
  auto coarse_grid = Multiscale::make_coarse_grid(*problem_, local_comm_);
  // create() new perm field
  problem_->getMutableModelData().prepare_new_evaluation(*problem_);
  typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
      PartViewType;
  const Multiscale::CommonTraits::SpaceType coarse_space(PartViewType::create(
                                                           *coarse_grid, Multiscale::CommonTraits::st_gdt_grid_level));
  std::unique_ptr<Multiscale::LocalsolutionProxy> msfem_solution(nullptr);

  Multiscale::LocalGridList localgrid_list(*problem_, coarse_space);
  Dune::XT::Common::timings().start("mlmc.single_msfem.msfem-solve");
  Multiscale::Elliptic_MsFEM_Solver().apply(*problem_, coarse_space, msfem_solution,
                                            localgrid_list);
  Dune::XT::Common::timings().stop("mlmc.single_msfem.msfem-solve");


  Dune::XT::Common::OutputScopedTiming tmd("mlmc.single_msfem.compute_inflow_difference", DXTC_LOG_INFO_0);
  return compute_inflow_difference(*coarse_grid, *msfem_solution);
}

//! workaround for https://github.com/wwu-numerik/dune-stuff/issues/42
void set_config_values(const std::vector<std::string> &keys,
                       const std::vector<std::string> &values) {
  assert(keys.size() == values.size());
  for (const auto i : Dune::XT::Common::value_range(keys.size()))
    DXTC_CONFIG.set(keys[i], values[i], true);

  // should just be
  // Dune::XT::Common::Config().add(Dune::XT::Common::Configuration(keys, values), "", true);
}

void handle_sigterm(int signal) {
  DXTC_TIMINGS.stop();
  DXTC_TIMINGS.output_per_rank("profiler");
  std::exit(signal);
}

void MultiLevelMonteCarlo::msfem_init(int argc, char **argv) {
  using namespace std;
#if DUNE_MULTISCALE_WITH_DUNE_FEM
  Dune::Fem::MPIManager::initialize(argc, argv);
#endif
  auto &helper = Dune::MPIHelper::instance(argc, argv);
  if (helper.size() > 1 &&
      !(Dune::Capabilities::isParallel<
        Dune::Multiscale::CommonTraits::GridType>::v)) {
    DUNE_THROW(Dune::InvalidStateException,
               "mpi enabled + serial grid = bad idea");
  }

  // config defaults defaults
  const vector<string> keys{"grids.macro_cells_per_dim",
                            "grids.micro_cells_per_macrocell_dim",
                            "msfem.oversampling_layers", "problem.name"};
  const vector<string> values{"8", "12", "1", "Random"};
  set_config_values(keys, values);

  //if (argc > 1 && boost::filesystem::is_regular_file(argv[1]))
  //  Dune::XT::Common::Config().read_command_line(argc, argv);

  if (argc > 1 ) {
      Dune::XT::Common::Config().read_command_line(argc, argv);
      Dune::Stuff::Common::Config().read_command_line(argc, argv);
  }
  Dune::XT::Common::test_create_directory(DXTC_CONFIG_GET("global.datadir", "data/"));

  // LOG_NONE = 1, LOG_ERROR = 2, LOG_INFO = 4,LOG_DEBUG = 8,LOG_CONSOLE =
  // 16,LOG_FILE = 32
  // --> LOG_ERROR | LOG_INFO | LOG_DEBUG | LOG_CONSOLE | LOG_FILE = 62
  Dune::XT::Common::Logger().create(DXTC_CONFIG_GET("logging.level", 62),
                       DXTC_CONFIG_GET("logging.file", std::string(argv[0]) + ".log"),
                       DXTC_CONFIG_GET("global.datadir", "data"),
                       DXTC_CONFIG_GET("logging.dir", "log" /*path below datadir*/));
  DXTC_TIMINGS.set_outputdir(DXTC_CONFIG_GET("global.datadir", "data"));
  Dune::XT::Common::install_signal_handler(SIGTERM, handle_sigterm);
}



