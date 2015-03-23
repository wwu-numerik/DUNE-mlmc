#include <config.h>

#include "msfem.hh"

#include <dune/multiscale/common/main_init.hh>
#include <dune/multiscale/msfem/localsolution_proxy.hh>
#include <dune/multiscale/msfem/localproblems/localgridlist.hh>
#include <dune/multiscale/problems/base.hh>
#include <dune/multiscale/problems/selector.hh>
#include <dune/multiscale/common/grid_creation.hh>
#include <dune/multiscale/msfem/msfem_solver.hh>
#include <dune/multiscale/msfem/fem_solver.hh>
#include <dune/multiscale/msfem/msfem_traits.hh>
#include <dune/multiscale/common/heterogenous.hh>

#if DUNE_MULTISCALE_WITH_DUNE_FEM
#include <dune/fem/mpimanager.hh>
#endif

#include <dune/stuff/common/ranges.hh>
#include <dune/stuff/grid/walker/functors.hh>
#include <dune/stuff/grid/walker/apply-on.hh>
#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/common/logging.hh>
#include <dune/stuff/common/configuration.hh>

#include <dune/gdt/products/h1.hh>
#include <dune/gdt/products/weightedl2.hh>
#include <dune/gdt/products/boundaryl2.hh>

#include <boost/filesystem.hpp>


double surface_flow_gdt(const Dune::Multiscale::CommonTraits::GridType &grid,
                    const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType& solution) {
  using namespace Dune::Multiscale;
  const auto gv = grid.leafGridView();
  typedef decltype(gv) ViewType;

  // Constants and types
  constexpr auto dim = CommonTraits::world_dim;
  typedef double REAL; //TODO read from input
  typedef typename Dune::FieldVector<REAL,dim> FV;   // point on cell
  typedef typename Dune::FieldMatrix<REAL,dim,dim> FM;   // point on cell
  typedef typename Dune::FieldMatrix<REAL,1,dim> Grad;   // point on cell
  typedef typename Dune::QuadratureRule<REAL,dim-1> QR;
  typedef typename Dune::QuadratureRules<REAL,dim-1> QRS;

  const auto& diffusion = DMP::getDiffusion();

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
      if(iFace->boundary() && iFace->geometry().center()[0]==0) {
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
  return localFlux;
}

void MultiLevelMonteCarlo::MsCgFemDifference::init(MPI_Comm global, MPI_Comm local) {
  // inits perm field only, no create()
  DMP::getMutableModelData().problem_init(global, local);
}

double MultiLevelMonteCarlo::MsCgFemDifference::compute_inflow_difference(const Dune::Multiscale::CommonTraits::GridType& fine_grid,
                                                                          const Dune::Multiscale::LocalsolutionProxy &msfem_solution,
                                                                          const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType* fine_function) {
  using namespace Dune;
  typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
      PartViewType;
  const Multiscale::CommonTraits::SpaceType fine_space(PartViewType::create(
                                                         fine_grid, Multiscale::CommonTraits::st_gdt_grid_level));
  Multiscale::CommonTraits::DiscreteFunctionType projected_msfem_solution(
        fine_space, "MsFEM_Solution");
  Multiscale::MsFEMProjection::project(
        msfem_solution, projected_msfem_solution, msfem_solution.search());
  const auto coarse_flow = surface_flow_gdt(fine_grid, projected_msfem_solution);
  if(fine_function) {
    const auto fine_flow = surface_flow_gdt(fine_grid, *fine_function);
    return fine_flow - coarse_flow;
  }
  return coarse_flow;
}

double MultiLevelMonteCarlo::MsCgFemDifference::eval() {
  using namespace Dune;
  DSC::OutputScopedTiming tm("mlmc.difference_cg-msfem", DSC_LOG_INFO_0);
  auto coarse_grid = Multiscale::make_coarse_grid();
  // create() new perm field
  DMP::getMutableModelData().prepare_new_evaluation();
  auto fine_grid = Multiscale::make_fine_grid(coarse_grid, true);

  typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
      PartViewType;
  const Multiscale::CommonTraits::SpaceType coarse_space(PartViewType::create(
                                                           *coarse_grid, Multiscale::CommonTraits::st_gdt_grid_level));


  std::unique_ptr<Multiscale::LocalsolutionProxy> msfem_solution(nullptr);

  Multiscale::LocalGridList localgrid_list(coarse_space);
  DSC::profiler().startTiming("mlmc.difference_cg-msfem.msfem-solve");
  Multiscale::Elliptic_MsFEM_Solver().apply(coarse_space, msfem_solution,
                                            localgrid_list);
  DSC::profiler().stopTiming("mlmc.difference_cg-msfem.msfem-solve");

  DSC::profiler().startTiming("mlmc.difference_cg-msfem.cgfem-solve");
  Multiscale::Elliptic_FEM_Solver fem(fine_grid);
  const auto &fine_fem_solution = fem.solve();
  DSC::profiler().stopTiming("mlmc.difference_cg-msfem.cgfem-solve");

  DSC::OutputScopedTiming tmd("mlmc.difference_cg-msfem.compute_inflow_difference", DSC_LOG_INFO_0);
  return compute_inflow_difference(*fine_grid, *msfem_solution,
                                   &fine_fem_solution);
}

//! workaround for https://github.com/wwu-numerik/dune-stuff/issues/42
void set_config_values(const std::vector<std::string> &keys,
                       const std::vector<std::string> &values) {
  assert(keys.size() == values.size());
  for (const auto i : DSC::valueRange(keys.size()))
    DSC_CONFIG.set(keys[i], values[i], true);

  // should just be
  // DSC::Config().add(DSC::Configuration(keys, values), "", true);
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

  if (argc > 1 && boost::filesystem::is_regular_file(argv[1]))
    DSC::Config().read_command_line(argc, argv);

  DSC::testCreateDirectory(DSC_CONFIG_GET("global.datadir", "data/"));

  // LOG_NONE = 1, LOG_ERROR = 2, LOG_INFO = 4,LOG_DEBUG = 8,LOG_CONSOLE =
  // 16,LOG_FILE = 32
  // --> LOG_ERROR | LOG_INFO | LOG_DEBUG | LOG_CONSOLE | LOG_FILE = 62
  const bool useLogger = false;
  DSC::Logger().create(
        DSC_CONFIG_GETB("logging.level", 62, useLogger),
        DSC_CONFIG_GETB("logging.file", std::string(argv[0]) + ".log", useLogger),
      DSC_CONFIG_GETB("global.datadir", "data", useLogger),
      DSC_CONFIG_GETB("logging.dir", "log" /*path below datadir*/, useLogger));
  DSC_CONFIG.set_record_defaults(true);
  DSC_PROFILER.setOutputdir(DSC_CONFIG_GET("global.datadir", "data"));
  DS::threadManager().set_max_threads(DSC_CONFIG_GET("threading.max_count", 1));
}


double MultiLevelMonteCarlo::MsFemSingleDifference::eval() {
  using namespace Dune;
  DSC::OutputScopedTiming tm("mlmc.single_msfem", DSC_LOG_INFO_0);
  auto coarse_grid = Multiscale::make_coarse_grid();
  auto fine_grid = Multiscale::make_fine_grid(coarse_grid, true);
  // create() new perm field
  DMP::getMutableModelData().prepare_new_evaluation();
  typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
      PartViewType;
  const Multiscale::CommonTraits::SpaceType coarse_space(PartViewType::create(
                                                           *coarse_grid, Multiscale::CommonTraits::st_gdt_grid_level));
  std::unique_ptr<Multiscale::LocalsolutionProxy> msfem_solution(nullptr);

  Multiscale::LocalGridList localgrid_list(coarse_space);
  DSC::profiler().startTiming("mlmc.single_msfem.msfem-solve");
  Multiscale::Elliptic_MsFEM_Solver().apply(coarse_space, msfem_solution,
                                            localgrid_list);
  DSC::profiler().stopTiming("mlmc.single_msfem.msfem-solve");


  DSC::OutputScopedTiming tmd("mlmc.single_msfem.compute_inflow_difference", DSC_LOG_INFO_0);
  return compute_inflow_difference(*fine_grid, *msfem_solution);
}
