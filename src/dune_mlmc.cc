#include <config.h>
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh>         // We use exceptions
#include <dune/common/parametertree.hh>

#if DUNE_MULTISCALE_WITH_DUNE_FEM
#include <dune/fem/mpimanager.hh>
#endif

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

#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/common/configuration.hh>
#include <dune/stuff/common/ranges.hh>
#include <dune/stuff/grid/walker/functors.hh>
#include <dune/stuff/grid/walker/apply-on.hh>

#include <dune/gdt/products/boundaryl2.hh>

#include <boost/filesystem.hpp>

#include "dune_mlmc.hh"

class MsFemDifference : public MultiLevelMonteCarlo::Difference {
public:
  virtual ~MsFemDifference() {}
  /// Initialize level.
  /// \param global  global communicator
  /// \param local   communicator of processors involved in this solution
  virtual void init(MPI_Comm global, MPI_Comm local) {
    // inits perm field only, no create()
    DMP::getMutableModelData().problem_init(global, local);
  }

  template <class GridType>
  double compute_inflow_difference(
      const GridType &grid,
      const Dune::Multiscale::CommonTraits::DiscreteFunctionType &
          coarse_function,
      const Dune::Multiscale::CommonTraits::ConstDiscreteFunctionType &
          fine_function) {
    using namespace Dune::Multiscale;
    auto view = grid->leafGridView();
    typedef decltype(view) ViewType;
    typedef typename DSG::Intersection<ViewType>::Type IntersectionType;

    const auto only_on_left_plane =
        [=](const ViewType &, const IntersectionType &it) {
      return it.boundary() && DSC::FloatCmp::eq(it.geometry().center()[0], 0.);
    };
    Dune::GDT::Products::BoundaryL2<ViewType> product(view, only_on_left_plane);
    const auto coarse_flow = product.apply2(coarse_function, coarse_function);
    const auto fine_flow = product.apply2(fine_function, fine_function);
    return fine_flow - coarse_flow;
  }

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval() {
    using namespace Dune;

    auto coarse_grid = Multiscale::make_coarse_grid();
    // create() new perm field
    DMP::getMutableModelData().prepare_new_evaluation();
    typedef Multiscale::CommonTraits::SpaceChooserType::PartViewType
        PartViewType;
    const Multiscale::CommonTraits::SpaceType coarse_space(PartViewType::create(
        *coarse_grid, Multiscale::CommonTraits::st_gdt_grid_level));
    std::unique_ptr<Multiscale::LocalsolutionProxy> msfem_solution(nullptr);

    Multiscale::LocalGridList localgrid_list(coarse_space);
    Multiscale::Elliptic_MsFEM_Solver().apply(coarse_space, msfem_solution,
                                              localgrid_list);
    auto fine_grid = Multiscale::make_fine_grid(coarse_grid, true);
    const Multiscale::CommonTraits::SpaceType fine_space(PartViewType::create(
        *fine_grid, Multiscale::CommonTraits::st_gdt_grid_level));
    Multiscale::Elliptic_FEM_Solver fem(fine_grid);
    const auto &fine_fem_solution = fem.solve();

    // sollte eigentlich auch ohne den umweg der projection auf das feine gitter
    // möglich sein
    Multiscale::CommonTraits::DiscreteFunctionType projected_msfem_solution(
        fine_space, "MsFEM_Solution");
    Multiscale::MsFEMProjection::project(
        *msfem_solution, projected_msfem_solution, msfem_solution->search());
    return compute_inflow_difference(fine_grid, projected_msfem_solution,
                                     fine_fem_solution);
  }
};

//! workaround for https://github.com/wwu-numerik/dune-stuff/issues/42
void set_config_values(const std::vector<std::string> &keys,
                       const std::vector<std::string> &values) {
  assert(keys.size() == values.size());
  for (const auto i : DSC::valueRange(keys.size()))
    DSC_CONFIG.set(keys[i], values[i], true);

  // should just be
  // DSC::Config().add(DSC::Configuration(keys, values), "", true);
}

void msfem_init(int argc, char **argv) {
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

int main(int argc, char **argv) {
  using namespace Dune;
  using namespace std;
  try {
    msfem_init(argc, argv);

    MsFemDifference msfem_diff;
    MultiLevelMonteCarlo::MLMC mlmc;
    mlmc.addDifference(msfem_diff, 1);
    const auto tolerance = DSC_CONFIG_GET("mlmc.tolerance", 0.1f);
    const auto breaks = DSC_CONFIG_GET("mlmc.breaks", 2u);
    const auto value = mlmc.expectation(tolerance, breaks);
    DSC_LOG_INFO << "Expected " << value << std::endl;

  } catch (Dune::Exception &e) {
    std::cerr << "Dune reported error: " << e << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
