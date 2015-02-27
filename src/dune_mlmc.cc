#include <config.h>
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions
#include <dune/common/parametertree.hh>

#if DUNE_MULTISCALE_WITH_DUNE_FEM
# include <dune/fem/mpimanager.hh>
#endif

#include <dune/multiscale/common/main_init.hh>
#include <dune/multiscale/msfem/localsolution_proxy.hh>
#include <dune/multiscale/msfem/localproblems/localgridlist.hh>
#include <dune/multiscale/problems/base.hh>
#include <dune/multiscale/problems/selector.hh>
#include <dune/multiscale/common/grid_creation.hh>
#include <dune/multiscale/msfem/msfem_solver.hh>
#include <dune/multiscale/msfem/msfem_traits.hh>

#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/common/configuration.hh>

#include <boost/filesystem.hpp>

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

//! workaround for https://github.com/wwu-numerik/dune-stuff/issues/42
void set_config_values(const std::vector<std::string>& keys, const std::vector<std::size_t>& values) {
  assert(keys.size()==values.size());
  for(const auto i : DSC::valueRange(keys.size()))
    DSC_CONFIG.set(keys[i], values[i]);

  // should just be
  // DSC::Config().add(DSC::Configuration(keys, values), "", true);
}

void msfem_init(int argc, char** argv) {
  using namespace std;
#if DUNE_MULTISCALE_WITH_DUNE_FEM
  Dune::Fem::MPIManager::initialize(argc, argv);
#endif
  auto& helper = Dune::MPIHelper::instance(argc, argv);
  if (helper.size() > 1 && !(Dune::Capabilities::isParallel<Dune::Multiscale::CommonTraits::GridType>::v)) {
    DUNE_THROW(Dune::InvalidStateException, "mpi enabled + serial grid = bad idea");
  }

  if (argc > 1 && boost::filesystem::is_regular_file(argv[1]))
    DSC::Config().read_command_line(argc, argv);

  DSC::testCreateDirectory(DSC_CONFIG_GET("global.datadir", "data/"));

  // LOG_NONE = 1, LOG_ERROR = 2, LOG_INFO = 4,LOG_DEBUG = 8,LOG_CONSOLE = 16,LOG_FILE = 32
  // --> LOG_ERROR | LOG_INFO | LOG_DEBUG | LOG_CONSOLE | LOG_FILE = 62
  const bool useLogger = false;
  DSC::Logger().create(DSC_CONFIG_GETB("logging.level", 62, useLogger),
                       DSC_CONFIG_GETB("logging.file", std::string(argv[0]) + ".log", useLogger),
                       DSC_CONFIG_GETB("global.datadir", "data", useLogger),
                       DSC_CONFIG_GETB("logging.dir", "log" /*path below datadir*/, useLogger));
  DSC_CONFIG.set_record_defaults(true);
  DSC_PROFILER.setOutputdir(DSC_CONFIG_GET("global.datadir", "data"));
  DS::threadManager().set_max_threads(DSC_CONFIG_GET("threading.max_count", 1));
}

int main(int argc, char** argv)
{
  using namespace Dune;
  using namespace std;
  try {
    msfem_init(argc, argv);

    const vector<string> keys{"grids.macro_cells_per_dim", "grids.micro_cells_per_macrocell_dim",
                              "msfem.oversampling_layers"};
    const vector<size_t> values{ 8, 12 , 1 };;
    set_config_values(keys, values);
    MsFemDifference msfem_diff;
    auto value = msfem_diff.eval();

    MultiLevelMonteCarlo::MLMC mlmc;

  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
