// This file is part of the dune-mlmc project:
//   http://users.dune-project.org/projects/dune-mlmc
// Copyright Holders: Rene Milk
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include <config.h>

#include <dune/common/exceptions.hh> // We use exceptions

#include <dune/mlmc/msfem.hh>
#include <dune/mlmc/mlmc.hh>
#include <dune/multiscale/common/main_init.hh>

#include <dune/xt/common/timings.hh>
#include <dune/xt/common/configuration.hh>
#include <dune/xt/common/signals.hh>

int main(int argc, char** argv) {
  using namespace MultiLevelMonteCarlo;
  using namespace std;
  try {
    msfem_init(argc, argv);
    const size_t max_threads = DXTC_CONFIG_GET("threading.max_count", 1);
    tbb::task_scheduler_init init(max_threads);
    Dune::XT::Common::threadManager().set_max_threads(max_threads);
    //    Dune::XT::Common::OutputScopedTiming tm("mlmc.all", DXTC_LOG_INFO_0);
    {
      Dune::XT::Common::ScopedTiming tm("mlmc.all");
      auto msfem_single = make_shared<MsFemSingleDifference>();
      auto ms_cg_fem_diff = make_shared<MsCgFemDifference>();
      MultiLevelMonteCarlo::MLMC mlmc;
      mlmc.addDifference(msfem_single, DXTC_CONFIG_GET("mlmc.coarse_ranks", 1u));
      mlmc.addDifference(ms_cg_fem_diff, DXTC_CONFIG_GET("mlmc.fine_ranks", 1u));
      const auto tolerance = DXTC_CONFIG_GET("mlmc.tolerance", 0.1f);
      const auto breaks = DXTC_CONFIG_GET("mlmc.breaks", 2u);
      const auto value = mlmc.expectation(tolerance, breaks);
      DXTC_LOG_INFO_0 << "\nExpected " << value << endl;
      if(Dune::MPIHelper::getCollectiveCommunication().rank() == 0) {
        unique_ptr<boost::filesystem::ofstream> csvfile(
            Dune::XT::Common::make_ofstream(DXTC_CONFIG_GET("global.datadir", "data/") + string("/errors.csv")));
        map<string, double> csv{{"expectation", value}};
        const string sep(",");
        for (const auto& key_val : csv) {
          *csvfile << key_val.first << sep;
        }
        *csvfile << endl;
        for (const auto& key_val : csv) {
          *csvfile << key_val.second << sep;
        }
        *csvfile << endl;
      }
    }
    DXTC_TIMINGS.output_per_rank("profiler");
    Dune::Multiscale::mem_usage();
  } catch (Dune::Exception& e) {
    return Dune::Multiscale::handle_exception(e);
  } catch (exception& s) {
    return Dune::Multiscale::handle_exception(s);
  }
  return 0;
}

