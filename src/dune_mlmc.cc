// This file is part of the dune-mlmc project:
//   http://users.dune-project.org/projects/dune-mlmc
// Copyright Holders: Rene Milk
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include <config.h>

#include <dune/common/exceptions.hh> // We use exceptions

#include <dune/mlmc/msfem.hh>
#include <dune/mlmc/mlmc.hh>
#include <dune/multiscale/common/main_init.hh>

#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/common/configuration.hh>
#include <dune/stuff/common/signals.hh>

int main(int argc, char** argv) {
  using namespace MultiLevelMonteCarlo;
  using namespace std;
  try {
    msfem_init(argc, argv);
    //    DSC::OutputScopedTiming tm("mlmc.all", DSC_LOG_INFO_0);
    {
      DSC::ScopedTiming tm("mlmc.all");
      auto msfem_single = make_shared<MsFemSingleDifference>();
      auto ms_cg_fem_diff = make_shared<MsCgFemDifference>();
      MultiLevelMonteCarlo::MLMC mlmc;
      mlmc.addDifference(msfem_single, DSC_CONFIG_GET("mlmc.coarse_ranks", 1u));
      mlmc.addDifference(ms_cg_fem_diff, DSC_CONFIG_GET("mlmc.fine_ranks", 1u));
      const auto tolerance = DSC_CONFIG_GET("mlmc.tolerance", 0.1f);
      const auto breaks = DSC_CONFIG_GET("mlmc.breaks", 2u);
      const auto value = mlmc.expectation(tolerance, breaks);
      DSC_LOG_INFO_0 << "\nExpected " << value << endl;
      if(Dune::MPIHelper::getCollectiveCommunication().rank() == 0) {
        unique_ptr<boost::filesystem::ofstream> csvfile(
            DSC::make_ofstream(DSC_CONFIG_GET("global.datadir", "data/") + string("/errors.csv")));
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
    DSC_PROFILER.outputTimings("profiler");
    Dune::Multiscale::mem_usage();
  } catch (Dune::Exception& e) {
    return Dune::Multiscale::handle_exception(e);
  } catch (exception& s) {
    return Dune::Multiscale::handle_exception(s);
  }
  return 0;
}

