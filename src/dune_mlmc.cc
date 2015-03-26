#include <config.h>

#include <dune/common/exceptions.hh> // We use exceptions

#include <dune/mlmc/msfem.hh>
#include <dune/mlmc/mlmc.hh>
#include <dune/multiscale/common/main_init.hh>

#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/common/configuration.hh>

int main(int argc, char** argv) {
  using namespace MultiLevelMonteCarlo;
  try {
    msfem_init(argc, argv);
//    DSC::OutputScopedTiming tm("mlmc.all", DSC_LOG_INFO_0);
    DSC::ScopedTiming tm("mlmc.all");
    MsFemSingleDifference msfem_single;
    MsCgFemDifference ms_cg_fem_diff;
    MultiLevelMonteCarlo::MLMC mlmc;
    mlmc.addDifference(msfem_single, DSC_CONFIG_GET("mlmc.coarse_ranks", 1u));
    mlmc.addDifference(ms_cg_fem_diff, DSC_CONFIG_GET("mlmc.fine_ranks", 1u));
    const auto tolerance = DSC_CONFIG_GET("mlmc.tolerance", 0.1f);
    const auto breaks = DSC_CONFIG_GET("mlmc.breaks", 2u);
    const auto value = mlmc.expectation(tolerance, breaks);
    DSC_LOG_INFO_0 << "\nExpected " << value << std::endl;
  } catch (Dune::Exception& e) {
    return Dune::Multiscale::handle_exception(e);
  } catch (std::exception& s) {
    return Dune::Multiscale::handle_exception(s);
  }
  DSC_PROFILER.outputTimings("profiler");
  Dune::Multiscale::mem_usage();
  return 0;
}

#if 0  // example code only
int run(int argc, char **argv) {
  Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
  if (argc!=10) {
    if(helper.rank()==0) {
  std::cout << "usage: dune_mlmc <pFast> <pMedium> <pSlow> <log2seg> <overlap> "
                << "<corrLen> <sigma> <tol> <breaks>\n";
    }
    return 1;
  }

  // Read parameters
  const auto pFast   = DSC_CONFIG_GET("mlmc.pFast", 1u);
  const auto pMedium = DSC_CONFIG_GET("mlmc.pMedium", 2u);
  const auto pSlow   = DSC_CONFIG_GET("mlmc.pSlow", 3u);
  const auto log2seg = DSC_CONFIG_GET("mlmc.log2seg", 2u);
  const auto overlap = DSC_CONFIG_GET("mlmc.overlap", 1u);
  const auto corrLen = DSC_CONFIG_GET("mlmc.corrLen", 0.2f);
  const auto sigma   = DSC_CONFIG_GET("mlmc.sigma", 1.0f);
  const auto tol     = DSC_CONFIG_GET("mlmc.tol", 0.03f);
  const auto breaks  = DSC_CONFIG_GET("mlmc.breaks", 4u);


  // Multi level Monte Carlo estimator
  MLMC mlmc;
  Single single(log2seg,overlap,corrLen,sigma);
  Fast fast(log2seg,overlap,corrLen,sigma);
  Medium medium(log2seg,overlap,corrLen,sigma);
  Slow slow(log2seg,overlap,corrLen,sigma);

//    mlmc.addDifference(single,pSlow);

  mlmc.addDifference(fast,pFast);
  mlmc.addDifference(medium,pMedium);
  mlmc.addDifference(slow,pSlow);

  // Compute mean flux
  double t = MPI_Wtime();
  double mean = mlmc.expectation(tol,breaks,MPI_COMM_WORLD);
  t = MPI_Wtime() - t;
  DSC_LOG_INFO_0 << "Expected value: " << mean
                 << " Time: " << t << "\n";
}
#endif // 0 example code only
