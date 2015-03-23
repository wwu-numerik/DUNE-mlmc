#include <config.h>

#include <dune/common/exceptions.hh> // We use exceptions

#include <dune/mlmc/msfem.hh>
#include <dune/mlmc/mlmc.hh>

#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/common/configuration.hh>

#if 0  // calcflux fehlt
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
  pFast   = atoi(argv[1]);
  pMedium = atoi(argv[2]);
  pSlow   = atoi(argv[3]);
  log2seg = atoi(argv[4]);
  overlap = atoi(argv[5]);
  corrLen = atof(argv[6]);
  sigma   = atof(argv[7]);
  tol     = atof(argv[8]);
  breaks  = atoi(argv[9]);


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
  if(helper.rank()==0) {
    std::cout << "Expected value: " << mean
              << " Time: " << t << "\n";
  }
}
#endif // 0 //calcflux fehlt

int main(int argc, char** argv) {
  using namespace Dune;
  using namespace std;
  using namespace MultiLevelMonteCarlo;
  try {
    msfem_init(argc, argv);

    MsFemDifference msfem_diff;
    MultiLevelMonteCarlo::MLMC mlmc;
    mlmc.addDifference(msfem_diff, 1);
    const auto tolerance = DSC_CONFIG_GET("mlmc.tolerance", 0.1f);
    const auto breaks = DSC_CONFIG_GET("mlmc.breaks", 2u);
    const auto value = mlmc.expectation(tolerance, breaks);
    DSC_LOG_INFO_0 << "Expected " << value << std::endl;

  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported error: " << e << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
