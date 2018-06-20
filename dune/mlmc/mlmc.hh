// This file is part of the dune-mlmc project:
//   http://users.dune-project.org/projects/dune-mlmc
// Copyright Holders: Jan Mohring
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef MLMC_H
#define MLMC_H

#include <mpi.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/noncopyable.hpp>
#include <dune/stuff/common/exceptions.hh>
#include <dune/xt/common/configuration.hh>

/// \file Environment for computing expected values by the Multi Level
/// Monte Carlo approach. For each level you have to provide extensions
/// of the base class Difference which implement the two methods
/// init and eval.
/// \author jan.mohring@itwm.fraunhofer.de
/// \date 2015

namespace MultiLevelMonteCarlo {

/// Terminates program if condition is not satisfied.
/// \param condition   condition to check
/// \param message     message to show when condition is false
/// TODO embed into exception handling
static void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << "\n";
    DUNE_THROW(Dune::InvalidStateException, message);
  }
}

/// Checks for power of 2
/// \param x  number to check
static bool isPowerOf2(int x) { return (x != 0) && ((x & (x - 1)) == 0); }

/// Base class of difference of solutions on subsequent levels.
/// On the coarsest level the solution itself has to be returned.
class Difference : boost::noncopyable {
public:
  /// Initialize level.
  /// \param global  global communicator
  /// \param local   communicator of processors involved in this solution
  virtual void init(MPI_Comm global, MPI_Comm local) = 0;

  /// Evaluate difference of solutions on subsequent levels.
  virtual double eval() = 0;

  virtual ~Difference() {}
};

/// Class for doing statistics on a given level.
class Level {
public:
  /// Constructs empty level.
  Level()
    : _n(0)
    , _N(0)
    , _p(0)
    , _g(0)
    , _sumX(0)
    , _sumX2(0)
    , _T(0)
    , _diff(nullptr)
    , _masters(MPI_COMM_NULL) {}

  /// Constructs level from Difference object.
  /// \param diff     Difference object
  /// \param minProc  minimal number of processors needed to compute solution
  Level(const std::shared_ptr<Difference> diff, int minProc = 1)
    : _n(0)
    , _N(0)
    , _p(minProc)
    , _g(0)
    , _sumX(0)
    , _sumX2(0)
    , _T(0)
    , _diff(diff)
    , _masters(MPI_COMM_NULL) {
    check(minProc > 0, "minProc must be positive.");
  }

  /// Assigns communicator and group
  /// \param world   global communicator
  void assignProcessors(MPI_Comm world = MPI_COMM_WORLD) {
    check(_diff != NULL, "No Difference object set.");
    int size, rank;
    int range[1][3];
    MPI_Group all, group, masters;
    MPI_Comm_size(world, &size);
    MPI_Comm_rank(world, &rank);
    MPI_Comm_group(world, &all);
    range[0][0] = rank - rank % _p;
    range[0][1] = range[0][0] + _p - 1;
    range[0][2] = 1;
    MPI_Group_range_incl(all, 1, range, &group);
    MPI_Comm_create(world, group, &_group);
    _g = size / _p;
    _diff->init(world, _group);
    range[0][0] = 0;
    range[0][1] = size - 1;
    range[0][2] = _p;
    MPI_Group_range_incl(all, 1, range, &masters);
    MPI_Comm_create(world, masters, &_masters);
    int grank;
    MPI_Comm_rank(_group, &grank);
    _isMaster = grank == 0;
  }

  /// Sets number of repetitions.
  /// \param N   number of repetitions
  void setRepetitions(int N) { _N = N; }

  /// Gets number of repetitions.
  int getRepetitions() const { return _N; }

  /// Returns number of repetitions till next break.
  /// \param iBreak  index of next break
  /// \param nBreak  total number of breaks
  int nextRepetitions(int iBreak, int nBreak) const { return _N * iBreak / nBreak - _n; }

  /// Returns communicator of masters
  MPI_Comm getMasters() const { return _masters; }

  /// Indicates master
  bool isMaster() const { return _isMaster; }

  /// Returns communicator of masters
  MPI_Comm getGroup() const { return _group; }

  /// Clears statistics.
  void clear() {
    _n = 0;
    _sumX = 0;
    _sumX2 = 0;
    _T = 0;
  }

  /// Updates statistics.
  void update(int n, double sumX, double sumX2, double T) {
    _n += n;
    _sumX += sumX;
    _sumX2 += sumX2;
    _T += T;
  }

  /// Returns mean value of realizations on present level.
  double mean() const { return _sumX / (_n * _g); }

  /// Returns empirical variance of realizations on present level.
  double var() const { return _sumX2 / (_n * _g) - mean() * mean(); }

  /// Returns average time of repetition on present level.
  double time() const { return _T / (_n *  _g); }

  /// Returns number of groups on present level.
  int groups() const { return _g; }

  /// Return number of processors per group.
  int procs() const { return _p; }

  int totalRepetitions() const { return _N * _g; }
  int doneRepetitions() const { return _n * _g; }

  /// Evaluate difference.
  double eval() { return _diff->eval(); }

private:
  int _n;                  ///< number of performed repetitions
  int _N;                  ///< number of required repetitions
  int _p;                  ///< number of processors per realization
  int _g;                  ///< number of groups acting in parallel per repetition
  double _sumX;            ///< sum of results so far
  double _sumX2;           ///< sum of squared results so far
  double _T;               ///< total time of repetitions so far
  const std::shared_ptr<Difference> _diff; ///< pointer to Difference object
  MPI_Comm _group;         ///< communicator of group
  MPI_Comm _masters;       ///< communicator of masters
  bool _isMaster;          ///< flag indicating master
};

/// Environment for computing expected values by the
/// Multi Level Monte Carlo Method.
class MLMC {

public:
  /// Adds difference object.
  /// \param diff     Difference object
  /// \param minProc  minimal number of processors needed to compute solution
  void addDifference(const std::shared_ptr<Difference> diff, int minProc) {
    check(isPowerOf2(minProc), "minProc must be power of 2.");
    _level.emplace_back(diff, minProc);
  }

  /// Computes optimal number of repetitions per level .
  /// \param tol  absolute tolerance of expected value
  void setRepetitions(double tol) {
    double alpha = 0;
    int n = _level.size();
    for (int i = 0; i < n; ++i)
      alpha += sqrt(_level[i].time() * _level[i].var());
    alpha /= tol * tol;
    for (int i = 0; i < n; ++i)
      _level[i].setRepetitions(ceil(alpha * sqrt(_level[i].var() / _level[i].time()) / _level[i].groups()));
  }

  /// Computes the expected value of a scalar random variable up to a given
  /// tolerance by the Multi Level Monte Carlo approach.
  /// \param tol    absolute tolerance of expected value
  /// \param nBreak number of breaks for recomputing required realizations
  double expectation(double tol, int nBreak, MPI_Comm world = MPI_COMM_WORLD) {
    int nLevel = _level.size();
    int rank, size;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    check(isPowerOf2(size), "Number of processors must be power of 2.");

    // XXX
    double duration;
    std::stringstream ss;
    std::fstream fs;
    if (rank == 0) {
      duration = MPI_Wtime();
      ss << DXTC_CONFIG_GET("global.datadir", "data/") << "/mlmc" << size << ".txt";
      fs.open(ss.str(), std::fstream::out);
    }
    // XXX

    // Check input.
    check(nLevel > 0, "No levels set.");
    check(tol > 0, "tol must be positive.");
    check(nBreak > 1, "nBreak must be greater than 1.");

    // Create groups and communicators for different levels.
    int startRepititions = DXTC_CONFIG_GET("mlmc.start_repititions", 16);
    for (int i = 0; i < nLevel; ++i) {
      _level[i].assignProcessors(world);
      _level[i].setRepetitions(std::max(nBreak, nBreak * startRepititions / _level[i].groups()));
    }

    // Loop over breaks and levels
    for (int iBreak = 1; iBreak <= nBreak; ++iBreak) {
      for (int iLevel = 0; iLevel < nLevel; ++iLevel) {
        // Moments of groups.
        Level& l = _level[iLevel];
        MPI_Comm masters = l.getMasters();
        MPI_Comm group = l.getGroup();

        int n = l.nextRepetitions(iBreak, nBreak);
        if (n <= 0)
          continue;

        double gdata[3] = {0};
        gdata[2] = MPI_Wtime();
        for (int i = 0; i < n; ++i) {
          double x = l.eval();
          gdata[0] += x;
          gdata[1] += x * x;
        }
        gdata[2] = MPI_Wtime() - gdata[2];

        double data[3];
        // Accumulate over group masters
        if (l.isMaster()) {
          MPI_Allreduce(gdata, data, 3, MPI_DOUBLE, MPI_SUM, masters);
        }
        // Distribute master results over groups
        MPI_Bcast(data, 3, MPI_DOUBLE, 0, group);
        // Update moments.
        l.update(n, data[0], data[1], data[2]);
      }

      // Compute optimal number of repetitions per algorithm.
      setRepetitions(tol);

      // XXX fs -> std::cout
      if (rank == 0) {
        fs << "Break: " << iBreak << "\n";
        for (int i = 0; i < nLevel; ++i)
          fs << "\tL: " << i 
             << "\tn: " << _level[i].doneRepetitions()
             << "\tN: " << _level[i].totalRepetitions()
             << "\tg: " << _level[i].groups()
             << "\tQ: " << _level[i].mean()
             << "\tV: " << _level[i].var()
             << "\tt: " << _level[i].time()
             << "\n";
        fs << "\n";
        fs.flush();
        // XXX
      }
    }

    // Compute expected value from mean values of differences.
    double e = 0;
    for (int i = 0; i < nLevel; ++i)
      e += _level[i].mean();

    // XXX
    if (rank == 0) {
      duration = MPI_Wtime() - duration;
      fs << "Expected value: " << e << " Duration: " << duration << " s\n";
      fs.close();
    }
    // XXX

    return e;
  }

private:
  std::vector<Level> _level; ///< vector of levels
};
} // namespace MultiLevelMonteCarlo {

#endif
