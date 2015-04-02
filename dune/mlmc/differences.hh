// This file is part of the dune-mlmc project:
//   http://users.dune-project.org/projects/dune-mlmc
// Copyright Holders: Jan Mohring
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_MLMC_DIFFERENCES_HH
#define DUNE_MLMC_DIFFERENCES_HH

#include <dune/mlmc/mlmc.hh>
#include <dune/mlmc/calcflux.hh>

namespace MultiLevelMonteCarlo {

// Single solver
class Single : public Difference {
public:
  Single(int log2seg, int overlap, double corrLen, double sigma)
    : _log2seg(log2seg)
    , _overlap(overlap)
    , _corr(corrLen, sigma) {}

  void init(MPI_Comm world, MPI_Comm group) {
    _group = group;
    int rank;
    MPI_Comm_rank(world, &rank);
    _perm.init(_group, _corr, _log2seg, rank + 1, _overlap);
  }

  double eval() {
    _perm.create();
    return calcFlux<DIM, PER>(_log2seg, _perm, _overlap, _group);
  };

private:
  int _log2seg;
  int _overlap;
  MPI_Comm _group;
  COR _corr;
  PER _perm;
};

// Fast solver
class Fast : public Difference {
public:
  Fast(int log2seg, int overlap, double corrLen, double sigma)
    : _log2seg(log2seg)
    , _overlap(overlap)
    , _corr(corrLen, sigma) {}

  void init(MPI_Comm world, MPI_Comm group) {
    _group = group;
    int rank;
    MPI_Comm_rank(world, &rank);
    _perm.init(_group, _corr, _log2seg - 2, rank + 1, _overlap); // -1 : coarser
  }

  double eval() {
    _perm.create();
    return calcFlux<DIM, PER>(_log2seg - 2, _perm, _overlap, _group);
  };

private:
  int _log2seg;
  int _overlap;
  MPI_Comm _group;
  COR _corr;
  PER _perm;
};

// Difference of medium and fast solver
class Medium : public Difference {
public:
  Medium(int log2seg, int overlap, double corrLen, double sigma)
    : _log2seg(log2seg)
    , _overlap(overlap)
    , _corr(corrLen, sigma) {}

  void init(MPI_Comm world, MPI_Comm group) {
    _group = group;
    int rank;
    MPI_Comm_rank(world, &rank);
    _perm.init(group, _corr, _log2seg - 1, rank + 1, 2 * _overlap);
  }

  double eval() {
    _perm.create();
    double fine = calcFlux<DIM, PER>(_log2seg - 1, _perm, _overlap, _group);
    double coarse = calcFlux<DIM, PER>(_log2seg - 2, _perm, _overlap, _group);
    return fine - coarse;
  }

private:
  int _log2seg;
  int _overlap;
  MPI_Comm _group;
  COR _corr;
  PER _perm;
};

// Difference of accurate and medium solver
class Slow : public Difference {
public:
  Slow(int log2seg, int overlap, double corrLen, double sigma)
    : _log2seg(log2seg)
    , _overlap(overlap)
    , _corr(corrLen, sigma) {}

  void init(MPI_Comm world, MPI_Comm group) {
    _group = group;
    int rank;
    MPI_Comm_rank(world, &rank);
    _perm.init(group, _corr, _log2seg, rank + 1, 2 * _overlap);
  }

  double eval() {
    _perm.create();
    double fine = calcFlux<DIM, PER>(_log2seg, _perm, _overlap, _group);
    double coarse = calcFlux<DIM, PER>(_log2seg - 1, _perm, _overlap, _group);
    return fine - coarse;
  }

private:
  int _log2seg;
  int _overlap;
  MPI_Comm _group;
  COR _corr;
  PER _perm;
};

} // namespace MultiLevelMonteCarlo

#endif // DUNE_MLMC_DIFFERENCES_HH
