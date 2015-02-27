#ifndef SURFACEFLOW
#define SURFACEFLOW

/// Integrates flux \f$ -k \frac{\partial u}{\partial x} \f$ over face x=0.
/// \tparam GV   type of grid view
/// \tparam GFS  type of grid function space
/// \tparam U    type of coefficient vector
/// \tparam PR   type of permeability function
/// \param gv    grid view
/// \param gfs   grid function space
/// \param u     coefficient vector of solution
/// \param perm  permeability
template<typename GV, typename GFS, typename U, typename PR>
double surfaceFlow (const GV& gv, const GFS& gfs, const U& u, const PR& perm) {

  // Constants and types
  const int dim = gv.dimension;
  typedef double REAL; //TODO read from input
  typedef typename Dune::FieldVector<REAL,dim> FV;   // point on cell
  typedef typename Dune::QuadratureRule<REAL,dim-1> QR;
  typedef typename Dune::QuadratureRules<REAL,dim-1> QRS;
  typedef typename Dune::PDELab::DiscreteGridFunctionGradient<GFS,U> DGFG;

  // Quadrature rule
  auto iCell = gv.template begin< 0,Dune::Interior_Partition >();
  auto iFace = gv.ibegin(*iCell);
  const QR& rule = QRS::rule(iFace->geometry().type(),2); // TODO order as para

  // Discrete grid function gradient
  DGFG dgfg(gfs,u);

  // Loop over cells
  REAL localFlux;
  for(iCell = gv.template begin< 0,Dune::Interior_Partition >(); 
      iCell != gv.template end< 0,Dune::Interior_Partition >(); ++iCell) {
    // Loop over interfaces
    for(iFace = gv.ibegin(*iCell); iFace != gv.iend(*iCell); ++iFace) {
      if(iFace->boundary() && iFace->geometry().center()[0]==0) {
        double area = iFace->geometry().volume();
        // Loop over gauss points
        for(auto iGauss = rule.begin(); iGauss != rule.end(); ++iGauss) {
          FV pos = iFace->geometry().global(iGauss->position());
          FV grad;
          dgfg.evaluate(*iCell,pos,grad);
          localFlux -= iGauss->weight() * area * perm(pos) * grad[0];
        }
      }
    }
  }
  REAL flux;
  MPI_Allreduce(&localFlux, &flux, 1, MPI_DOUBLE, MPI_SUM, gv.comm());
  return flux;
};

#endif
