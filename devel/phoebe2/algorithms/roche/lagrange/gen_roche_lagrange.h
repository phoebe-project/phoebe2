#if !defined(__gen_roche_lagrange_h)
#define __gen_roche_langrange_h


/*
  Supporting functions
*/
 
namespace gen_roche {

template <class T> T sqr(const T & x) { return x*x; }

} // namespace gen_roche

// Lagrange fixed points L1, L2, L3
#include "gen_roche_lagrange_L1.h"
#include "gen_roche_lagrange_L2.h"
#include "gen_roche_lagrange_L3.h"

#endif //#if !defined(__gen_roche_lagrange_h)
