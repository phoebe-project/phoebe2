#include <iostream>
#include <cmath>

#include "utils.h"

int main(){
  
  long long n = 1LL << 32;
  
  double a[3] = {1,2,3};
  
  volatile double res;
  
  for (long long i = 0; i < n; ++i) {
    res = utils::hypot3(a);
  }
  
  return 0;
}
