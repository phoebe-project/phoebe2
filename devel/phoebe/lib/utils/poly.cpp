#include <iostream>
#include <cmath>

int main(){

  int n = 3;
  
  double a[]={1,2.3,3.4,4.5};
  
  double x = 0.4,
         f = a[n], 
         df = 0, 
         d2f = 0;
  
  for (int i = n - 1; i >= 0; --i) { 
    d2f = df + x*d2f;
    df = f + x*df;
    f  = a[i] + x*f;
  }
  
  d2f *= 2;
  
  std::cout << f << ' ' << df << ' ' << d2f << '\n';
  
  return 0;
}
