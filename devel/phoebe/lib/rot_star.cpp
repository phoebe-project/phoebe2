#include <iostream>
#include <cmath>

#include "rot_star.h"

int main(){
  
  double 
    av[2], 
    Omega0 = 3,
    omega = 1;     
        
  std::cout.precision(16);
  std::cout << std::scientific;
  
  rot_star::area_volume(av, 3, Omega0, omega);
  
  std::cout << av[0] << '\t' << av[1] << '\n';
  
  rot_star::volume(av, 3, Omega0, omega);
  
  std::cout << av[0] << '\t' << av[1] << '\n';
  
  return 0;
}
