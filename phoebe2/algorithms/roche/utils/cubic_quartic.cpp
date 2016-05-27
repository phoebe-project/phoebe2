/*
  Testing real roots of polynomials up to degree 4.
  
  Compile: g++ cubic_quartic.cpp -o cubic_quartic -O3 -Wall -DDEBUG -std=c++11
  
  Author: Martin Horvat, April 2016
*/ 
#include <iostream>
#include <cmath>
#include <list>

#include "utils.h"

int main(){

  std::cout.precision(16);

  std::vector<double> roots;
  
  
  
  double a2[3] = {0.1, 1, 2};
  
  utils::solve_quadratic(a2, roots);
  
  std::cout << "Quadratic:\n";
  
  for (auto && v : roots) std::cout << v << '\n';
  
  
  double a3[4] = {0.1, 1, 2, 1};
  
  utils::solve_cubic(a3, roots);
  
  std::cout << "Cubic:\n";
  
  for (auto && v : roots) std::cout << v << '\n';
  
  
  
  double a4[5] = {0.5, 7, 2, -10, -17};
  
  utils::solve_quartic(a4, roots);
  
  std::cout << "Quartic:\n";
  
  for (auto && v : roots) std::cout << v << '\n';  
  
  return 0;
}
