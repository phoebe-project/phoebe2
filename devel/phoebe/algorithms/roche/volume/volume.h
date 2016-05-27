#if !defined(__volume_h)
#define __volume_h

#include <iostream>
#include <cmath>

/*
  Calculating volume and area of generalized Roche lobes implicitely 
  defined by
   
    Omega(x,y,z) = Omega_0
  
  where Omega is generalized Kopal potential

    Omega = 
    1/rho 
    + q [(delta^2 + rho^2 - 2 rho lambda delta)^(-1/2) - rho lambda/delta^2]
    + 1/2 F^2(1 + q) rho^2 (1 - nu^2)

  with position in spherical coordinates is given as

  x = rho lambda      lambda = sin(theta) cos(phi)
  y = rho  mu         mu = sin(theta) sin(phi)
  z = rho nu          nu = cos(theta)

  Author: Martin Horvat,  April 2016 
*/


namespace gen_roche {
  
  /*
    Computing area of the surface and the volume of the Roche lobes
    intersecting x-axis
    
      {x0,0,0}  {x1,0,0}
    
    The range on x-axis is [x0, x1].
  
    Input:
      x_bounds[2] = {x0,x1} 
      q - mass ratio M2/M1
      F - synchronicity parameter
      delta - separation between the two objects
    
    Using: Integrating surface in cylindric geometry
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in x direction
    
    Precision:
      Relative precision should be better than 1e-4.
      
    Stability:
      It works for overcontact and well detached cases, but could be problematic
      d(polar radius)/dz is too small.
    
    Todo:
      Adaptive step.
    
    Output:
      av[2] = {area, volume}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
  
  template<class T> 
  void area_volume(
    T av[2],
    T xrange[2],
    const T & q,
    const T & F = 1,
    const T & delta = 1)
  {
    
    #if 1
    const int n = 10;

    // Gauss-Lagrange nodes x_i in [0, Pi/2] 
    const T nodes[n]={
      0.02049376457927702323,
      0.10597898397750650975,
      0.25179113626074132181,
      0.4450102168234243146,
      0.66847253098426601147,
      0.90232379581063060776,
      1.12578610997147230463,
      1.31900519053415529742,
      1.46481734281739010948,
      1.550302562215619596};
        
    // Gauss-Lagrange weights * Pi
    const T weights[n]={
      0.20945420548513033,
      0.4695152605605472,
      0.6882801069819194,
      0.84592634724050947,
      0.9284167333216868272,
      0.9284167333216868272,
      0.84592634724050947,
      0.6882801069819194,
      0.4695152605605472,
      0.20945420548513033};
   
    #else
    const int  n = 15;
    
    // Gauss-Lagrange nodes x_i in [0, Pi/2] 
    const T nodes[n]={
      0.009430654293738701511525795554,
      0.049265362404638007011696640038,
      0.119218270605274226185118482656,
      0.216441807654624660222653933546,
      0.336957667679651055804940797624,
      0.475832419302099833753591472893,
      0.627380691485449656524030115277,
      0.78539816339744830961566084582,
      0.943415635309446962707291576363,
      1.09496390749279678547773021875,
      1.23383865911524556342638089402,
      1.35435451914027195900866775809,
      1.45157805618962239304620320898,
      1.5215309643902586122196250516,
      1.56136567250115791771979589609};
        
    // Gauss-Lagrange weights * Pi
    const T weights[n]={
      0.0966141591290711189794453,
      0.22106145785079100848207,
      0.336650619784076362391549,
      0.4384742164293535129419684,
      0.5223501155128774575517226,
      0.584842030033819626900854873,
      0.6233908965445645786498187508,
      0.636418316610479145130430187624,
      0.6233908965445645786498187508,
      0.584842030033819626900854873,
      0.5223501155128774575517226,
      0.4384742164293535129419684,
      0.336650619784076362391549,
      0.22106145785079100848207,
      0.0966141591290711189794453};
    #endif
    
    const int dim = n + 2;
    
    const int m = 1000;
   
    T d2 = delta*delta,
      d3 = d2*delta,
      a = d3*F*F,
      b = a*(1 + q),
      t = xrange[0]/delta, 
      dt = (xrange[1] - xrange[0])/(m*delta),
      c[n], w[n], tmp, t1, f, s, f1, f2, s1, s2;
      

    for (int i = 0; i < n; ++i) {
      tmp = std::cos(nodes[i]);
      c[i] = tmp*tmp;
      w[i] = weights[i]*dt;
    }
    
    //
    // Integration over the surface with RK4
    //  y = R^2, 
    //  y_i(t)' = F(t, y_i(t), c_i) i = 0, .., n -1
    //  A'(t) = sqrt(R + F^2/4) df dx
    //  V'(t) = 1/2 R df dx
    
    T y1[dim], y[dim], k[4][dim];
    
    // init point
    for (int i = 0; i < dim; ++i) y[i] = 0;

    for (int j = 0; j < m; ++j){
    
      // 1. step
      k[0][n] = k[0][n+1] = 0;
      s1 = t*t, s2 = (t-1)*(t-1);
      for (int i = 0; i < n; ++i) {
        s = y[i];
        
        // f1 = (R + t^2)^(-3/2)
        f1 = s + s1;
        f1 = 1/(f1*std::sqrt(f1));
        
        // f1 = (R + (t-1)^2)^(-3/2)
        f2 = s + s2;
        f2 = 1/(f2*std::sqrt(f2));
        
        // k0 = -dx F_t/F_R
        f = 2*(q*(1 + (t-1)*f2) + t*(f1 - b))/(b*c[i] - f1 - q*f2);
        k[0][i] = dt*f;
              
        if (s != 0){
          k[0][n] += w[i]*std::sqrt(s + f*f/4);
          k[0][n+1] += w[i]*s;
        } 
      }
      
     
      // prepare: y1 = y + k0/2
      for (int i = 0; i < n; ++i) y1[i] = y[i] + k[0][i]/2;
      
      // 2. step
      t1 = t + dt/2;
      
      k[1][n] = k[1][n+1] = 0; 
      s1 = t1*t1, s2 = (t1-1)*(t1-1);
      for (int i = 0; i < n; ++i){
        s = y1[i];
        
        // f1 = (R + t1^2)^(-3/2)
        f1 = s + s1;
        f1 = 1/(f1*std::sqrt(f1));
        
        // f1 = (R + (t1-1)^2)^(-3/2)
        f2 = s + s2;
        f2 = 1/(f2*std::sqrt(f2));
        
        // k0 = -dx F_t/F_R
        f = 2*(q*(1 + (t1-1)*f2) + t1*(f1 - b))/(b*c[i] - f1 - q*f2);
        k[1][i] = dt*f;
        
        if (s != 0){
          k[1][n] += w[i]*std::sqrt(s + f*f/4);
          k[1][n+1] += w[i]*s;
        } 
      }
      
      // prepare: y1 = y + k1/2
      for (int i = 0; i < n; ++i) y1[i] = y[i] + k[1][i]/2;
      
      // 3. step
      k[2][n] = k[2][n+1] = 0;
      for (int i = 0; i < n; ++i) {
        s = y1[i];
        
        // f1 = (R + t1^2)^(-3/2)
        f1 = s + s1;
        f1 = 1/(f1*std::sqrt(f1));
        
        // f1 = (R + (t1-1)^2)^(-3/2)
        f2 = s + s2;
        f2 = 1/(f2*std::sqrt(f2));
        
        // k0 = -dx F_t/F_R
        
        f = 2*(q*(1 + (t1-1)*f2) + t1*(f1 - b))/(b*c[i] - f1 - q*f2);
        k[2][i] = dt*f;
                
        if (s != 0){
          k[2][n] += w[i]*std::sqrt(s + f*f/4);
          k[2][n+1] += w[i]*s;
        } 
      }
      
    
      // y1 = y + k2
      for (int i = 0; i < n; ++i) y1[i] = y[i] + k[2][i];
      
      // 4. step
      t1 = t + dt;
      k[3][n] = k[3][n+1] = 0;
      s1 = t1*t1, s2 = (t1-1)*(t1-1);
      for (int i = 0; i < n; ++i){
        s = y1[i];
        
        // f1 = (R + t1^2)^(-3/2)
        f1 = s + s1;
        f1 = 1/(f1*std::sqrt(f1));
        
        // f1 = (R + (t1-1)^2)^(-3/2)
        f2 = s + s2;
        f2 = 1/(f2*std::sqrt(f2));
        
        // k0 = -dx F_t/F_R
        f = 2*(q*(1 + (t1-1)*f2) + t1*(f1 - b))/(b*c[i] - f1 - q*f2);
        k[3][i] = dt*f;
               
        if (s != 0){
          k[3][n] += w[i]*std::sqrt(s + f*f/4);
          k[3][n+1] += w[i]*s;
        }
      }
            
      for (int i = 0; i < dim; ++i)
        y[i] += (k[0][i] + 2*(k[1][i] + k[2][i]) + k[3][i])/6;  
      
      t += dt;
    }   
    
    /*
    for (int i = 0; i < dim; ++i) 
      std::cout << "y=" << y[i] << '\n';
    */
      
    av[0] = d2*y[n];
    av[1] = d3*y[n+1]/2;
  }
  
}


#endif // #if !defined(__volume_h)
