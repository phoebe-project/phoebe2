#pragma once
/*
  WD model of Planck central intensity and intensities with atmospheres
  that available in

    * atmx.f
    * planckint.f

  from http://phoebe-project.org/1.0/releases/wd_2007-08-15_phoebe.tar.gz

  or combined in atomcof.f in phoebe pre v2

  The tables
    * atmcofplanck.dat ~ planck table: 1250 real numbers
    * atmcof.dat ~ atmospheres table: 250800 real numbers

  are available from ftp://ftp.astro.ufl.edu/pub/wilson/lcdc2015/


  Bandpass Label Assignments for Stellar Atmospheres

  Label   Bandpass   Reference for Response Function
  (ifil)
  -----   --------   -------------------------------
     1        u      Crawford, D.L. and Barnes, J.V. 1974, AJ, 75, 978
     2        v          "                "           "
     3        b          "                "           "
     4        y          "                "           "
     5        U      Buser, R. 1978, Ang, 62, 411
     6        B      Azusienis and Straizys 1969, Sov. Astron., 13, 316
     7        V          "             "                "
     8        R      Johnson, H.L. 1965, ApJ, 141, 923
     9        I         "            "    "
    10        J         "            "    "
    11        K         "            "    "
    12        L         "            "    "
    13        M         "            "    "
    14        N         "            "    "
    15        R_c    Bessell, M.S. 1983, PASP, 95, 480
    16        I_c       "            "    "
    17      230      Kallrath, J., Milone, E.F., Terrell, D., Young, A.T. 1998, ApJ, 508, 308
    18      250         "             "             "           "
    19      270         "             "             "           "
    20      290         "             "             "           "
    21      310         "             "             "           "
    22      330         "             "             "           "
    23     'TyB'    Tycho catalog B
    24     'TyV'    Tycho catalog V
    25     'HIP'    Hipparcos catalog

  Author: Martin Horvat, August 2016
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <string>

namespace wd_atm {

  const double M_LGPI = 0.4971498726941338;  // log10(pi)

  const int N_planck = 1250;

  const int N_atm = 250800;

  template<class T> inline T pow10(const T& x){
    return std::pow(10, x);
  }

  /*
    Reading a fortran file of maximal length n.

    Input:
      filename - name of the read file

    Output:
      data - array of length n
  */
  template <class T, int n>
  int read_data(const char *filename, T *data) {

    // store data in a string
    std::string s;
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) return 0;

    in.seekg(0, std::ios::end);
    s.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read((char*)s.data(), s.size());
    in.close();

    // replace D with E, FORTRAN JOKE ON HUMANITY
    std::replace(s.begin(), s.end(), 'D', 'E');

    // read word by word from a string
    std::istringstream iss(s);
    int len = 0;
    while (iss >> *(data++) && len < n) ++len;

    return len;
  }

  /*
    Generate values of Legendre polynomials P_i(x), where subscript
    i = 0, 1,2,.. is the degree of the polynomal.

    Input:
      x - value of the argument
      n = maximal degree - 1

    Output:
     pleg - array of length n,  pleg[i] = P_i(x)

  */

  template <class T>
  void legendre(const T & x, T *pleg, const int &n) {

    pleg[0] = 1;
    pleg[1] = x;

    if (n <= 2) return;

    for (int i = 2; i < n; ++i)
      pleg[i] = (x*(2*i - 1)*pleg[i-1] - (i - 1)*pleg[i-2])/i;

  }

  template <class T>
  T legendre_sum(const T & x, const T *a, const int &n) {

    if (n == 1) return a[0];

    if (n == 2) return a[0] + a[1]*x;

    T sum = a[0] + a[1]*x,
      tmp,
      f[2] = {1, x};

    for (int i = 2; i < n; ++i) {
      tmp = f[1];
      sum += a[i]*(f[1] = (x*(2*i - 1)*tmp - (i - 1)*f[0])/i);
      f[0] = tmp;
    }

    return sum;
  }

  template <class T, int n>
  T legendre_sum_nocheck(const T & x, const T *a) {

    if (n == 1) return a[0];

    if (n == 2) return a[0] + a[1]*x;

    T sum = a[0] + a[1]*x,
      tmp,
      f[2] = {1, x};

    for (int i = 2; i < n; ++i) {
      tmp = f[1];
      sum += a[i]*(f[1] = (x*(2*i - 1)*tmp - (i - 1)*f[0])/i);
      f[0] = tmp;
    }

    return sum;
  }


  template <class T, int n>
  T legendre_sum(const T & x, const T *a) {

    if (n == 1) return a[0];

    if (n == 2) return a[0] + a[1]*x;

    T sum = a[0] + a[1]*x,
      tmp, f[2] = {1, x};

    for (int i = 2; a[i] != 0 && i < n; ++i) {
      tmp = f[1];
      sum += a[i]*(f[1] = (x*(2*i - 1)*tmp - (i - 1)*f[0])/i);
      f[0] = tmp;
    }

    return sum;
  }


  /*
    This subroutine returns the log10 (ylog) of a Planck central
    intensity (y), as well as the Planck central intensity (y) itself.
    The subroutine ONLY WORKS FOR TEMPERATURES GREATER THAN OR EQUAL
    500 K OR LOWER THAN 500300 K. For temperatures outside this range,
    the program stops and prints a message.

  Input:
    t - temperature
    ifil - index of the filter 1,2, ..., 25
    plcof - array of coefficients

  Output:
    ylog - log of Planck central intensity
    y - Planck central intensity

  Return:
    true - if no errors, false - otherwise
  */

template <class T>
bool planckint_onlylog(const T & t, const int & ifil, const T *plcof, T & ylog) {

  const char * fname = "planckint_onlylog::";

  if (t <= 500|| t >= 500300){
    std::cerr << fname << "T=" << t << " is illegal.\n";
    return false;
  }

  int ibin;

  T tb, te;

  if (t < 1900) {
    tb=500;
    te=2000;
    ibin=0;
  } else if (t < 5500) {
    tb=1800;
    te=5600;
    ibin=1;
  } else if (t < 20000) {
    tb=5400;
    te=20100;
    ibin=2;
  } else if (t < 100000) {
    tb=19900;
    te=100100;
    ibin=3;
  } else if (t < 500300) {
    tb=99900;
    te=500300;
    ibin=4;
  } else {
    std::cerr << fname << "T=" << t << " is illegal.\n";
    return false;
  }

  // obtain values of lagrange polynomial for argument phas
  T phas = (t - tb)/(te - tb);

  int ib = (ifil - 1)*50 + ibin*10;   // offset in table

  T s = legendre_sum_nocheck<T, 10>(phas, plcof + ib);

  ylog = s - M_LGPI;

  #if defined(LIMB_DARKENING) // NOT USED limb darkening coefficients

  const int ld = 1;
  const T xld = 0;
  const T yld = 0;

  T dark = 1 - xld/3;

  switch (ld) {
    case 2: dark += yld/4.5; break;
    case 3: dark += 0.2*yld; break;
  }

  ylog -= std::log10(dark);
  #endif

  return true;
}


template <class T>
bool planckint(const T & t, const int & ifil, const T *plcof, T & ylog, T & y) {

  bool status = planckint_onlylog(t, ifil, plcof, ylog);

  if (status) y = pow10(ylog);

  return status;
}


/*
  Finding the index the element which still smaller or larger than y.

  Input:
    x - array of size n
    n - size of the array
    y - element which we compare to

  Return:
    index of the element
*/

template <class T, int n>
int binnum(const T *x, const T &y) {

  if (x[0] > x[1]) {  // descending order
    for (int i = 0; i < n; ++i) if (y > x[i]) return i;
  } else {
    for (int i = 0; i < n; ++i) if (y <= x[i]) return i;
  }

  return n;
}

/*
  Calculating logarithm of the intensity using the atmospheres models.

  Input:
   t - temperature
   g - logarithm of surface gravity
   abunin - abundance/metallicity
   ifil - index of the filter 1,2, ...
   plcof - planck table
   grand - atmospheres table

  Output:
    abunin -  the allowed value nearest to the input value.
    xintlog - log of intensity

  Return:
    true -  if everything OK, false - otherwise
*/

template <class T>
bool atmx_onlylog(
  const T &t, const T &g, T &abunin, const int &ifil,
  const T *plcof, const T *grand, T & xintlog)
{
    /* Initialized data */

    const T effwvl[25] = { 350.,412.,430.,546.,365.,440.,550.,680.,
	    870.,1220.,2145.,3380.,4900.,9210.,650.,790.,230.,250.,270.,290.,
	    310.,330.,430.,520.,500. };

    const T abun[19] = { 1.,.5,.3,.2,.1,0.,-.1,-.2,-.3,-.5,-1.,-1.5,
	    -2.,-2.5,-3.,-3.5,-4.,-4.5,-5. };

    const T glog[11] = { 0.,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5. };

    int ifreturn, j, k, k_, m, ib, ii, nj,  it,
        it1, iab, ibb,  iij, kik, njj, ibin, icase, istart, ibinsav;

    T  thighrec, ghightol, thighlog, thightol,
      gg, tb, te,  tt, yy[4], tlowmidlog, dif, pha[4],
      tte[2], dif1, dif2, thighmidlog, trec, tlog, glow, tlow, xnum,
      ghigh, tbrec, denom, thigh, terec, tblog, telog, slope,
	    yylow, yyhigh, wvlmax, fractol, tlowrec, glowtol, tlowlog, tlowtol;


/* cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
/*  Ramp ranges are set below. The following values seem to work. */
/*  They may be changed. */
    tlowtol = 1500.;
    thightol = 5e4;
    glowtol = .5;
    ghightol = .5;

/* ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
/*  The following lines take care of abundances that may not be among */
/*  the 19 Kurucz values (see abun array). abunin is reset at the allowed value nearest */
/*  the input value. */
    iab = binnum<T,19>(abun,abunin);
    dif1 = abunin - abun[iab - 1];
    if (iab != 19) {
      dif2 = abun[iab] - abun[iab - 1];
      dif = dif1 / dif2;
      if (dif < 0. || dif > .5) ++iab;
    }

/* ~       if(dif1.ne.0.d0) write(6,287) abunin,abun(iab) */
    abunin = abun[iab - 1];
    istart = (iab - 1) * 13200 + 1;

/* *************************************************************** */
    tlog = std::log10(t);
    trec = 1. / t;
    tlow = 3500. - tlowtol;
    if (t < tlow) return  planckint_onlylog(t, ifil, plcof, xintlog);

    thigh = thightol + 5e4;
    fractol = thightol / 5e4;

    glow = 0. - glowtol;
    if (g < glow) return planckint_onlylog(t, ifil, plcof, xintlog);

    ghigh = ghightol + 5.;
    if (g > ghigh) return planckint_onlylog(t, ifil, plcof, xintlog);

    tt = t;
    gg = g;

    if (g < 0.) gg = 0.;
    if (g > 5.) gg = 5.;

/* cccccccccccccccccccccccccccccccccccccccccccccccccccc */
/* The following is for 4-point interpolation in log g. */
/* cccccccccccccccccccccccccccccccccccccccccccccccccccc */
    m = 4;
    ifreturn = 0;
    icase = istart + (ifil - 1) * 528;
    j = binnum<T, 11>(glog, g);
    k = std::min(std::max(j - (m - 1) / 2, 1), 12 - m);
    if (g <= 0.) j = 1;

    do {
      ib = icase + (k - 1) * 48;
      ib -= 48;
  /* cccccccccccccccccccccccccccccccccccccccccccccccccccc */

      for (ii = 0; ii < m; ++ii) {

        ib += 48;

        bool ok = true;
        for (ibin = 1; ibin <= 4; ++ibin) {
            it = ib + (ibin - 1) * 12;
            it1 = it + 1;
            if (tt <= grand[it1 - 1]) {
              ok = false;
              break;
            }
        }
        if (ok) --ibin;

        tb = grand[it - 1];
        if (tb == 0.) {
          if (ibin == 1) ibin = 2; else if (ibin == 4) ibin = 3;

          it = ib + (ibin - 1) * 12;
          it1 = it + 1;
          tb = grand[it - 1];
        }

        te = grand[it1 - 1];
        ibinsav = ibin;
        thigh = te + fractol * te;
        ibb = ib + 1 + (ibin - 1) * 12;

        pha[ii] = (tt - tb) / (te - tb);

        yy[ii] = legendre_sum<T,10>((pha[ii] < 0 ? 0. : pha[ii]), grand + ibb);

        if (pha[ii] < 0.) {

          tlow = tb - tlowtol;
          planckint_onlylog(tlow, ifil, plcof, yylow);

          if (t < tlow) {
            planckint_onlylog(t, ifil, plcof, yy[ii]);
          } else {

            tlowmidlog = std::log10(tb * tlow) * .5;
            wvlmax = pow10(6.4624 - tlowmidlog);

            if (effwvl[ifil - 1] < wvlmax) {
              tbrec = 1. / tb, tlowrec = 1. / tlow;
              slope = (yy[ii] - yylow) / (tbrec - tlowrec);
              yy[ii] = yylow + slope * (trec - tlowrec);
            } else {
              tblog = std::log10(tb), tlowlog = std::log10(tlow);
              slope = (yy[ii] - yylow) / (tblog - tlowlog);
              yy[ii] = yylow + slope * (tlog - tlowlog);
            }
          }
        }
        ibin = ibinsav;
      }

      /* ccccccccccccccccccccccccccccccccccccccccccccccccccccccc */
      /* Next, do a m-point Lagrange interpolation. */

      xintlog = 0.;
      k_ = k - 1;

      for (ii = 0; ii < m; ++ii) {
        xnum = 1.;
        denom = 1.;
        nj = k_ + ii;

        for (iij = 0; iij < m; ++iij) {
          njj = k_ + iij;
          if (ii != iij) {
            xnum *= gg - glog[njj];
            denom *= glog[nj] - glog[njj];
          }
        }
        xintlog += yy[ii] * xnum / denom;
      }

      /* ccccccccccccccccccccccccccccccccccccccccccccccc */
      /*  Check if a ramp function will be needed, or if we are */
      /*  close to the border and need to interpolate between less */
      /*  than 4 points. */
      /* cccccccccccccccccccccccccccccccccccccccccccccccc */
      if (g < 0.) {

        thigh = (fractol + 1.) * 6e3;

        if (t > thigh) return planckint_onlylog(t, ifil, plcof,  xintlog);

        if (pha[0] > 1.) break;

        if (t < 3500.) {
          planckint_onlylog(tlow, ifil, plcof, yylow);
          tlowmidlog = std::log10(tlow * 3500.) * .5;
          wvlmax = pow10(6.4624 - tlowmidlog);

          if (effwvl[ifil - 1] < wvlmax) {
            tlowrec = 1. / tlow;
            slope = (xintlog - yylow) / (2.8571428571428574e-4 - tlowrec);
            xintlog = yylow + slope * (trec - tlowrec);
          } else {
            tlowlog = std::log10(tlow);
            slope = (xintlog - yylow) / (std::log10(3500.) - tlowlog);
            xintlog = yylow + slope * (tlog - tlowlog);
          }
        }

        planckint_onlylog(t, ifil, plcof, yylow);
        slope = (yylow - xintlog) / glow;
        xintlog = yylow + slope * (g - glow);
        return true;
      }


      if (g > 5.) {
        thigh = (fractol + 1.) * 5e4;

        if (t > thigh)
          return planckint_onlylog(t, ifil, plcof,  xintlog);

        if (t > 5e4) { j = 10; break; }

        planckint_onlylog(t, ifil, plcof, yyhigh);
        slope = (yyhigh - xintlog) / (ghigh - 5.);
        xintlog = yyhigh + slope * (g - ghigh);
        return true;
      }

      if (t < 3500. || pha[0] <= 1. || ifreturn == 1)  return true;

      if (j == 1 || pha[2] > 1.)  break;

      ++k;

      if (pha[1] > 1) {
        if (j < 10) break;
        ++k;
      }

      if (k > 8) m = 12 - k;
      ifreturn = 1;

    } while (1);

/* cccccccccccccccccccccccccccccccccccccccccccccccc */
    ib = icase + (j - 1) * 48;
    ib -= 48;

    for (kik = 0; kik < 2; ++kik) {

      ib += 48;

      bool ok = true;
      for (ibin = 1; ibin <= 4; ++ibin) {
        it = ib + (ibin - 1) * 12;
        it1 = it + 1;
        if (tt <= grand[it1 - 1]) {
          ok = false;
          break;
        }
      }
      if (ok) --ibin;

      tb = grand[it - 1];
      if (tb == 0.) {
        if (ibin == 1) ibin = 2; else if (ibin == 4) ibin = 3;

        it = ib + (ibin - 1) * 12;
        it1 = it + 1;
        tb = grand[it - 1];
      }
      te = grand[it1 - 1];
      tte[kik] = t;
      if (t > te) tte[kik] = te;

      ibb = ib + 1 + (ibin - 1) * 12;
      pha[kik] = (tte[kik] - tb) / (te - tb);
      yy[kik] = legendre_sum<T,10>(pha[kik], grand + ibb);

      ibin = ibinsav;
    }

    if (g > 5.) {

      yy[0] = yy[1];
      te = tte[1];
      planckint_onlylog(thigh, ifil, plcof, yyhigh);
      thighmidlog = std::log10(te * thigh) * .5;
      wvlmax = pow10(6.4624 - thighmidlog);

      if (effwvl[ifil - 1] < wvlmax) {
        thighrec = 1. / thigh, terec = 1. / te;
        slope = (yyhigh - yy[0]) / (thighrec - terec);
        xintlog = yyhigh + slope * (trec - thighrec);
      } else{
        thighlog = std::log10(thigh), telog = std::log10(te);
        slope = (yyhigh - yy[0]) / (thighlog - telog);
        xintlog = yyhigh + slope * (tlog - thighlog);
      }

      planckint_onlylog(t, ifil, plcof, yyhigh);
      slope = (yyhigh - xintlog) / (ghigh - 5.);
      xintlog = yyhigh + slope * (g - ghigh);
      return true;
    }

    if (g < 0.) {

      te = tte[0];
      planckint_onlylog(thigh, ifil, plcof, yyhigh);
      thighmidlog = std::log10(te * thigh) * .5;
      wvlmax = pow10(6.4624 - thighmidlog);

      if (effwvl[ifil - 1] < wvlmax) {
        thighrec = 1. / thigh, terec = 1. / te;
        slope = (yyhigh - yy[0]) / (thighrec - terec);
        xintlog = yyhigh + slope * (trec - thighrec);
      } else {
        thighlog = std::log10(thigh), telog = std::log10(te);
        slope = (yyhigh - yy[0]) / (thighlog - telog);
        xintlog = yyhigh + slope * (tlog - thighlog);
     }

      planckint_onlylog(t, ifil, plcof, yylow);
      slope = (yylow - xintlog) / glow;
      xintlog = yylow + slope * (g - glow);
      return true;
    }

    slope = (yy[1] - yy[0]) * 2.;
    yy[0] = yy[1] + slope * (g - glog[j]);

    slope = (tte[1] - tte[0]) * 2.;
    te = tte[0] + slope * (g - glog[j - 1]);
    thigh = te * (fractol + 1.);

    if (t > thigh) return planckint_onlylog(t, ifil, plcof,  xintlog);

    planckint_onlylog(thigh, ifil, plcof, yyhigh);
    thighmidlog = std::log10(te * thigh) * .5;
    wvlmax = pow10(6.4624 - thighmidlog);

    if (effwvl[ifil - 1] < wvlmax) {
      thighlog = std::log10(thigh), telog = std::log10(te);
      slope = (yyhigh - yy[0]) / (thighlog - telog);
      xintlog = yyhigh + slope * (tlog - thighlog);
    } else {
      thighrec = 1. / thigh, terec = 1. / te;
      slope = (yyhigh - yy[0]) / (thighrec - terec);
      xintlog = yyhigh + slope * (trec - thighrec);
    }
    return true;
  }


 /*
  Calculation of intensity using the atmospheres models

  Input:
   t - temperature
   g - logarithm of surface gravity
   abunin - abundance/metalicity
   ifil - index of the filter 1,2, ...
   plcof - planck table
   grand - atmospheres table

  Output:
    abunin -  the allowed value nearest to the input value.
    xintlog - log of intensity
    xint - intensity
  Return:
    true - if everything OK, false - otherwise
*/

template <class T>
bool atmx(
  const T &t, const T &g, T &abunin, const int &ifil,
  const T *plcof, const T *grand, T & xintlog, T & xint)
{
  bool status = atmx_onlylog(t, g, abunin, ifil, plcof, grand, xintlog);
  xint = pow10(xintlog);
  return status;
}

} // namespace wd_atm

