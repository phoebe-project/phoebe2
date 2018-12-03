#pragma once

/*
  Export meshed surface for profesional rendering Pov-Ray tool.

  Author: Martin Horvat, July 2016

  Ref:
  * http://xahlee.info/3d/povray-surface.html
  * http://www.povray.org/documentation/3.7.0/
  * http://jmsoler.free.fr/util/blenderfile/fr/povanim_Mesh2_en.htm
  * http://paulbourke.net/miscellaneous/povexamples/
*/

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <utility>
#include <limits>

#include "utils.h"
#include "triang_mesh.h"

namespace povray_export_support {

  std::string sep(const int & n) { return std::string(2*n, ' '); }

  template <class T>
  struct Tprn:T3Dpoint<T>{Tprn(T3Dpoint<T> & v):T3Dpoint<T>(v){}};

  // http://www.povray.org/documentation/view/3.7.0/15/
  template <class T>
  std::ostream& operator << (std::ostream& os, const Tprn<T> & rhs)  {
    os << '<' << rhs[0] << ',' << rhs[2] << ',' << rhs[1] << '>';
    return os;
  }
}

/*

  Exporting triangular mesh in face-vertex format to povray file.


*/

template <class T>
void triangle_mesh_export_povray (
  std::ostream & file,
  std::vector <T3Dpoint<T>> & V,
  std::vector <T3Dpoint<T>> & NatV,
  std::vector <T3Dpoint<int>> & Tr,
  std::string body_color,
  T3Dpoint<T> camera_location,
  T3Dpoint<T> camera_look_at,
  T3Dpoint<T> light_source,
  T *plane_z = 0){

  using namespace povray_export_support;


  int n = 0;

  // colors
  file << sep(n) << "#include \"colors.inc\"\n" ;
  file << sep(n) << "#include \"rad_def.inc\"\n\n";

  // global settings
  // http://wiki.povray.org/content/HowTo:Use_radiosity
  file << sep(n) << "global_settings {\n";
  ++n;
  file << sep(n)<< "radiosity {\n";
  ++n;
  file << sep(n) << "Rad_Settings(Radiosity_Normal,off,off)\n";
  --n;
  file << sep(n) << "}\n";
  --n;
  file << sep(n) <<  "}\n\n";

  file << sep(n) << "#default {finish{ambient 0}}\n\n";

  //
  // camera
  // http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2
  //

  file << sep(n) << "camera {\n";
  ++n;
  file << sep(n) << "location " << Tprn<T>(camera_location) << "\n";
  file << sep(n) << "look_at "  << Tprn<T>(camera_look_at) << "\n";
  --n;
  file << sep(n) << "}\n\n"; // camera

  //
  // light source
  //

  file << sep(n) << "light_source {\n";
  ++n;
  file << sep(n) << Tprn<T>(light_source) << " color White\n";
  --n;
  file << sep(n) << "}\n\n";  // light_source


  //
  // plane
  // http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_5_3_1
  //
  if (plane_z) {
    T3Dpoint<T> plane_n(0,0,1);

    file << sep(n) << "plane {\n";
    ++n;
    file << sep(n) << Tprn<T>(plane_n) << "," << *plane_z << '\n';
    file << sep(n) << "pigment {checker color White, color Gray}\n";
    --n;
    file << sep(n) << "}\n\n"; // plane
  }
  //
  // mesh2
  // http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_5_2_4
  //
  file << sep(n) << "mesh2 {\n";

  ++n;

   // vertex vectors
  file << sep(n) << "vertex_vectors {\n";
  ++n;
  file << sep(n) << V.size();
  for (auto && v : V) file << ",\n" << sep(n) << Tprn<T>(v);
  --n;
  file << sep(n) << "}\n\n"; // vertex_vectors

  // normal at vertices
  file << sep(n) << "normal_vectors {\n";
  ++n;
  file << sep(n) << NatV.size();
  for (auto && v : NatV) file << ",\n" << sep(n) << Tprn<T>(v);
  --n;
  file << sep(n) << "}\n\n"; // normal_vectors

  // faces of triangles defined by triples of indices
  file << sep(n) << "face_indices {\n";
  ++n;
  file << sep(n) << Tr.size();
  for (auto && t : Tr) file << ",\n" << sep(n) << Tprn<int>(t);
  --n;
  file << sep(n) << "}\n\n"; // face_indices


  // color of the triangles
  file << sep(n) << "pigment {\n";
  ++n;
  file << sep(n) << "color " << body_color << "\n";
  --n;
  file << sep(n) << "}\n";

  // surface finish
  file << sep(n) << "finish {\n";
  ++n;
  file << sep(n) << "reflection {0.1}\n";
  file << sep(n) << "ambient 0.1\n";
  file << sep(n) << "diffuse 0.1\n";
  --n;
  file << sep(n) << "}\n";

  --n;
  file << sep(n) << "}\n"; // mesh2

}
