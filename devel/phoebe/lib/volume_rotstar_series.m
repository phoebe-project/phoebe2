Clear[MyInverseSeries];
MyInverseSeries[a_] := Module[{n = Length[a], b, c, i, j, k},
  b = a/a[[1]];
  Do[
   Do[
    c[i, j + 1] = Sum[c[k, 1] c[i - k, j], {k, 1, i - j}],
    {j, i - 1, 1, -1}
    ];
   c[i, 1] = Boole[i == 1] - Sum[b[[j]] c[i, j], {j, 2, i}],
   {i, n}
   ];
  Table[c[i, 1], {i, n}]/a[[1]]^Range[n]
];

n = 15;
m = 10;

DS = Simplify[ Table[D[1 - Sqrt[(1 - β*s/2)^(-2) - s], {s, i}]/i!, {i, n}] /. s -> 0];
iDS = MyInverseSeries[DS];
Sana[x_, β_] = iDS.(x^Range[Length[iDS]]);
  
Vana[β_]=Integrate[Normal[Series[3/2*Sana[x,β],{β,0,m}]],{x,0,1}];
Aana[β_]=Integrate[Normal[Series[Sqrt[Sana[x,β]+D[Sana[x,β],x]^2/4],{β,0,m}]],{x,0,1}];

Export["v_m"<>ToString[m]<>"_n="<>ToString[n]<>".txt", CForm/@N[CoefficientList[Vana[β], β], 20], "Table"];
Export["a_m"<>ToString[m]<>"_n="<>ToString[n]<>".txt", CForm/@N[CoefficientList[Aana[β], β], 20], "Table"];
