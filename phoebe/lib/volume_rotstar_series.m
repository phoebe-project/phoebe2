n=10;
m=10;
Sana[x_,β_]=Normal[FullSimplify[InverseSeries[Simplify[Series[1-Sqrt[(1-1/2β s)^-2-s],{s,0,n}]],x]]];
Vana[β_]=Integrate[Normal[Series[3/2*Sana[x,β],{β,0,m}]],{x,0,1}];
Aana[β_]=Integrate[Normal[Series[Sqrt[Sana[x,β]+D[Sana[x,β],x]^2/4],{β,0,m}]],{x,0,1}];
Export["v_m"<>ToString[m]<>"_n="<>ToString[n]<>".txt", CForm/@N[CoefficientList[Vana[β], β]], "Table"];
Export["a_m"<>ToString[m]<>"_n="<>ToString[n]<>".txt", CForm/@N[CoefficientList[Aana[β], β]], "Table"];
