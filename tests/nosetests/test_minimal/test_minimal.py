import phoebe

b = phoebe.default_binary()
b.add_dataset('lc', passband='Johnson:V', times=[0.25])
b.run_compute()

assert(b['value@fluxes@lc01@model']-2.008089 < 1e-5)
