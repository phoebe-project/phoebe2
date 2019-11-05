import phoebe
from astropy import units as u

pb = phoebe.atmospheres.passbands.Passband(
    ptf='johnson_v.ptf',
    pbset='Johnson',
    pbname='V',
    effwl=5500.,
    wlunits=u.AA,
    calibrated=True,
    reference='Maiz Apellaniz (2006), AJ 131, 1184',
    version=2.0,
    comments=''
)

pb.compute_blackbody_response()
pb.compute_bb_reddening(verbose=True)

pb.compute_ck2004_response(path='tables/ck2004i', verbose=True)
pb.compute_ck2004_intensities(path='tables/ck2004i', verbose=True)
pb.compute_ck2004_ldcoeffs()
pb.compute_ck2004_ldints()
pb.compute_ck2004_reddening(path='tables/ck2004i', verbose=True)

pb.compute_phoenix_response(path='tables/phoenix', verbose=True)
pb.compute_phoenix_intensities(path='tables/phoenix', verbose=True)
pb.compute_phoenix_ldcoeffs()
pb.compute_phoenix_ldints()
pb.compute_phoenix_reddening(path='tables/phoenix', verbose=True)

pb.import_wd_atmcof('tables/wd/atmcofplanck.dat', 'tables/wd/atmcof.dat', 7)

pb.save_asdf('johnson_v.asdf')
