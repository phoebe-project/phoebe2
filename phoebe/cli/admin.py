"""
phoebe-admin CLI — administrative utilities for the PHOEBE package.

Usage examples::

    phoebe-admin create stock-passbands
    phoebe-admin create stock-passbands --overwrite
    phoebe-admin create stock-passbands --quiet
"""

import argparse
import os

from phoebe import u
from phoebe.atmospheres import models
from phoebe.atmospheres.passbands import Passband


def create_stock_passbands(tables_dir=None, overwrite=False, verbose=True):
    """
    Generate the bolometric and Johnson V stock passband FITS files. Stock passbands
    only include blackbody, ck2004 and phoenix model atmospheres. This is to minimize
    the size of stock passbands for pip.

    Parameters
    ----------
    tables_dir : str or None
        Base directory that contains ``ptfs/``, ``wd/`` and ``passbands/``
        sub-directories. Defaults to ``tables/`` relative to the cwd.
    overwrite : bool
        When *True* regenerate files that already exist on disk.
    verbose : bool
        Pass ``verbose=True`` to ``Passband.compute_intensities``.
    """

    if tables_dir is None:
        tables_dir = 'tables'

    pb_dir = os.path.join(tables_dir, 'passbands')
    wd_dir = os.path.join(tables_dir, 'wd')

    # Bolometric passband:
    bol_fits = os.path.join(pb_dir, 'bolometric.fits')
    if not overwrite:
        try:
            Passband.load(bol_fits)
            if verbose:
                print(f'Skipping bolometric (already exists): {bol_fits}')
        except FileNotFoundError:
            _create_bolometric(tables_dir, bol_fits, verbose)
    else:
        _create_bolometric(tables_dir, bol_fits, verbose)

    # Johnson V passband:
    jv_fits = os.path.join(pb_dir, 'johnson_v.fits')
    if not overwrite:
        try:
            Passband.load(jv_fits)
            if verbose:
                print(f'Skipping Johnson V (already exists): {jv_fits}')
        except FileNotFoundError:
            _create_johnson_v(tables_dir, wd_dir, jv_fits, verbose)
    else:
        _create_johnson_v(tables_dir, wd_dir, jv_fits, verbose)


def _create_bolometric(tables_dir, out_path, verbose):
    pb = Passband(
        ptf=os.path.join(tables_dir, 'ptfs', 'bolo900.ptf'),
        pbset='Bolometric',
        pbname='900-40000',
        wlunits=u.nm,
        calibrated=True,
        reference='Flat response to simulate bolometric throughput',
        version=1.6,
        comment='TMAP/Tremblay model atmospheres added',
    )

    atm = models.BlackbodyModelAtmosphere()
    pb.compute_intensities(atm=atm, include_mus=False, include_ld=False, include_extinction=False, verbose=verbose)

    supported_atms = models.get_supported_atms(return_dict=True)
    for atm_name in ['ck2004', 'phoenix']:
        atm = supported_atms[atm_name].from_path(os.path.join(tables_dir, atm_name))
        pb.compute_intensities(atm=atm, include_mus=True, include_ld=True, include_extinction=False, verbose=verbose)

    pb.save(out_path)
    if verbose:
        print(f'Saved: {out_path}')


def _create_johnson_v(tables_dir, wd_dir, out_path, verbose):
    pb = Passband(
        ptf=os.path.join(tables_dir, 'ptfs', 'johnson_v.ptf'),
        pbset='Johnson',
        pbname='V',
        wlunits=u.AA,
        calibrated=True,
        reference='Maiz Apellaniz (2006), AJ 131, 1184',
        version=1.6,
        comment='TMAP/Tremblay model atmospheres added',
    )

    atm = models.BlackbodyModelAtmosphere()
    pb.compute_intensities(atm=atm, include_mus=False, include_ld=False, include_extinction=False, verbose=verbose)

    supported_atms = models.get_supported_atms(return_dict=True)
    for atm_name in ['ck2004', 'phoenix']:
        atm = supported_atms[atm_name].from_path(os.path.join(tables_dir, atm_name))
        pb.compute_intensities(atm=atm, include_mus=True, include_ld=True, include_extinction=False, verbose=verbose)

    pb.import_wd_atmcof(os.path.join(wd_dir, 'atmcofplanck.dat'), os.path.join(wd_dir, 'atmcof.dat'), 7)

    pb.save(out_path)
    if verbose:
        print(f'Saved: {out_path}')


def _init_parser():
    parser = argparse.ArgumentParser(
        prog='phoebe-admin',
        description='Administrative utilities for the PHOEBE package.',
    )
    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND')
    subparsers.required = True

    # 'create' sub-command
    create_parser = subparsers.add_parser('create', help='Create resources.')
    create_subparsers = create_parser.add_subparsers(
        dest='resource', metavar='RESOURCE'
    )
    create_subparsers.required = True

    # 'stock-passbands' action:
    sp = create_subparsers.add_parser(
        'stock-passbands',
        help='Generate bolometric and Johnson V stock passband files.',
    )
    sp.add_argument(
        '--tables-dir',
        default=None,
        metavar='DIR',
        help=(
            'Base directory containing ptfs/, wd/ and passbands/ sub-directories. '
            'Defaults to tables/ relative to the current working directory.'
        ),
    )
    sp.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing passband files.',
    )
    sp.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output.',
    )

    return parser


def main():
    parser = _init_parser()
    args = parser.parse_args()

    if args.command == 'create' and args.resource == 'stock-passbands':
        create_stock_passbands(
            tables_dir=args.tables_dir,
            overwrite=args.overwrite,
            verbose=not args.quiet,
        )
