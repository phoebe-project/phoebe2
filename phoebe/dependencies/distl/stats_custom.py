from scipy import stats as _stats
from scipy import interpolate as _interpolate
import numpy as _np

### custom subclasses of scipy.stats.rv_continuous

class delta_gen(_stats.rv_continuous):
    def _pdf(self, x):
        return _np.asarray(x==0.0, dtype='int')

    def _cdf(self, x):
        return _np.asarray(x>=0.0, dtype='int')

    def _ppf(self, q):
        return _np.zeros_like(q)

delta = delta_gen(name='delta')

class base_callable(object):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, x):
        raise NotImplementedError("{} must override __call__".format(self.__class__.__name__))

    def __mul__(self, other):
        # this is an UGLY hack to allow our callable to pass the scipy stats argument checks
        return self

    def __gt__(self, other):
        return True

class interpolate_callable(base_callable):
    def __init__(self, x, y):
        self._interpolator = _interpolate.interp1d(x, y, bounds_error=False, fill_value=0.0, assume_sorted=True)

    def __call__(self, x):
        return self._interpolator(x)

# class composite_callable(base_callable):
#     def __init__(self, d1_call, d2_call, math):
#         # math should be the math on the pdfs, not __and__/__or__
#         self._d1_call = d1_call
#         self._d2_call = d2_call
#         self._math = math
#
#    def __call__(self, x):
#        return getattr(self._d1_call(x), self._math)(self._d2_call(x))


# class generic_pdf_gen(_stats.rv_continuous):
#     """
#     this class takes simply a callable function to expose an arbitrary pdf,
#     and relies on the stats.rv_continuous infrastructure to compute and expose
#     everything else (cdf, ppf, etc).  This results in overheads which should
#     be avoided whenever possible, but does allow for full flexibility when it
#     isn't.
#
#     NOTE: it is important that the pdf_callable has an integral of 1
#     """
#     def _pdf(self, x, pdf_callable):
#         # TODO: I have ZERO idea why sometimes this is receiving an array with
#         # the interpolator and other times just gets the interpolator, but for
#         # now we'll just add another hack ;-)
#         if hasattr(pdf_callable, '__iter__'):
#             # if len(pdf_callable) > 1:
#             #     raise ValueError("could not parse arguments")
#             return pdf_callable[0](x)
#         return pdf_callable(x)

# class generic_cdf_gen(_stats.rv_continuous):
#     """
#     this class takes simply a callable function to expose an arbitrary cdf,
#     and relies on the stats.rv_continuous infrastructure to compute and expose
#     everything else (pdf, ppf, etc).  This results in overheads which should
#     be avoided whenever possible, but does allow for full flexibility when it
#     isn't.
#
#     NOTE: it is important that the cdf_callable starts at 0 (at -inf) and ends at 1 (at +inf)
#     """
#
#     def _cdf(self, x, cdf_callable):
#         # TODO: I have ZERO idea why sometimes this is receiving an array with
#         # the interpolator and other times just gets the interpolator, but for
#         # now we'll just add another hack ;-)
#         if hasattr(cdf_callable, '__iter__'):
#             # if len(cdf_callable) > 1:
#             #     raise ValueError("could not parse arguments")
#             return cdf_callable[0](x)
#         return cdf_callable(x)

class generic_pdf_cdf_ppf_gen(_stats.rv_continuous):
    """
    this class takes callable functions for each of pdf, cdf, and ppf.

    NOTE: it is important that the pdf_callable has an integral of 1
    NOTE: it is important that the cdf_callable starts at 0 (at -inf) and ends at 1 (at +inf)
    """
    def _pdf(self, x, pdf_callable, cdf_callable, ppf_callable):
        # TODO: I have ZERO idea why sometimes this is receiving an array with
        # the interpolator and other times just gets the interpolator, but for
        # now we'll just add another hack ;-)
        if hasattr(pdf_callable, '__iter__'):
            # if len(pdf_callable) > 1:
            #     raise ValueError("could not parse arguments")
            return pdf_callable[0](x)
        return pdf_callable(x)

    def _cdf(self, x, pdf_callable, cdf_callable, ppf_callable):
        # TODO: I have ZERO idea why sometimes this is receiving an array with
        # the interpolator and other times just gets the interpolator, but for
        # now we'll just add another hack ;-)
        if hasattr(cdf_callable, '__iter__'):
            # if len(cdf_callable) > 1:
            #     raise ValueError("could not parse arguments")
            return cdf_callable[0](x)
        return cdf_callable(x)

    def _ppf(self, x, pdf_callable, cdf_callable, ppf_callable):
        # TODO: I have ZERO idea why sometimes this is receiving an array with
        # the interpolator and other times just gets the interpolator, but for
        # now we'll just add another hack ;-)
        if hasattr(ppf_callable, '__iter__'):
            # if len(ppf_callable) > 1:
                # print(ppf_callable)
                # raise ValueError("could not parse arguments")
            return ppf_callable[0](x)
        return ppf_callable(x)

# generic_pdf = generic_pdf_gen(name='generic_pdf')
# generic_cdf = generic_cdf_gen(name='generic_cdf')
generic_pdf_cdf_ppf = generic_pdf_cdf_ppf_gen(name='generic_pdf_cdf_ppf')
