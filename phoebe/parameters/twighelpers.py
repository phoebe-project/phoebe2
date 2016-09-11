


def _uniqueid_to_uniquetwig(bundle, uniqueid):
    """
    """
    # print "*** parameters.twighelpers._uniqueid_to_uniquetwig uniqueid: {}".format(uniqueid)
    return bundle.get_parameter(uniqueid=uniqueid, check_visible=False, check_default=False).uniquetwig

def _twig_to_uniqueid(bundle, twig, **kwargs):
    """
    kwargs are passed on to filter
    """
    res = bundle.filter(twig=twig, force_ps=True, check_visible=False, check_default=False, **kwargs)
    # we force_ps instead of checking is_instance(res, ParameterSet) to avoid
    # having to import from parameters
    if len(res) == 1:
        #~ print "twighelpers._twig_to_uniqueid len(res): {}, res: {}".format(len(res), res)
        return res.to_list()[0].uniqueid
    else:
        raise ValueError("did not return a single unique match to a parameter for '{}'".format(twig))
