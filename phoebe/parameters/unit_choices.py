from phoebe import u

def unit_choices(unit):
    """
    Return a (non-complete) list of available units of the same physical type.

    Arguments
    -----------
    * `unit` (astropy.Unit object or str): unit to be converted

    Returns
    ----------
    * (list of strings) list of available units with the same physical type
    """
    if isinstance(unit, str):
        unit = u.Unit(unit)

    physical_type = unit.physical_type

    if physical_type == 'dimensionless':
        if unit.to_string() in ['', 'dimensionless']:
            return ['dimensionless']
        else:
            physical_types = [un.physical_type for un in unit.bases]
            if all([pt=='time' for pt in physical_types]):
                # then we're time/time, eg. s / yr
                # NOTE: doing same units on numerator and denominator will force this to be dimensionless
                # and then we won't know that its time at all.
                return ['d / yr', 'h / d', 'h / yr', 'min / d', 'min / h', 'min / yr', 's / min', 's / h', 's / d', 's / yr']
            elif physical_types == ['power', 'length']:
                # then assume flux units
                return ['W / m2', 'W / km2', 'W / pc2', 'W / AU2', 'W / solRad2', 'solLum / m2', 'solLum / km2', 'solLum / pc2', 'solLum / AU2', 'solLum / solRad2']
            else:
                return [unit.to_string()]
    elif physical_type == 'time':
        return ['yr', 'd', 'h', 'min', 's']
    elif physical_type == 'mass':
        return ['solMass', 'jupiterMass', 'earthMass', 'kg', 'g']
    elif physical_type == 'length':
        return ['solRad', 'jupiterRad', 'earthRad', 'AU', 'pc', 'km', 'm', 'cm', 'mm', 'um', 'nm', 'angstrom']
    elif physical_type == 'area':
        return ['solRad2', 'jupiterRad2', 'earthRad2', 'AU2', 'pc2', 'km2', 'm2', 'cm2']
    elif physical_type == 'volume':
        return ['solRad3', 'jupiterRad3', 'earthRad3', 'AU3', 'pc3', 'km3', 'm3', 'cm3']
    elif physical_type == 'speed':
        return ['solRad / d', 'solRad / s', 'km / h', 'km / s', 'm / h', 'm / s']
    elif physical_type == 'angle':
        if unit.to_string() == 'cycle':
            return ['cycle']
        return ['rad', 'deg']
    elif physical_type == 'angular speed':
        return ['rad / yr', 'rad / d', 'rad / h', 'rad / s', 'deg / yr', 'deg / d', 'deg / h', 'deg / s']
    elif physical_type == 'temperature':
        # astropy cannot convert to u.deg_C or u.imperial.deg_F
        # could support with equivalencies: https://docs.astropy.org/en/stable/units/equivalencies.html#temperature-equivalency
        # but then we'd need to manually pass that within PHOEBE whenever
        # the physical type is temperature
        return ['K']
    elif physical_type == 'power':
        return ['W', 'solLum']
    else:
        return [unit.to_string()]
