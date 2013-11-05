"""
Compute magnetic fields.

**Main conventions**

"""
import numpy as np

def get_dipole(time, rs_center, r_polar, beta, phi0, B_polar):
    r"""
    Compute the magnetic field vectors of an oblique magnetic dipole field.
    
    The magnetic field moment is defined as:
    
    .. math::
    
        \vec{B}(\vec{r}) = \frac{B_p}{2} [3(\vec{m}\cdot\vec{r})\vec{r} - \vec{m}]
        
    where we defined :math:`\vec{r}` as the radius vectors of the surface
    elements, normalised to a reference (polar) radius :math:`R_p`. The field is
    scaled according to its polar value :math:`B_p`. The magnetic moment :math:`\vec{m}`
    is given by
    
    .. math::
    
        \vec{m} = \left(\begin{array}{c} \sin\beta\cos\phi_0 \\ \sin\beta\sin\phi_0 \\ \cos\beta \end{array}\right)
    
    :math:`\beta` is the obliquity angle, and :math:`\phi_0` the phase angle of the
    oblique field.
    
    Conventions:
    
    All images are made at :math:`t=0` unless stated otherwise:
    
    +----------------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
    | ::                                                                   | ::                                                                   | ::                                                                   |
    |                                                                      |                                                                      |                                                                      |
    |     Bpolar 1.0    G Polar magnetic field strength                    |     Bpolar 1.0    G Polar magnetic field strength                    |     Bpolar 1.0    G Polar magnetic field strength                    |
    |     beta   0.0  deg Magnetic field angle wrt rotation axis           |     beta   45.0 deg Magnetic field angle wrt rotation axis           |     beta   45.0 deg Magnetic field angle wrt rotation axis           |
    |     phi0   90.0 deg Phase angle of magnetic field                    |     phi0   90.0 deg Phase angle of magnetic field                    |     phi0   0.0  deg Phase angle of magnetic field                    |
    |                                                                      |                                                                      |                                                                      |
    +----------------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
    |                                                                      |                                                                      |                                                                      |
    | .. image:: images/atmospheres_magfield_convention01.png              | .. image:: images/atmospheres_magfield_convention02.png              | .. image:: images/atmospheres_magfield_convention03.png              |
    |    :scale: 40%                                                       |    :scale: 40%                                                       |    :scale: 40%                                                       |
    |    :align: center                                                    |    :align: center                                                    |    :align: center                                                    |
    |                                                                      |                                                                      |                                                                      |
    |                                                                      |                                                                      |                                                                      |
    +----------------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
    
    See, e.g. `Magnetic dipole <http://en.wikipedia.org/wiki/Magnetic_dipole>`_.
    
    See also how to generate :py:func:`Stokes profiles <phoebe.backend.observatory.stokes>`.
    
    """
    r_ = rs_center / r_polar
    m_ = np.array([np.sin(beta) * np.cos(phi0) - 0.0*np.sin(phi0),
                       np.sin(beta) * np.sin(phi0) + 0.0*np.cos(phi0),
                       np.cos(beta)])
    dotprod = np.dot(m_, r_.T).reshape(-1, 1)
    B =     (3*dotprod    *r_ - m_)
    B = B / 2.0 * B_polar
    
    return B