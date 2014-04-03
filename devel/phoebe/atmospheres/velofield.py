"""
Compute velocity fields.

.. autosummary::

    get_macroturbulence
    get_meridional
    
"""
from phoebe.utils import coordinates
import numpy as np
import logging

logger = logging.getLogger("ATM.VELOFIELD")

def get_macroturbulence(normal, vmacro_rad=0.0, vmacro_tan=0.0):
    """
    Compute macroturbulent velocity field.
    
    In order to get it similar to tools.broadening_macroturbulent, I
    suspiciously need to divide my vmacro with pi/sqrt(2)....
    
    """
    # Compute the local coordinate frames that have one axis as the
    # normal on each surface element, one that is horizontal (wrt
    # the z-axis) and one vertical.
    x, y, z = normal.T
    n = np.sqrt(x**2+y**2)
    horizontal = np.array([y/n, -x/n, np.zeros_like(x)]).T
    vertical = np.array([z*x/n, z*y/n, -(x**2+y**2)/n]).T
    vmacro_radial = vmacro_rad
    vmacro_tanx = vmacro_tan/np.sqrt(2)
    vmacro_tany = vmacro_tan/np.sqrt(2)
    
    # Then generate randomly distributed Gaussian velocities for
    # each direction
    np.random.seed(1111)
    vmacro = np.random.normal(size=normal.shape)
    vmacro[:,0] *= vmacro_radial
    vmacro[:,1] *= vmacro_tanx
    vmacro[:,2] *= vmacro_tany
    
    # And convert them from the local reference frame to the global
    # one
    vmacro = vmacro[:,0][:,None] * normal + \
            vmacro[:,1][:,None] * horizontal + \
            vmacro[:,2][:,None] * vertical
    
    logger.info("Added macroturbulent velocity field with vR,vT={},{}".format(vmacro_rad,vmacro_tan))
    return vmacro
    
    
def get_meridional(center, radius=1.0, inner_radius=0.5, vmeri_ampl=1.0, wc=0.2,
                   angle=0.0):
    r"""
    Compute velocity field due to meridional circulation.
    
    Inspired upon [Mitra2011]_:
    
    .. math::
    
        U_r^\mathrm{circ} & = v g(r) \frac{1}{\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta\psi)\\
        U_\theta^\mathrm{circ} & = -v g(r) \frac{1}{r}\frac{\partial}{\partial r}(r\psi)\\
        U_\phi^\mathrm{circ} & = 0
    
    with
    
    .. math::
    
        \psi & = \frac{f(r)}{r}\sin^2(\theta-\theta_1)\cot\theta\\
        f(r) & = (r-r_2)(r-r_1)^2\\
        g(r) & = \frac{1}{2} \left[ 1 - \tanh\left(\frac{r-r_2}{w_\mathrm{circ}}\right)\right]
    
    Note: the absolute magnitude of the theta component at the solar surface is
    about 10 to 20 m/s.
    
    **Example usage**:
    
    ::

        # Define parameters
        theta1 = 0.0
        r1 = 0.5
        r2 = 1.0
        theta = np.linspace(theta1, np.pi/2,20)
        r = np.linspace(r1, r2, 20)
        
        # Define location of surface
        center = np.column_stack([np.cos(np.pi/2-theta), np.zeros_like(np.pi/2-theta), np.sin(np.pi/2-theta)])

        # Cycle over different depths, compute the vector field and plot
        for radius in np.linspace(r1+1e-3, r2, 20):
            vmeri = velofield.get_meridional(center, radius=radius, inner_radius=r1, vmeri_ampl=1.0, wc=0.2).T

            plt.subplot(111, aspect='equal')
            plt.quiver(center[:,0]*radius, center[:,2]*radius,  vmeri[0],  vmeri[2], color='k')
            plt.quiver(center[:,0]*radius,-center[:,2]*radius,  vmeri[0], -vmeri[2], color='k')
        
        plt.xlabel("Inner radius")
        plt.ylabel("Inner radius")
        plt.show()
        
    +----------------------------------------------------------------------+
    |                                                                      |
    | .. image:: images/atmospheres_velofield_meri01.png                   |
    |    :scale: 40%                                                       |
    |    :align: center                                                    |
    |                                                                      |
    |                                                                      |
    +----------------------------------------------------------------------+
    
    
    :parameter inner_radius: inner radius of the circulation cell
    :type inner_radius: float
    :parameter vmeri_ampl: magnitude of circulation speed (Rsol/d)
    :type vmeri_ampl: float
    :parameter radius: relative radius at which to intersect with the velocity field
    :type radius: float
    :parameter wc: effective depth of penetration outside of radius
    """
    # Convert from Cartesian to spherical coordinates
    index = np.array([1, 0, 2])
    index_inv = np.array([1,0,2])
    r, phi, ctheta = coordinates.cart2spher_coord(*center.T[index])
    
    # Scale radius to requested location wrt r2
    r = r*radius
    
    theta = np.pi/2.0 - ctheta
    
    # Define helper functions
    r1 = inner_radius
    theta1 = angle # wedge angle
    r2 = 1.0 # this is the outer radius
    dtheta = theta - theta1
    sintheta = np.sin(theta)
    sindtheta = np.sin(theta)
    gr = 0.5*(1 - np.tanh( (r-r2)/wc))
    fr = (r-r2)*(r-r1)**2
    
    # Compute spherical velocity components
    ur = vmeri_ampl * gr / sintheta * fr/r * (2*sindtheta*np.cos(dtheta) * np.cos(theta) - sindtheta**2 * sintheta)
    utheta = -vmeri_ampl * gr * 1.0/r * sindtheta**2 / np.tan(theta) * (3*r**2 - 4*r*r1 + r1**2 - 2*r2*r + 2*r1*r2)
    uphi = np.zeros_like(ur)
    
    # Convert to Cartesian vectors
    position =  (r, phi, ctheta)
    direction = (ur, uphi, -utheta)
    vmeri = np.array(coordinates.spher2cart(position, direction))[index_inv].T
    vmeri *= 20
    
    # Solve for zero vectors
    vmeri[np.isnan(vmeri)] = 0.0
    vmeri *= -1
    
    
    #print np.sqrt(vmeri[0]**2 + vmeri[1]**2 + vmeri[2]**2).max()
    
    # That's it!
    return vmeri
    