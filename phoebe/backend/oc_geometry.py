import numpy as np
from math import sqrt, sin, cos, acos, atan2, trunc, pi

# from potentials import dBinaryRochedx,dBinaryRochedy,dBinaryRochedz as dpdx,dpdy,dpdz
def BinaryRoche (r, D, q, F, Omega=0.0):
    r"""
    Computes a value of the asynchronous, eccentric Roche potential.

    If :envvar:`Omega` is passed, it computes the difference.

    The asynchronous, eccentric Roche potential is given by [Wilson1979]_

    .. math::

        \Omega = \frac{1}{\sqrt{x^2 + y^2 + z^2}} + q\left(\frac{1}{\sqrt{(x-D)^2+y^2+z^2}} - \frac{x}{D^2}\right) + \frac{1}{2}F^2(1+q)(x^2+y^2)

    @param r:      relative radius vector (3 components)
    @type r: 3-tuple
    @param D:      instantaneous separation
    @type D: float
    @param q:      mass ratio
    @type q: float
    @param F:      synchronicity parameter
    @type F: float
    @param Omega:  value of the potential
    @type Omega: float
    """
    return 1.0/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]) + q*(1.0/sqrt((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])-r[0]/D/D) + 0.5*F*F*(1+q)*(r[0]*r[0]+r[1]*r[1]) - Omega

def dBinaryRochedx (r, D, q, F):
    """
    Computes a derivative of the potential with respect to x.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[0]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*(r[0]-D)*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 -q/D/D + F*F*(1+q)*r[0]

def d2BinaryRochedx2(r, D, q, F):
    """
    Computes second derivative of the potential with respect to x.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return (2*r[0]*r[0]-r[1]*r[1]-r[2]*r[2])/(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**2.5 +\
          q*(2*(r[0]-D)*(r[0]-D)-r[1]*r[1]-r[2]*r[2])/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**2.5 +\
          F*F*(1+q)

def dBinaryRochedy (r, D, q, F):
    """
    Computes a derivative of the potential with respect to y.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[1]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[1]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 + F*F*(1+q)*r[1]

def  dBinaryRochedz(r, D, q, F):
    """
    Computes a derivative of the potential with respect to z.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[2]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[2]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5

def dBinaryRochedr (r, D, q, F):
    """
    Computes a derivative of the potential with respect to r.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """

    r2 = (r*r).sum()
    r1 = np.sqrt(r2)

    return -1./r2 - q*(r1-r[0]/r1*D)/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**1.5 - q*r[0]/r1/D/D + F*F*(1+q)*(1-r[2]*r[2]/r2)*r1

class MeshVertex:
    def __init__(self, r, dpdx, dpdy, dpdz, *args):
        """
        """
        # Normalized normal:
        #~ n = np.array([dpdx(r, *args), dpdy(r, *args), dpdz(r, *args)])
        #~ n /= np.sqrt(np.sum(n*n))
        nx = dpdx(r, *args)
        ny = dpdy(r, *args)
        nz = dpdz(r, *args)
        nn = sqrt(nx*nx+ny*ny+nz*nz)
        nx /= nn
        ny /= nn
        nz /= nn

        # Now we choose the first tangential direction. We have a whole
        # plane to choose from, so we'll choose something convenient.
        # The choice here is to set one tangential coordinate to 0. From
        # n \dot t=0: nx tx + ny ty = 0 => tx = -ny, ty = nx.
        # Since we are normalizing, we need to be careful that we don't
        # divide by a small number, hence the two prescriptions, either
        # for tz = 0 or for ty = 0.

        if nx > 0.5 or ny > 0.5:
            nn = sqrt(ny*ny+nx*nx)
            t1x = ny/nn
            t1y = -nx/nn
            t1z = 0.0
        else:
            nn = sqrt(nx*nx+nz*nz)
            t1x = -nz/nn
            t1y = 0.0
            t1z = nx/nn

        t2x = ny*t1z - nz*t1y
        t2y = nz*t1x - nx*t1z
        t2z = nx*t1y - ny*t1x

        self.r = r
        self.n = np.array((nx, ny, nz))
        self.t1 = np.array((t1x, t1y, t1z))
        self.t2 = np.array((t2x, t2y, t2z))

    def __repr__(self):
        repstr  = " r = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.r[0],  self.r[1], self.r[2])
        repstr += " n = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.n[0],  self.n[1], self.n[2])
        repstr += "t1 = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.t1[0], self.t1[1], self.t1[2])
        repstr += "t2 = (% 3.3f, % 3.3f, % 3.3f)"   % (self.t2[0], self.t2[1], self.t2[2])
        return repstr

def project_onto_potential(r, pot_name, *args):
    """
    TODO: add documentation
    """

    pot = globals()[pot_name]
    dpdx = globals()['d%sdx'%(pot_name)]
    dpdy = globals()['d%sdy'%(pot_name)]
    dpdz = globals()['d%sdz'%(pot_name)]
    dpdr = globals()['d%sdr'%(pot_name)]

    n_iter = 0

    rmag, rmag0 = np.sqrt((r*r).sum()), 0
    lam, nu = r[0]/rmag, r[2]/rmag
    dc = np.array((lam, np.sqrt(1-lam*lam-nu*nu), nu)) # direction cosines -- must not change during reprojection
    D, q, F, p0 = args

    while np.abs(rmag-rmag0) > 1e-12 and n_iter < 100:
        rmag0 = rmag
        rmag = rmag0 - pot(rmag0*dc, *args)/dpdr(rmag0*dc, *args[:-1])
        n_iter += 1
    if n_iter == 100:
        logger.warning('projection did not converge')

    r = rmag*dc

    return MeshVertex(r, dpdx, dpdy, dpdz, *args[:-1])

def nekmin(omega_in,q,x0=0.5,z0=0.5):

    '''Computes the position of the neck (minimal radius) in an contact_binary star1'''

    def Omega_xz(q,x,z):
            return 1./np.sqrt(x**2+z**2)+q/np.sqrt((1-x)**2+z**2)+(q+1)*x**2/2.-q*x

    def Omega_xy(q,x,y):
            return 1./np.sqrt(x**2+y**2)+q/np.sqrt((1-x)**2+y**2)+(q+1)*(x**2+y**2)/2.-q*x

    def dOmegadx_z(q,x,z):
            return -x/(x**2+z**2)**(3./2)+q*(1-x)/((1-x)**2+z**2)**(3./2.)+(q+1)*x-q

    def dOmegadx_y(q,x,y):
            return -x/(x**2+y**2)**(3./2)+q*(1-x)/((1-x)**2+y**2)**(3./2.)+(q+1)*x-q

    def dOmegadz(q,x,z):
            return -z/(x**2+z**2)**(3./2)-q*z/((1-x)**2+z**2)**(3./2.)

    def dOmegady(q,x,y):
            return -y/(x**2+y**2)**(3./2)-q*y/((1-x)**2+y**2)**(3./2.)+(q+1)*y

    def d2Omegadx2_z(q,x,z):
            return (2*x**2-z**2)/(x**2+z**2)**(5./2)+q*(2*(1-x)**2-z**2)/((1-x)**2+z**2)**(5./2)+(q+1)

    def d2Omegadx2_y(q,x,y):
            return (2*x**2-y**2)/(x**2+y**2)**(5./2)+q*(2*(1-x)**2-y**2)/((1-x)**2+y**2)**(5./2)+(q+1)

    def d2Omegadxdz(q,x,z):
            return 3*x*z/(x**2+z**2)**(5./2)-3*q*x*(1-x)/((1-x)**2+z**2)**(5./2)

    def d2Omegadxdy(q,x,y):
            return 3*x*y/(x**2+y**2)**(5./2)-3*q*x*(1-x)/((1-x)**2+y**2)**(5./2)

    xz,z = x0,z0
    dxz, dz = 1.,1.

    # find solution in xz plane
    while abs(dxz)>1e-8 and abs(dz)>1e-8:

            delz = 1.
            z=0.05
            while abs(delz) > 0.000001:
                    delom = omega_in - Omega_xz(q,xz,z)
                    delz = delom/dOmegadz(q,xz,z)
                    z = abs(z+delz)

            DN = np.array([[dOmegadx_z(q,xz,z),dOmegadz(q,xz,z)],[d2Omegadx2_z(q,xz,z),d2Omegadxdz(q,xz,z)]])
            EN = np.array([omega_in-Omega_xz(q,xz,z),(-1)*dOmegadx_z(q,xz,z)])

            a,b,c,d = DN[0][0],DN[0][1],DN[1][0],DN[1][1]

            if (a*d-b*c)!=0.:
                    DNINV = 1./(a*d-b*c)*np.array([[d,(-1)*b],[(-1)*c,d]])
                    #DNINV = inv(DN)

                    dd = np.dot(DNINV,EN)
                    dxz,dz = dd[0],dd[1]
                    # print dxz,dz
                    xz=xz+dxz
                    z=z+dz
            else:
                    xz = xz+0.5
                    z = z+0.5
                    dxz = 1.
                    dz = 1.

    return xz,z

def compute_frac_areas(new_mesh,xmin):

    '''Computes fractional areas belonging to each triangle vertex that are a result
    of the intersection with the neck plane and assigns them to each vertex per triangle.'''

    def intersection(point1,point2,xmin):
        ''' point1 = (x1,y1,z1), point2 = (x2,y2,z2
        parameters for matrices: a1=(z2-z1)/(x2-x1), b1=0, c1=-1, d1=z1-(z2-z1)/(x2-x1)*x1
                                 a2= , b2=(z2-z1)/(y2-y1), c2=-1, d2=z1-(z2-z1)/(y2-y1)*y1
                                 a3=1, b3=0, c3=0, d3=-xmin'''

        a1 = (point2[2]-point1[2])/(point2[0]-point1[0])
        b1 = 0.
        c1 = -1.
        d1 = point1[2] - point1[0]*(point2[2]-point1[2])/(point2[0]-point1[0])

        b2 = (point2[2]-point1[2])/(point2[1]-point1[1])
        a2 = 0.
        c1 = -1.
        d1 = point1[2] - point1[1]*(point2[2]-point1[2])/(point2[1]-point1[1])

        a3, b3, c3, d3 = 1., 0., 0., (-1)*xmin


        R_c = np.array([[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]])
        R_d = np.array([[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]])

        # condition for one-point intersection
        if np.linalg.matrix_rank(R_c)==3 and np.linalg.matrix_rank(R_d)==3:
            # build the arrays whose determinants are used to compute the intersection coords
            x_arr = np.array([[d1,b1,c1],[d2,b2,c2],[d3,b3,c3]])
            y_arr = np.array([[a1,d1,c1],[a2,d2,c2],[a3,d3,c3]])
            z_arr = np.array([[a1,b1,d1],[a2,b2,d2],[a3,b3,d3]])

            x = np.linalg.det(x_arr)/np.linalg.det(R_c)
            y = np.linalg.det(y_arr)/np.linalg.det(R_c)
            z = np.linalg.det(z_arr)/np.linalg.det(R_c)

        # next we need to check whether the point is actually on the line section that makes the triangle
        x_low,x_up = min(point1[0],point2[0]),max(point1[0],point2[0])
        y_low,y_up = min(point1[1],point2[1]),max(point1[1],point2[1])
        z_low,z_up = min(point1[2],point2[2]),max(point1[2],point2[2])

        if x>x_low and x<x_up and y>y_low and y<y_up and z>z_low and z<z_up:
            return np.array(x,y,z)
        else:
            return None

    '''Used only in the Envelope class, computes triangle fractional areas
    based on the intersection with the neck plane and assigns them to each vertex
    to be used as weights in triangle temperature averaging'''

    # only works for marching now
    frac_areas = np.zeros((len(new_mesh['triangles']),3))
    frac_areas[(new_mesh['env_comp3']==0)|(new_mesh['env_comp3']==1)] = 1.

    triangles_neck = new_mesh['triangles'][(new_mesh['env_comp3']!=0)&(new_mesh['env_comp3']!=1)]

    for i in range(len(triangles_neck)):

        # take the vertex indices of each triangle and pull the vertices
        # new_mesh already has the env_comp and env_comp3 columns
        vind = triangles_neck[i]
        v1,v2,v3 = new_mesh['vertices'][vind[0]],new_mesh['vertices'][vind[1]],new_mesh['vertices'][vind[2]]
        env1,env2,env3 = new_mesh['env_comp'][vind[0]],new_mesh['env_comp'][vind[1]],new_mesh['env_comp'][vind[2]]

        # take lines formed by vertices on opposite sides of neck (different env_comp)
        # be smart about this, now it's just so it's there

        if (env1==0 and env2==0 and env3==1) or (env1==1 and env2==1 and env3==0):
            line1,line2=[v1,v3],[v2,v3]
            triangle1,triangle2,triangle3 = [v1,v2,0],[v2,0,0],[v3,0,0]
        if (env1==0 and env2==1 and env3==1) or (env1==1 and env2==0 and env3==0):
            line1,line2=[v1,v2],[v1,v3]
            triangle1,triangle2,triangle3 = [v2,v3,0],[v3,0,0],[v1,0,0]
        if (env1==0 and env2==1 and env3==0) or (env1==1 and env2==0 and env3==1):
            line1,line2=[v1,v2],[v2,v3]
            triangle1,triangle2,triangle3 = [v1,v3,0],[v3,0,0],[v2,0,0]

        # compute the points of intersection
        if line1!=None and line2!=None:
            o1 = intersection(line1[0],line1[1],xmin)
            o2 = intersection(line2[0],line2[1],xmin)

            # create the new triangles whose areas need to be computed
            # triangle1 is the one containing the two vertices of the original
            # triangle3 is the top one that is on the other side
            triangle1[2] = o1
            triangle2[1] = o1
            triangle2[2] = o2
            triangle3[1] = o1
            triangle3[2] = o2

            triangle0 = [v1,v2,v3]

            # compute the areas
            area0,area1,area2,area3 = 0.,0.,0.,0.
            for triangle,area in zip([triangle0,triangle1,triangle2,triangle3],[area0,area1,area2,area3]):
                a = np.sqrt((triangle[0][0]-triangle[1][0])**2+(triangle[0][1]-triangle[1][1])**2+(triangle[0][2]-triangle[1][2])**2)
                b = np.sqrt((triangle[0][0]-triangle[2][0])**2+(triangle[0][1]-triangle[2][1])**2+(triangle[0][2]-triangle[2][2])**2)
                c = np.sqrt((triangle[2][0]-triangle[1][0])**2+(triangle[2][1]-triangle[1][1])**2+(triangle[2][2]-triangle[1][2])**2)
                s = (a+b+c)/2.

                area = np.sqrt(s*(s-a)*(s-b)*(s-c))


            # assign fractional areas to vertices based on where they are with respect to the intersection
            # the top one gets its own fractional area, the two bottom ones get half of the remaining

            if (env1==0 and env2==0 and env3==1) or (env1==1 and env2==1 and env3==0):
				frac_areas[i]=[0.5*(area1+area2)/area0,0.5*(area1+area2)/area0,area3/area0]
            if (env1==0 and env2==1 and env3==1) or (env1==1 and env2==0 and env3==0):
                frac_areas[i]=[area1/area0,0.5*(area2+area3)/area0,0.5*(area2+area3)/area0]
            if (env1==0 and env2==1 and env3==0) or (env1==1 and env2==0 and env3==1):
                frac_areas[i]=[0.5*(area1+area3)/area0,area2/area0,0.5*(area1+area3)/area0]


def wd_mesh_fill(xmin,y1,z1,r1,r2,r3,r4):
    # a simple sorting of the rs array by the value of the x coordinate will do the trick
    # will only need to take the two vertices with highest x value -> this will also work
    # for trapezoids that are not crossing the boundary but are the last ones

    rxs = [r1[0],r2[0],r3[0],r4[0]]
    rxs.sort()

    # take the two vertices with largest x-value and shift them to the neck
    for rx in [rxs[2],rxs[3]]:
        for r in [r1,r2,r3,r4]:

            if r[0]==rx:
                r[0] = xmin

                # compute angle of point (as seen from the ellipse center) in the yz plane
                tan_theta = r[2]/r[1]
                # compute the eccentric anomaly of the point on the ellipse
                t = np.arctan(y1/z1*tan_theta)
                # project the point on the neck ellipse
                r[1] = y1*np.cos(t)
                r[2] = z1*np.sin(t)

    # compute the new triangle areas
    side1 = sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)
    side2 = sqrt((r1[0]-r3[0])**2 + (r1[1]-r3[1])**2 + (r1[2]-r3[2])**2)
    side3 = sqrt((r2[0]-r3[0])**2 + (r2[1]-r3[1])**2 + (r2[2]-r3[2])**2)
    s = 0.5*(side1 + side2 + side3)

    dsigma_t_sq = s*(s-side1)*(s-side2)*(s-side3)
    dsigma_t = sqrt(dsigma_t_sq) if dsigma_t_sq > 0 else 0.0

    # compute the fractional change in area based on the new triangle area

    return r1,r2,r3,r4,dsigma_t

def wd_recompute_neck(r0,xmin,y1,z1,r01,r02,r03,r04,theta,dtheta,potential,d,q,F,Phi):

    side1 = sqrt((r01[0]-r02[0])**2 + (r01[1]-r02[1])**2 + (r01[2]-r02[2])**2)
    side2 = sqrt((r01[0]-r03[0])**2 + (r01[1]-r03[1])**2 + (r01[2]-r03[2])**2)
    side3 = sqrt((r02[0]-r03[0])**2 + (r02[1]-r03[1])**2 + (r02[2]-r03[2])**2)

    sidemax = max(side1,side2,side3)
    if sidemax==side1:
        point1,point2 = r01,r02
    elif sidemax==side2:
        point1,point2 = r01,r03
    else:
        point1,point2 = r02,r03

    x_half, y_half = (point1[0]+point2[0])/2., (point1[1]+point2[1])/2.
    #r = np.sqrt(x_half**2+y_half**2+z_half**2)
    phi_half = np.arctan(y_half/x_half)
    phi_1 = np.arctan(point1[1]/point1[0])
    phi_2 = np.arctan(point2[1]/point2[0])

    dphi = abs(phi_2-phi_1)
    rc = np.array((r0*sin(theta)*cos(phi_half), r0*sin(theta)*sin(phi_half), r0*cos(theta)))
    vc = project_onto_potential(rc, potential, d, q, F, Phi).r

    # Next we need to find the tangential plane, which we'll get by finding the normal,
    # which is the negative of the gradient:
    dpdx,dpdy,dpdz = dBinaryRochedx,dBinaryRochedy,dBinaryRochedz
    nc = np.array((-dpdx(vc, d, q, F), -dpdy(vc, d, q, F), -dpdz(vc, d, q, F)))

    # Then we need to find the intercontext of +/-dtheta/dphi-deflected
    # radius vectors with the tangential plane. We do that by solving
    #
    #   d = [(p0 - l0) \dot n] / (l \dot n),
    #
    # where p0 and l0 are reference points on the plane and on the line,
    # respectively, n is the normal vector, and l is the line direction
    # vector. For convenience l0 can be set to 0, and p0 is just vc. d
    # then measures the distance from the origin along l.

    l1 = np.array((sin(theta-dtheta/2)*cos(phi_half-dphi/2), sin(theta-dtheta/2)*sin(phi_half-dphi/2), cos(theta-dtheta/2)))
    l2 = np.array((sin(theta-dtheta/2)*cos(phi_half+dphi/2), sin(theta-dtheta/2)*sin(phi_half+dphi/2), cos(theta-dtheta/2)))
    l3 = np.array((sin(theta+dtheta/2)*cos(phi_half+dphi/2), sin(theta+dtheta/2)*sin(phi_half+dphi/2), cos(theta+dtheta/2)))
    l4 = np.array((sin(theta+dtheta/2)*cos(phi_half-dphi/2), sin(theta+dtheta/2)*sin(phi_half-dphi/2), cos(theta+dtheta/2)))

    r1 = np.dot(vc, nc) / np.dot(l1, nc) * l1
    r2 = np.dot(vc, nc) / np.dot(l2, nc) * l2
    r3 = np.dot(vc, nc) / np.dot(l3, nc) * l3
    r4 = np.dot(vc, nc) / np.dot(l4, nc) * l4

    # This sorts out the vertices, now we need to fudge the surface
    # area. WD does not take curvature of the equipotential at vc
    # into account, so the surface area computed from these vertex-
    # delimited surfaces will generally be different from what WD
    # computes. Thus, we compute the surface area the same way WD
    # does it and assign it to each element even though that isn't
    # quite its area:
    #
    #   dsigma = || r^2 sin(theta)/cos(gamma) dtheta dphi ||,
    #
    # where gamma is the angle between l and n.

    cosgamma = np.dot(vc, nc)/np.sqrt(np.dot(vc, vc))/np.sqrt(np.dot(nc, nc))
    dsigma = np.abs(np.dot(vc, vc)*np.sin(theta)/cosgamma*dtheta*dphi)

    side1 = sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)
    side2 = sqrt((r1[0]-r3[0])**2 + (r1[1]-r3[1])**2 + (r1[2]-r3[2])**2)
    side3 = sqrt((r2[0]-r3[0])**2 + (r2[1]-r3[1])**2 + (r2[2]-r3[2])**2)
    s = 0.5*(side1 + side2 + side3)

    dsigma_t_sq = s*(s-side1)*(s-side2)*(s-side3)
    dsigma_t = sqrt(dsigma_t_sq) if dsigma_t_sq > 0 else 0.0

    #return vc,nc,r1,r2,r3,r4,dsigma,dsigma_t,phi_half,dphi
    # for now, we don't want to recompute everything, just change the area assigned to the vertex
    return dsigma
