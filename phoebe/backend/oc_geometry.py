import numpy as np

def nekmin(omega_in,q,x0=0.5,y0=0.05,z0=0.5):
    
    '''Computes the position of the neck (minimal radius) in an overcontact envelope'''
    
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
            
    xy,y = x0,y0
    xz,z = x0,z0
    dxy, dy = 1.,1.
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
                    print "ok"
                    DNINV = 1./(a*d-b*c)*np.array([[d,(-1)*b],[(-1)*c,d]])
                    #DNINV = inv(DN)
                    
                    dd = np.dot(DNINV,EN)
                    dxz,dz = dd[0],dd[1]
                    print dxz,dz
                    xz=xz+dxz
                    z=z+dz
            else:
                    print "not ok"
                    xz = xz+0.5
                    z = z+0.5
                    dxz = 1.
                    dz = 1.
    
    # find solution in xy plane
    while abs(dxy)>1e-8 and abs(dy)>1e-8:
            
            dely = 1.
            y=0.05
            while abs(dely) > 0.000001:
                    delom = omega_in - Omega_xy(q,xy,y)
                    dely = delom/dOmegady(q,xy,y)
                    y = abs(y+dely)

            DN = np.array([[dOmegadx_y(q,xy,y),dOmegady(q,xy,y)],[d2Omegadx2_y(q,xy,y),d2Omegadxdy(q,xy,y)]])
            EN = np.array([omega_in-Omega_xy(q,xy,y),(-1)*dOmegadx_y(q,xy,y)])
            
            a,b,c,d = DN[0][0],DN[0][1],DN[1][0],DN[1][1]
            
            if (a*d-b*c)!=0.:
                    print "ok"
                    DNINV = 1./(a*d-b*c)*np.array([[d,(-1)*b],[(-1)*c,d]])
                    #DNINV = inv(DN)
                    
                    dd = np.dot(DNINV,EN)
                    dxy,dy = dd[0],dd[1]
                    print dxy,dy
                    xy=xy+dxy
                    y=y+dy
            else:
                    print "not ok"
                    xy = xy+0.5
                    y = y+0.5
                    dxy = 1.
                    dy = 1.    
            
    return xy,xz,y,z
        
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
