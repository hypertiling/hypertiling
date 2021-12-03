import numpy as np

def minor(M, i, j):
    M = np.delete(M, i, 0)
    M = np.delete(M, j, 1)
    return M


def unit_circle_inversion(z):
    denom = z.real**2 + z.imag**2
    return complex(z.real/denom, z.imag/denom)


def circle_through_three_points(z1,z2,z3, verbose=True):
    x1 = z1.real
    y1 = z1.imag
    x2 = z2.real
    y2 = z2.imag
    x3 = z3.real
    y3 = z3.imag
    
    a1 = np.array([0, 0, 0, 1])
    a2 = np.array([x1*x1+y1*y1, x1, y1, 1])
    a3 = np.array([x2*x2+y2*y2, x2, y2, 1])
    a4 = np.array([x3*x3+y3*y3, x3, y3, 1])
    
    A = np.stack([a1,a2,a3,a4])
    
    M00 = np.linalg.det(minor(A,0,0))
    M01 = np.linalg.det(minor(A,0,1))
    M02 = np.linalg.det(minor(A,0,2))
    M03 = np.linalg.det(minor(A,0,3))
    
    if np.abs(M00) < 1e-10:
        if verbose:
            print("Error: Points are collinar!")
        return complex(0,0), -1

    x0 = 0.5 * M01 / M00
    y0 = - 0.5 * M02 / M00
    radius = np.sqrt(x0*x0 + y0*y0 + M03 / M00)
    
    return complex(x0,y0), radius


def compute_midpoint(z1,z2,zc):
    ax = z1.real-zc.real
    ay = z1.imag-zc.imag
    bx = z2.real-zc.real
    by = z2.imag-zc.imag

    angle = np.arctan2(by,bx) - np.arctan2(ay,ax)

    if angle < 0:
        angle = 2*np.pi + angle

    xm = zc.real + ax*np.cos(angle/2) - ay*np.sin(angle/2)
    ym = zc.imag + ax*np.sin(angle/2) + ay*np.cos(angle/2)
    
    return complex(xm,ym)


def geodesic_midpoint(z1,z2):
    z3 = unit_circle_inversion(z1)
    zc, radius = circle_through_three_points(z1,z2,z3)
    
    zm = compute_midpoint(z1,z2,zc)
    if np.abs(zm) > 1:
        zm = compute_midpoint(z2,z1,zc)
    
    return zm


def geodesic_angles(z1,z2):
    z3 = unit_circle_inversion(z1)
    zc, radius = circle_through_three_points(z1,z2,z3)    
    
    if radius == -1:
        return 0,0,0,-1

    ax = z1.real-zc.real
    ay = z1.imag-zc.imag
    bx = z2.real-zc.real
    by = z2.imag-zc.imag

    angle1 = np.arctan2(by,bx)
    angle2 = np.arctan2(ay,ax)
    
    return angle1, angle2, zc, radius


import matplotlib.patches as mpatches

def geodesic_arc(z1,z2,**kwargs):
    t1, t2, zc, r = geodesic_angles(z1,z2)
    
    if r == -1:
        return mpatches.Arrow(z1.real, z1.imag, (z2-z1).real, (z2-z1).imag, width=0, **kwargs)
    
    if t1 < 0:
        t1 = 2*np.pi + t1
            
    if t2 < 0:
        t2 = 2*np.pi + t2
        
    t = np.sort([t1,t2])
    
    t1 = t[0]
    t2 = t[1]
    
    dt1 = t2-t1
    dt2 = t1-t2+2*np.pi
    
    if dt1<dt2:
        return mpatches.Arc([zc.real,zc.imag], 2*r, 2*r, 0, theta1=np.degrees(t1), theta2=np.degrees(t2), **kwargs)
    else:
        return mpatches.Arc([zc.real,zc.imag], 2*r, 2*r, 0, theta1=np.degrees(t2), theta2=np.degrees(t1), **kwargs)