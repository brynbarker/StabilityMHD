import numpy as np
import matplotlib.pyplot as plt
from stablab.contour import winding_number

def root_solver2(box,tol,p,s,e,m,c):
    # out = root_solver2(box,p,s,e,m,c)
    #
    # Subdivides boxes until boxes containing roots are within tolerance.
    # Always subdivides vertically. Meant to be used for finding real roots.
    # The input box = [a b c] is a box with lower left coordinate (a,b)
    # and upper right coordinate (c,-b). The input tol is a bound on the
    # relative tolerance of the location of the roots assuming they are real.
    # The inputs p, s, e, m, and c are the standard STABLAB structures.
    #
    # Alternatively, if c.moments = 'on', then the program subdivides boxes
    # until there is only one root in the box and then computes the moment to
    # determine the location of the root. When there is only one root inside a
    # box, computing the moments generally provides a good approximation.

    # Future improvements: Make it so the program re-uses the contours already
    # computed. Also, make it so it doesn't store contours if not necessary.

    box = np.concatenate((box[0:3], [-box[1]]))

    # Initialize the boxes.
    # Makes box_s into a nested list
    box_s = [[None,None,None,None,None]]
    rts = []

    box_s[0][0] = cont_box(box,c) # box
    box_s[0][1] = box # box coordinates
    num_boxes = 1
    location = 0
    num_rts = 0
    box_s[0][2] = -1 # box number of parent

    while location < num_boxes:

        # Don't evaluate a box if we already know there are no roots in it
        parent_id = box_s[location][2]
        if parent_id >= 0:
            if box_s[parent_id][3] == box_s[parent_id][4]:
                location = location + 1
                continue

        # plot the box that we are currently computing
        if 'pic_stats' in c and c['pic_stats'] == 'on':
            #hold on #Plot
            corners = get_corners(box_s[location][1])
            plt.plot(np.real(corners),np.imag(corners),'.-m')
            plt.pause(0.01)

        # Evans function call
        D,domain = c['root_fun'](box_s[location][0],c,s,p,m,e)

        # winding number processing
        wnd = winding_number(D)
        box_s[location][3] = wnd; # winding number of this box
        box_s[location][4] = 0; # number of roots found in children of this box
        if parent_id >= 0:
           box_s[parent_id][4] = box_s[parent_id][4] + wnd

        # find roots or subdivide box
        if wnd > 0:

            # box coordinates
            aa = box_s[location][1][0]
            bb = box_s[location][1][1]
            cc = box_s[location][1][2]
            dd = box_s[location][1][3]

            # use the method of moments to find root when wnd = 1
            if 'moments' in c and c['moments'] == 'on':
                if wnd == 1:
                    num_rts = num_rts+1
                    rts.append(0)
                    rts[num_rts-1] = moments_roots(domain,D)
                    if 'pic_stats' in c and c['pic_stats'] == 'on':
                         plt.plot(np.real(rts),np.imag(rts),'.k')
                         plt.show()

                    location = location + 1
                    continue

            # use the relative size of width of box to find root(s) when within
            # error bounds
            if np.linalg.norm(aa-cc)/min(np.linalg.norm(aa),np.linalg.norm(cc)) < tol:
                # find the multiplicity of the root within error bounds
                for qind in range(wnd):
                    num_rts = num_rts+1
                    rts.append(0)
                    rts[num_rts-1] = 0.5*(aa+cc)+0.5*(bb+dd)*1j
                if 'pic_stats' in c and c['pic_stats'] == 'on':
                     plt.plot(np.real(rts),np.imag(rts),'.k')
                     plt.show()

                location = location + 1
                continue

            # if wnd > 1, divide boxes
            box_s.append([0,0,0,0,0])
            num_boxes = num_boxes + 1
            temp_box = [aa,bb,0.5*(aa+cc),dd]
            box_s[num_boxes-1][0] = cont_box(temp_box,c)
            box_s[num_boxes-1][1] = temp_box
            box_s[num_boxes-1][2] = location

            box_s.append([0,0,0,0,0])
            num_boxes = num_boxes + 1
            temp_box = [0.5*(aa+cc),bb,cc,dd]
            box_s[num_boxes-1][0] = cont_box(temp_box,c)
            box_s[num_boxes-1][1] = temp_box
            box_s[num_boxes-1][2] = location

        location = location + 1

    return rts

def root_solver1(box, tol, p, s, e, m, c):
    """
     Subdivides boxes until boxes containing roots are within tolerance.
     The input box = [a b c d] is a box with lower left coordinate (a,b)
     and upper right coordinate (c,d). The input tol is a bound on the
     relative tolerance of the location of the roots. The inputs p, s, e, m,
     and c are the standard STABLAB structures.

     Alternatively, if c.moments = 'on', then the program subdivides boxes
     until there is only one root in the box and then computes the moment to
     determine the location of the root. When there is only one root inside a
     box, computing the moments generally provides a good approximation.

     Future improvements: Make it so the program re-uses the contours already
     computed. Also, make it so it doesn't store contours if not necessary.
    """

    #initialize the boxes.
    # Makes box_s into a nested list
    box_s = [[None,None,None,None,None]]
    rts = []
    # box
    box_s[0][0] = cont_box(box,c)
    # box coordinates
    box_s[0][1] = box
    # box number of parent (this box doesn't have a parent, whence assigned -1)
    box_s[0][2] = -1

    # Variables assisting in iterating through the following while loop
    num_boxes = 1
    location = 0
    num_rts = 0

    while location < num_boxes:

        # Don't evaluate a box if we already know there are no roots in it
        parent_id = box_s[location][2]
        if parent_id >= 0:
            if box_s[parent_id][3] == box_s[parent_id][4]:
               location = location + 1
               continue

        # plot the box that we are currently computing
        if 'pic_stats' in c and c['pic_stats'] == 'on':
            corners = get_corners(box_s[location][1])
            plt.plot(np.real(corners),np.imag(corners),'.-m')
            #plt.show()
            plt.pause(0.00001) # For some reason, this works to display the box

        # Evans function call
        D,domain = c.root_fun(box_s[location][0],c,s,p,m,e)

        # winding number processing
        wnd = winding_number(D)

        # Winding number of this box
        box_s[location][3] = wnd

        # Number of roots found in children of this box
        box_s[location][4] = 0
        if parent_id >= 0:
            box_s[parent_id][4] = box_s[parent_id][4] + wnd

        # find roots or subdivide box
        if wnd > 0:

            # box coordinates
            aa = box_s[location][1][0]
            bb = box_s[location][1][1]
            cc = box_s[location][1][2]
            dd = box_s[location][1][3]

            # use the method of moments to find root when wnd = 1
            if 'moments' in c:
                if c['moments'] == 'on':
                    if wnd == 1:
                        do_solve = 1
                        if 'moments_tol' in c:
                            if c['moments_tol'] == 'on':
                                print()
                                print("inside c.moments_tol == on")
                                print(norm([aa,bb]-[cc,dd]))
                                print()
                                if norm([aa,bb]-[cc,dd]) > tol:
                                    do_solve = 0;

                        if do_solve == 1:
                            box_s.append([None,None,None,None,None])
                            num_rts = num_rts+1
                            rts.append(0)
                            rts[num_rts-1] = moments_roots(domain,D)
                            if 'pic_stats' in c and c['pic_stats'] == 'on':
                                #Draw the real and imaginary parts of rts.
                                plt.plot(np.real(rts),np.imag(rts),'.k')
                                plt.pause(0.00001)
                            location = location + 1
                            continue

            # use the relative size of diagonal of box to find root when wnd = 1
            array1 = np.linalg.norm(np.array([aa-cc,bb-dd]))
            array2 = np.linalg.norm(np.array([aa,bb]))
            array3 = np.linalg.norm(np.array([cc,dd]))
            if array1/min(array2,array3) < tol:
                box_s.append([None,None,None,None,None])
                num_rts = num_rts+1
                rts.append(0)
                rts[num_rts-1] = 0.5*(aa+cc)+0.5*(bb+dd)*1j
                if 'pic_stats' in c and c['pic_stats'] == 'on':
                    #Plot the real and imaginary parts of rts.
                    plt.plot(np.real(rts),np.imag(rts),'.k')
                    plt.plot()

                location = location + 1

                continue

            # if wnd > 0, subdivide the current box into 4 boxes contained
            #  within it
            box_s.append([0,0,0,0,0])
            num_boxes = num_boxes + 1
            temp_box = [aa,bb,0.5*(aa+cc),0.5*(bb+dd)]
            box_s[num_boxes-1][0] = cont_box(temp_box,c)
            box_s[num_boxes-1][1] = temp_box
            box_s[num_boxes-1][2] = location

            box_s.append([0,0,0,0,0])
            num_boxes = num_boxes + 1
            temp_box = [0.5*(aa+cc),bb,cc,0.5*(bb+dd)]
            box_s[num_boxes-1][0] = cont_box(temp_box,c)
            box_s[num_boxes-1][1] = temp_box
            box_s[num_boxes-1][2] = location

            box_s.append([0,0,0,0,0])
            num_boxes = num_boxes + 1
            temp_box = [0.5*(aa+cc),0.5*(bb+dd),cc,dd]
            box_s[num_boxes-1][0] = cont_box(temp_box,c)
            box_s[num_boxes-1][1] = temp_box
            box_s[num_boxes-1][2] = location

            box_s.append([0,0,0,0,0])
            num_boxes = num_boxes + 1
            temp_box = [aa,0.5*(bb+dd),0.5*(aa+cc),dd]
            box_s[num_boxes-1][0] = cont_box(temp_box,c)
            box_s[num_boxes-1][1] = temp_box
            box_s[num_boxes-1][2] = location

        location = location + 1

    return rts

def cont_box(box,c):
    # make contour out of box coordinates
    line1 = np.linspace(box[0],box[2],10+9*c.ksteps+(10+9*c.ksteps-1)*c.lambda_steps)+box[1]*1j
    line2 = np.linspace(box[1],box[3],10+9*c.ksteps+(10+9*c.ksteps-1)*c.lambda_steps)*1j+box[2]
    line3 = (np.linspace(box[0],box[2],10+9*c.ksteps+(10+9*c.ksteps-1)*c.lambda_steps))+box[3]*1j
    line4 = (np.linspace(box[1],box[3],10+9*c.ksteps+(10+9*c.ksteps-1)*c.lambda_steps))*1j+box[0]

    # Reverse lines 3 and 4
    line3 = line3[::-1]
    line4 = line4[::-1]

    #Concatenate the results
    out = np.concatenate((line1, line2[1:], line3[1:], line4[1:]))
    return out


def get_corners(cont):
    """
    prep box coordinates for plotting
    """
    out = [cont[0]+1j*cont[1], cont[2]+1j*cont[1], cont[2]+1j*cont[3],
           cont[0]+1j*cont[3], cont[0]+1j*cont[1]]
    return out

def moments_roots(domain,range):
    #
    # out=moments_roots(domain, range)
    #
    # Uses the method of moments to find the roots of the analytic function
    # F(lambda) evaluated at the points "domain" of the simple, positively
    # oriented contour Gamma. Here range=F(domain). The results become less
    # accurate generally as the number of roots contained in the contour
    # increase. If the contour has winding number of zero, then the function
    # throws an error. Generally, it is better to not use this when more than
    # one root is inside the contour because of the numerical error.

    # NOTE: Todd Kapitula's student studied the method of moments and came up
    # with an improvement by integrating differently.

    # Created: 17 Dec 2010
    # Last updated: 1 Jan 2011

    # make sure the dimensions are correct for the moments function.

    # In python, np.shape() will return a 1-tuple if domain has only one
    #  dimension. This try except block will handle that case.
    """
    try:
        sx,sy = np.shape(domain)
    except:
        sx = np.shape(domain)
        sy = 0
    if sx > sy:
       domain = np.transpose(domain)

    try:
        sx,sy = np.shape(range)
    except:
        sx = np.shape(range)
        sy = 0
    if sx > sy:
       range=np.transpose(range)
    """
    # find the winding number of the domain contour
    wnd = round(winding_number(range))
    out = np.zeros((wnd),dtype=np.complex)
    # if wnd < 4, solve algebrically for the roots using closed formulas
    # Case when there is one root
    if wnd == 1:
        out = (moments(domain,range,1,0)/(2*np.pi*1j))
    # Case when there are two roots
    elif wnd == 2:
        mu1 = moments(domain,range,1,0)/(4*np.pi*1j)
        mu2 = moments(domain,range,2,mu1)/(2*np.pi*1j)
        out[0] = mu1+np.sqrt(mu2/2)
        out[1] = mu1-np.sqrt(mu2/2)
    # Case when there are three roots
    elif wnd == 3:
        mu1 = moments(domain,range,1,0)/(2*np.pi*1j) #a+b+c
        mu2 = moments(domain,range,2,0)/(2*np.pi*1j) #a**2+b**2+c**2
        mu3 = moments(domain,range,3,0)/(2*np.pi*1j) #a**3+b**3+c**3
        r1,r2,r3 = np.roots(np.array([1,-mu1,.5*(mu1**2-mu2),
                              (-1/6)*(mu1**3-3*mu1*mu2+2*mu3)]))
        out[0] = r1
        out[1] = r2
        out[2] = r3

    return out

def moments(z,w,pow,mu=0,order=4):
    """
    # Integrates f'(z)/f(z)*z**pow along a contour given by z where w = f(z).
    # The moments are calculated about the complex constant mu. The order of
    # the derivative difference approximation may be specified as 2 or 4 and
    # defaults to 4.

    # NOTE: This is not state of the art code. Using analyticity of the Evans
    # function, we can do much better by using Chebyshev interpolation
    """
    """ Not necessary since z,w should be 1-dimensional arrays anyway
    sx,sy = np.shape(z)
    if sx > sy:
        z = z.T;
    sx,sy = np.shape(w)
    if sx > sy:
        w = w.T;
    """
    # Remove any repeat points from contours
    aind = 0
    bind = 1
    count = 0
    index = []

    while bind < len(z):
        if abs(z[aind]-z[bind]) > 10**-14:
            index.append(aind)
            count = count+1
            aind = bind
            bind = bind+1
        else:
            bind = bind+1

    z = z[index]
    w = w[index]

    # Make sure the contour has an odd number of points
    if (len(w) % 2) == 0:
       z = np.concatenate((z[:-1], [0.5*(z[-2]+z[-1])], [z[-1]]))
       w = np.concatenate((w[:-1], [0.5*(w[-2]+w[-1])], [w[-1]]))

    if order == 2:
        h2 = np.diff(z)
        h1 = np.array([0, -1*np.diff(z[:-1])])

        n = len(z)
        intgrand = np.zeros(n,dtype=np.complex)
        #Here we approximate and store the values of z**pow*f'(z)/f(z)
        for k in range(1,n-1):
            intgrand[k] = ((z[k]-mu)**pow/w[k])*(1/((h2[k]-h1[k])*h1[k]*h2[k]))*(h2[k]**2*w[k-1]-h1[k]**2*w[k+1]+(h1[k]**2-h2[k]**2)*w[k])

        intgrand[0] = ((z[0]-mu)**pow/w[0])*((w[1]-w[0])/(z[1]-z[0]))
        intgrand[-1] = ((z[-1]-mu)**pow/w[-1])*((w[-1]-w[-2])/(z[-1]-z[-2]))

    elif order == 4:
        longer_z = np.concatenate([z[-3:-1], z, z[1:3]])
        longer_w = np.concatenate([w[-3:-1], w, w[1:4]])
        n = len(z)
        fprime = np.zeros(n,dtype=np.complex)
        for j in range(n):
            mid = j+2
            h1 = longer_z[mid+2]-longer_z[mid]
            h2 = longer_z[mid+1]-longer_z[mid]
            h3 = longer_z[mid-1]-longer_z[mid]
            h4 = longer_z[mid-2]-longer_z[mid]
            vec = np.array([
                -h3 * h2 * h4 / h1 / (-h4 + h1) / (-h3 + h1) / (-h2 + h1),
                h4 * h1 * h3 / (-h2 + h1) / h2 / (-h4 + h2) / (-h3 + h2),
                -h1 * h2 * h4 / (h3 ** 2 - h1 * h3 - h2 * h3 + h1 * h2) / h3 /
                    (-h4 + h3),
                h2 * h1 * h3 / h4 / (-h4 ** 3 - h2 * h1 * h4 + h1 * h4 ** 2 +
                    h2 * h4 ** 2 + h3 * h4 ** 2 - h3 * h1 * h4 - h3 * h2 * h4 +
                    h2 * h1 * h3)
                ])
            fprime[j] = (vec[0]*longer_w[mid+2]+vec[1]*longer_w[mid+1]+vec[2]*
                         longer_w[mid-1]+vec[3]*longer_w[mid-2]-longer_w[mid]*
                         (np.sum(vec)))

        intgrand = np.zeros(n,dtype=np.complex)
        # Here we approximate and store the values of z**pow*f'(z)/f(z)
        for k in range(n):
            intgrand[k] = ((z[k]-mu)**pow/w[k])*fprime[k]

    #Simpson Integration
    y = 0
    for j in range(0,n-2,2):
        x0 = z[j]
        x1 = z[j+1]
        x2 = z[j+2]

        a = intgrand[j]/( (x0-x1)*(x0-x2) )
        b = intgrand[j+1]/( (x1-x0)*(x1-x2) )
        c = intgrand[j+2]/( (x2-x0)*(x2-x1) )

        SUM = ( (x2**3- x0**3)*(a + b + c)/3 + (x0**2- x2**2)* ( a*(x1+x2) +
                b*(x0 + x2) + c*(x0+x1) )/2 + (x2-x0)*(a*x1*x2 + b*x0*x2 +
                c*x0*x1) )

        y = y + SUM

    return y

def evan_root_bisector(fun,a,b,tol):
    """
    Returns the root of the Evans function with absolute tolerance tol

    Input is a left and right endpoint bounding the root, the tolerance, and
    a function handle that takes as input a and b and returns the values of
    the Evans function evaluated at the points a, b, and 0.5(a+b), as well
    as the points. [f,dom] = fun(a,b), where f = [D(a),D(0.5*(a+b)),D(b)],
    and dom = [a,0.5*(a+b),b];
    """

    # compute the Evans function on the points [a,0.5*(a+b),b].
    f,dom = fun(a,b)
    if f[0]*f[2] >= 0:
        print("dom:",dom,'\n')
        print("f:",f,'\n')
        raise ValueError('Root is not bracketed.')

    # update the interval based on compute values
    if np.sign(f[0]) == np.sign(f[1]):
        a = dom[1]
    elif np.sign(f[1]) == np.sign(f[2]):
        b = dom[1]
    else:
        out = dom[1]
        return out

    # if width of interval is smaller than tolerance, return midpoint
    if abs(a-b) < tol:
       out = dom[1]
       return out

    # recursion step
    out = evan_root_bisector(fun,a,b,tol)
    return out
