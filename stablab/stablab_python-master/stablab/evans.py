import numpy as np
from scipy import linalg
from scipy.misc import comb
from scipy.integrate import complex_ode
from itertools import combinations
from multiprocessing import Pool

from stablab.contour import analytic_basis, projection2
from stablab.Struct import Struct
from stablab import cheby_bvp

def evans(yl,yr,lamda,s,p,m,e):
    
    if e['evans'] == "lopatinski":
        out = lopatinski(yl,yr,lamda,s,p,m,e)

    elif e['evans'] == "reg_reg_polar":
        muL = np.trace(np.dot(np.dot(np.conj((linalg.orth(yl)).T),
                            e['LA'](e['Li'][0],lamda,s,p)),linalg.orth(yl)))
        muR = np.trace(np.dot(np.dot(np.conj((linalg.orth(yr)).T),
                            e['RA'](e['Ri'][0],lamda,s,p)),linalg.orth(yr)))

        omegal,gammal = manifold_polar(e['Li'],linalg.orth(yl),lamda,e['LA'],
                                        s,p,m,e['kl'],muL)
        omegar,gammar = manifold_polar(e['Ri'],linalg.orth(yr),lamda,e['RA'],
                                        s,p,m,e['kr'],muR)

        out = (linalg.det(np.dot(np.conj(linalg.orth(yl).T),yl))*
               linalg.det(np.dot(np.conj(linalg.orth(yr).T),yr))*gammal*
               gammar*linalg.det(np.concatenate((omegal,omegar),axis=1)))

    elif e['evans'] == "adj_reg_polar":
        muL = np.trace(np.dot(np.dot(np.conj((linalg.orth(yl)).T),
                            e['LA'](e['Li'][0],lamda,s,p)),linalg.orth(yl)))
        muR = np.trace(np.dot(np.dot(np.conj((linalg.orth(yr)).T),
                            e['RA'](e['Ri'][0],lamda,s,p)),linalg.orth(yr)))

        omegal,gammal = manifold_polar(e['Li'],linalg.orth(yl),lamda,e['LA'],
                                        s,p,m,e['kl'],muL)
        omegar,gammar = manifold_polar(e['Ri'],linalg.orth(yr),lamda,e['RA'],
                                        s,p,m,e['kr'],muR)

        out = (np.conj(linalg.det(np.dot(np.conj(linalg.orth(yl).T),yl)))*
                linalg.det(np.dot(np.conj(linalg.orth(yr).T),yr))*
                np.conj(gammal)*gammar*linalg.det(
                np.conj(omegal.T).dot(omegar)))

    elif e['evans'] == "reg_adj_polar":
        muL = np.trace(np.dot(np.dot(np.conj((linalg.orth(yl)).T),
                        e['LA'](e['Li'][0],lamda,s,p)),linalg.orth(yl)))
        muR = np.trace(np.dot(np.dot(np.conj((linalg.orth(yr)).T),
                        e['RA'](e['Ri'][0],lamda,s,p)),linalg.orth(yr)))

        omegal,gammal = manifold_polar(e['Li'],linalg.orth(yl),lamda,e['LA'],
                                        s,p,m,e['kl'],muL)
        omegar,gammar = manifold_polar(e['Ri'],linalg.orth(yr),lamda,e['RA'],
                                        s,p,m,e['kr'],muR)

        out = ( linalg.det(np.dot(np.conj(linalg.orth(yl).T),yl))*
                    np.conj(linalg.det(np.dot(np.conj(linalg.orth(yr).T),yr)))*
                    np.conj(gammar)*gammal*linalg.det(
                    np.conj(omegar.T).dot(omegal)))


    elif e['evans'] == "adj_reg_compound":
        Lmani = manifold_compound(e['Li'],wedgieproduct(yl),lamda,s,p,m,
                                  e['LA'],e['kl'],1)
        Rmani = manifold_compound(e['Ri'],wedgieproduct(yr),lamda,s,p,m,
                                  e['RA'],e['kr'],-1)

        out = np.inner(np.conj(Lmani),Rmani)

    elif e['evans'] == "reg_adj_compound":
        Lmani = manifold_compound(e['Li'],wedgieproduct(yl),lamda,s,p,m,
                                  e['LA'],e['kl'],1)
        Rmani = manifold_compound(e['Ri'],wedgieproduct(yr),lamda,s,p,m,
                                  e['RA'],e['kr'],-1)

        out = np.inner(np.conj(Rmani),Lmani)

    elif e['evans'] == "reg_reg_bvp_cheb":
        VLa,VLb = bvp_basis_cheb(s,p,m,lamda,e['A_pm'],e['LA'],-1,e['kl'])
        VRa,VRb = bvp_basis_cheb(s,p,m,lamda,e['A_pm'],e['RA'],1,e['kr'])
        temp = linalg.null_space(yl.T)
        detCL = ( np.linalg.det(np.hstack([yl,temp]))
                 / np.linalg.det(np.hstack([VLa,temp])) )
        temp = linalg.null_space(yr.T)
        detCR = ( np.linalg.det(np.hstack([yr,temp]))
                 / np.linalg.det(np.hstack([VRb,temp])) )

        out = np.linalg.det(np.hstack([VLb,VRa]))*detCL*detCR

    elif e['evans'] == "regular_periodic":
        sigh = manifold_periodic(e['Li'],np.eye(e['kl']),lamda,s,p,m,e['kl'])
        out = np.zeros((1,len(kappa)),dtype=np.complex)
        for j in range(len(kappa)):
            out[j] = np.det(sigh-np.exp(1j*kappa[j]*p['X'])
                        *np.exp(e['kl']))

    elif e['evans'] == "balanced_periodic":
        sigh = manifold_periodic(e['Li'],np.eye(e['kl']),lamda,s,p,m,e['kl'])
        phi = manifold_periodic(e['Ri'],np.eye(e['kr']),lamda,s,p,m,e['kr'])
        out = np.zeros((1,len(kappa)),dtype=np.complex)
        for j in range(len(kappa)):
            out[j] = np.linalg.det(sigh-np.exp(1j*kappa[j]*p['X'])*phi)

    elif e['evans'] == "balanced_polar_scaled_periodic":
        kappa = yr
        Amatrix = e['A'](e['Li'][0],lamda,s,p)
        k, kdud = np.shape(Amatrix)
        egs = np.linalg.eigvals(Amatrix)
        real_part_egs = np.real(egs)
        cnt_pos = len(np.where(real_part_egs > 0)[0])
        cnt_neg = len(np.where(real_part_egs < 0)[0])
        if not (cnt_neg == e['dim_eig_R']):
            raise ValueError("consistent splitting failed")
        if not (cnt_pos == e['dim_eig_L']):
            raise ValueError("consistent splitting failed")
        index1 = np.argsort(-real_part_egs)
        muL = np.sum(egs[index1[0:e['dim_eig_L']]])
        index2 = np.argsort(real_part_egs)
        muR = np.sum(egs[index2[0:e['dim_eig_R']]])
        # Initializing vector
        ynot = linalg.orth(np.vstack([np.eye(k),np.eye(k)]))
        # Compute the manifolds
        sigh, gammal = manifold_polar(e['Li'],ynot,lamda,A_lift_matrix,s,p,m,k,muL)
        phi, gammar = manifold_polar(e['Ri'],ynot,lamda,A_lift_matrix,s,p,m,k,muR)
        #print(sigh, '\n', gammal, '\n', phi, '\n', gammar)
        #STOP
        out = np.zeros((1,len(kappa)),dtype=np.complex)
        for j in range(1,len(kappa)+1):
            out[:,j-1] = gammal*gammar*np.linalg.det(np.vstack([np.concatenate(
                [sigh[:k,:k],np.exp(1j*kappa[j-1]*p['X'])*phi[:k,:k]],axis=1),
                np.concatenate([sigh[k:2*k,:k], phi[k:2*k,:k]],axis=1)]))

    elif e['evans'] == "bpspm":
        out = Struct()
        out.lamda = lamda
        muL = 0
        muR = 0
        # initializing vector
        ynot = linalg.orth(np.vstack([np.eye(e['kl']),np.eye(e['kr'])]))
        # compute the manifolds
        out.sigh,out.gammal = manifold_polar(e['Li'],ynot,lamda,A_lift_matrix,s,p,
                                             m,e['kl'],muL)
        out.phi,out.gammar = manifold_polar(e['Ri'],ynot,lamda,A_lift_matrix,s,p,
                                             m,e['kr'],muR)

    elif e['evans'] == "balanced_polar_periodic":
        kappa = yr
        muL = 0
        muR = 0
        k = e['kl']

        # initializing vector
        ynot = linalg.orth(np.vstack([np.eye(k), np.eye(k)]))

        # compute the manifolds
        sigh, gammal = manifold_polar(e['Li'],ynot,lamda,A_lift_matrix,s,p,m,
                                      k,muL)
        phi, gammar = manifold_polar(e['Ri'],ynot,lamda,A_lift_matrix,s,p,m,
                                     k,muR)
        out = np.zeros(len(kappa),dtype=np.complex)
        for j in range(len(kappa)):
            out[j] = gammal*gammar*np.linalg.det(np.vstack([np.concatenate(
                    [sigh[:k,:k],np.exp(1j*kappa[j]*p.X)*phi[:k,:k]],axis=1),
                    np.concatenate([sigh[k:2*k,:k], phi[k:2*k,:k]],axis=1)]))

    else:
        raise ValueError("e['evans'], '"+e['evans']+"', is not implemented.")

    return out

def lopatinski(yl,yr,lamda,s,p,m,e):
    # the jump condition could be evaluated once per parameter set resulting
    # in a little bit of speed up. However, this increases the risk of the user 
    # failing to update it, so we perform the evaluation here
    left = e['jump'](-1,lamda,s,p)
    right = e['jump'](1,lamda,s,p)
    
    # evaluate jump condition
    jump = right - left
    
    # construct lopatinksi matrix
    matrix = np.hstack((yl,jump,yr))
    
    # return determinant
    return np.linalg.det(matrix)

def A_lift_matrix(x,lamda,s,p):
    # out=A_lift_matrix(x,lamda,s,p)
    #
    # Returns matrix with s.A(x,lambda,s,p) in the upper left had corner and
    # zeros elsewhere. Used in polar coordinate method for periodic
    # Evans function.

    Amatrix = Aper(x,lamda,s,p)
    sx,sy = np.shape(Amatrix)
    out = np.vstack([np.hstack([Amatrix, np.zeros((sx,sy))]),
                    np.hstack([np.zeros((sx,sy)), np.zeros((sx,sy))])])
    return out

def argsortToMatch(conjEigVals, regEigVals, tol=1e-5):
    """
    This function takes two lists of eigenvalues, conjEigVals and regEigVals,
    where conjEigVals are the eigenvalues of A_inf.conj().T, and regEigVals are
    the eigenvalues of A_inf. It matches them up, finding which eigenvalues
    from conjEigVals are closest to the eigenvalues from regEigVals, and
    returns a list of indices which would index conjEigVals in the same order
    as regEigVals.
    Optional: tol
        tol is the tolerance for the match. We will only allow values where
        if a \in conjEigVals, and b \in regEigVals is the closest match to a,
        then norm(a.conj() - b) <= tol
    """
    ind = np.zeros(len(regEigVals),dtype=int)
    # First, we take the conjugate because we should have that, within machine
    #  tolerance, for all a \in conjEigVals, there exists b \in regEigVals with
    #  a.conj() == b
    conjEigVals = conjEigVals.conj()
    for i,eigVal in enumerate(regEigVals):
        # Subtract eigVal from conjEigVals to find which one is closest to 0
        temp = conjEigVals - eigVal
        # Take the norm
        temp = np.abs(temp)
        # Find the index of the smallest one (closest to 0) and write it to ind
        ind[i] = np.argmin(temp)
        # Raise error if they are not within tolerance
        if temp[ind[i]] > tol:
            raise ValueError(
                "It looks like the "+str(i)+"th "+
                "element of regEigVals does not have a match in conjEigVals "+
                "(within tol="+str(tol)+")")
    return ind


def bvp_basis_cheb(s,p,m,lamda,A_pm,Avec,side,kdim):
    # Get end state eigenvectors
    # evaulate Evans matrix at $\pm \infty$
    Apm = A_pm(side,lamda,s,p)
    mu_vals,eig_vec = np.linalg.eig(Apm)
    ind = np.argsort(mu_vals)
    # this line reverses ind if side=-1, and keeps it the same if side=1
    ind = ind[::side]
    mu_vals = mu_vals[ind]
    zj = eig_vec[:,ind].conj().T

    # get eigenvectors of adjoint at endstate
    # evaluate Evans matrix at $\pm \infty$
    Apm = A_pm(side,lamda,s,p)
    temp_mu_vals,eig_vec = np.linalg.eig(Apm.conj().T)
    ind = argsortToMatch(temp_mu_vals, mu_vals)
    adj_proj = eig_vec[:,ind].conj().T
    m.sys_dim = np.shape(Apm)[0]

    # Solve the boundary value problem
    # Evans matrix
    PR = np.array([]).reshape(0,m.sys_dim)

    # initial mesh
    if side == 1:
        num_inf = s.R
    elif side == -1:
        num_inf = abs(s.L)

    dom = side*(np.linspace(num_inf**0.25,0,m.num_polys+1))**4
    dom[0] = side*num_inf
    dom = np.sort(dom)
    VL = np.zeros((m.sys_dim,kdim),dtype=np.complex)
    VR = np.zeros((m.sys_dim,kdim),dtype=np.complex)

    cnt = 0
    for mu in mu_vals[:kdim]:

        # number of eigenvalues in cluster
        PL0 = adj_proj[cnt+1:,:]

        temp = np.array([]).reshape(0,m.sys_dim)
        PL = np.vstack([zj[cnt,:],PL0])

        muI = mu*np.eye(m.sys_dim)
        Afun = lambda x: Avec(x,lamda,s,p) - muI

        # Get BL and BR, the boundary conditions
        blArray = np.zeros(PL.shape[0],dtype=np.complex)
        blArray[0] = 1
        brArray = np.zeros(PR.shape[0],dtype=np.complex)

        # If we are looking at the right side [0,30], switch PR and PL,
        #  B_R and B_L
        if side == 1:
            PL,PR = PR,PL
            blArray,brArray = brArray,blArray

        B_L = blArray
        B_R = brArray
        Proj_L = PL
        Proj_R = PR

        sol = cheby_bvp.bvpinit(Afun, dom, m.degree, Proj_L, B_L, Proj_R, B_R)
        sol.solve()
        sol.deval(dom)
        VL[:,cnt],VR[:,cnt] = sol.y[:,0], sol.y[:,-1]
        # check on the B.C.s
        BC_L = np.dot(PL,VL[:,cnt]) - blArray
        BC_R = np.dot(PR,VR[:,cnt]) - brArray

        # If we are looking at the right side [0,30], switch back PR and PL,
        #  B_R and B_L
        if side == 1:
            PL,PR = PR,PL
            blArray,brArray = brArray,blArray

        temp = np.vstack([temp,VR[:,cnt].conj().T/np.linalg.norm(VR[:,cnt])])
        cnt = cnt + 1

        PR = np.vstack([PR,temp])

    return VL,VR

def manifold_polar(x,y,lamda,A,s,p,m,k,mu):
    """
     Returns "Omega", the orthogonal basis for the manifold evaluated at x[-1]
     and "gamma" the radial equation evaluated at x[-1].

     Input "x" is the interval on which the manifold is solved, "y" is the
     initializing vector, "lambda" is the point in the complex plane where the
     Evans function is evaluated, "A" is a function handle to the Evans
     matrix, s, p,and m are structures explained in the STABLAB documentation,
     and k is the dimension of the manifold sought.
    """

    def ode_f(x,y):
        return m['method'](x,y,lamda,A,s,p,m['n'],k,mu,m['damping'])

    t0 = x[0]
    y0 = y.reshape(m['n']*k,1,order='F')
    y0 = np.concatenate((y0,np.array([[0.0]],dtype=np.complex)),axis=0)
    y0 = y0.T[0]

    #initiate integrator object
    if 'options' in m:
        try:
            integrator = complex_ode(ode_f).set_integrator('dopri5',
                         atol=m['options']['AbsTol'],
                         rtol=m['options']['RelTol'],
                         nsteps=m['options']['nsteps'])
        except KeyError:
            integrator = complex_ode(ode_f).set_integrator('dopri5',atol=1e-6,
                                                            rtol=1e-5,
                                                            nsteps=10000)
    else:
        integrator = complex_ode(ode_f).set_integrator('dopri5',atol=1e-6,
                                                                rtol=1e-5,
                                                                nsteps=10000)

    integrator.set_initial_value(y0,t0) # set initial time and initial value
    integrator.integrate(x[-1])
    Y = integrator.y
    Y = np.array([Y.T]).T

    omega = Y[0:k*m['n'],-1].reshape(m['n'],k,order = 'F')
    gamma = np.exp(Y[m['n']*k,-1])

    return omega, gamma

def manifold_compound(x, z, lamda, s, p, m, A, k, pmMU):
    """
    manifold_compound(x,z,lambda,s,p,m,A,k,pmMU)

    Returns the vector representing the manifold evaluated at x(2).

    Input "x" is the interval the manifold is computed on, "z" is the
    initializing vector for the ode solver, "lambda" is the point on the
    complex plane where the Evans function is computed, s,p,m are structures
    explained in the STABLAB documentation, "A" is the function handle to the
    desired Evans matrix, "k" is the dimension of the manifold sought, and
    "pmMU" is 1 or -1 depending on if respectively the growth or decay
    manifold is sought.
    """

    eigenVals,eigenVects = np.linalg.eig(A(x[0],lamda,s,p))

    ind = np.argmax(np.real(pmMU*eigenVals))
    MU = eigenVals[ind]

    # Solve the ODE
    def ode_f(x,y):
        return capa(x,y,lamda,s,p,A,m['n'],k,MU)
    if 'options' in m:
        try:
            integrator = complex_ode(ode_f).set_integrator('dopri5',
                         atol=m['options']['AbsTol'],
                         rtol=m['options']['RelTol'],
                         nsteps=m['options']['nsteps'])
        except KeyError:
            integrator = complex_ode(ode_f).set_integrator('dopri5',atol=1e-6,
                                                            rtol=1e-5,
                                                            nsteps=10000)
    else:
        integrator = complex_ode(ode_f).set_integrator('dopri5',atol=1e-6,
                                                                rtol=1e-5,
                                                                nsteps=10000)
    x0 = x[0]
    z0 = z.T[0]
    integrator.set_initial_value(z0,x0)
    integrator.integrate(x[-1])
    Z = integrator.y

    out = Z
    return out

def capa(x,y,lamda,s,p,A,n,k,MU):
    """
    Returns the value y'(x) of the first order system
          y' = (A(x,lambda)-(mu)*Identity)*y

    Input x is the value where y'(x) is evaluated, y is the vector y(x),
    lamda is the point in the complex plane where the Evans function is
    evaluated, s,p are structures explained in the STABLAB documentation, A
    is a function handle to the Evans matrix, n is the dimension of the
    system and k is the dimension of the manifold sought, and MU is the
    eigenvalue corresponding to the largest or smallest eigenvalue of
    A(\pm \infty,lambda)
    """

    out = np.dot((A(x,lamda,s,p)-MU*np.eye(int(comb(n,k)))), y)

    return out

def wedgieproduct(M):
    """
    Take wedgie product of columns of M.
    """
    n,k = np.shape(M)
    allCombos = combinations(range(0,n),k)
    out = np.zeros((int(comb(n,k)),1),dtype=np.complex)
    for i,currCombo in enumerate(allCombos):
        out[i] = linalg.det(M[currCombo,:])
    return out

def drury(t,y,lambda0,A,s,p,n,k,mu,damping):
    y = np.array([y.T]).T; W = y[0:k*n,0].reshape(n,k,order = 'F')
    A_temp = A(t,lambda0,s,p)

    y_temp = (np.eye(n)-W.dot(np.conj(W.T) ) ).dot( A_temp.dot(W) )+damping*W.dot( (np.eye(k)-np.conj(W.T).dot(W)) )
    ydot = np.concatenate( ( y_temp.reshape(n*k,1,order = 'F'), np.array([[0]]) ),axis=0)
    ydot[-1] = (np.trace(np.conj(W.T).dot( A_temp.dot(W) ) )-mu)

    return ydot.T[0]

def drury_no_radial(t,y,lambda0,A,s,p,n,k,mu,damping):
    y = np.array([y.T]).T; W = y[0:k*n,0].reshape(n,k,order = 'F')
    A_temp = A(t,lambda0,s,p)

    y_temp = (np.eye(n)-W.dot(np.conj(W.T) ) ).dot( A_temp.dot(W) )+damping*W.dot( (np.eye(k)-np.conj(W.T).dot(W)) )
    ydot = np.concatenate( ( y_temp.reshape(n*k,1,order = 'F'), np.array([[0]]) ),axis=0)

    return ydot.T[0]

def relative_error(x):
    """
    Returns max(|x(j+1)-x(j)|/max(|x(j)|,|x(j+1)|)
    Input "x" is a vector whose relative error is sought
    """

    out1 = np.divide(x[1:]-x[:-1],x[:-1])
    out2 = np.divide(x[1:]-x[:-1],x[1:])
    out1 = max(abs(out1))
    out2 = max(abs(out2))
    out = max(out1,out2)
    return out


def Evans_compute(pre_preimage,c,s,p,m,e):
    """
    Returns the Evans function output for the given input. The structures
    c, s, p, m, and e are described in the STABLAB documentation. The
    input pre_preimage contains the contour points from which the Evans
    function will be evaluated. The sctructure c should contain a field
    lambda_steps and ksteps. The positive integer, c.ksteps indicates how
    many Kato steps will be taken between points on which the Evans function
    is initially evaluated. The positive integer c.lambda_steps indicates how
    many additional points are specified between Kato steps in the contour,
    pre_preimage, for optional evaluation of the Evans function if needed to
    obtain desired relative error, if specified.

    Example: Suppose we want to evaluate the Evans function on a contour with
    3 points. In order to get accurate results we determine that we need to
    take 2 Kato steps bewteen each point. Then our preimage contour will have
    entries [ 1 2 3 4 5 6 7] with entries 1, 4, and 7 corresponding to the 3
    points we want to Evaluate the Evans function on and entries 2 and 3
    intermediate points on our contour between 1 and 4, and points 5 and 6
    intermediate points on the preimage contour between 4 and 7. If we want our
    Evans function output to vary in relative error between consecutive points
    by less than some tolerance, c.tol, then we set c.refine = 'on'. Now the
    Evans function will be evaluated also on entries 2,3,5, and 6 if needed.
    Perhaps we can achieve relative error only evaluating the Evans function on
    the additional point 3. Then the Evans function is not evaluated on points
    2, 5, and 6. Suppose now that even after evaluating the Evans function on
    all the preimage points we don't meet relative tolerance requirements.
    Perhaps the region where we don't meet tolerance is only in one small
    region. We don't want to slow the computation way down by computing the
    Kato basis numerically on additional points on all of the contour. If we
    specify c.lambda_steps = 1, for example, then between each Kato step, we
    can compute the Evans function on an extra point if needed without
    originally computing the Evans function on that point. So now our
    preimage contour has 13 entries with the original points we compute on
    residing in entries 1 7, and 13. The entries on which the Kato basis are
    computed are 1,3,5,7,9,11,13. The entries on which the Kato basis can be
    computed if needed and the Evans fucntion evaluated are 2,4,6,8,10,12.

    If desired, one can set c.check = 'on'. Then the Evans fucntion is first
    evaluated on the last entry of the preimage contour to determine if the
    output and the conjugate of the output are within specified relative
    error, c.tol. This can save time computing the Evans function on half
    a contour approaching the origin if the Evans function is going to fail
    at the origin.
    """

   #Print some info if stats is print
    if 'stats' in c:
        cstats = (c['stats'] == 'on')
        if c['stats'] == 'print':
            print('Finding the kato basis')
    else:
        cstats = 0

    # Find the subset on which Kato basis is initially evaluated
    pre_index = np.arange(0,len(pre_preimage),c['lambda_steps']+1)
    preimage = pre_preimage[pre_index]

    # Find the subset on which Evans function is initially evaluated
    
    lbasis,lproj = c['basisL'](c['Lproj'],c['L'],preimage,s,p,c['LA'],1,c['epsl'])
    rbasis, rproj = c['basisR'](c['Rproj'],c['R'],preimage,s,p,c['RA'],-1,c['epsr'])

    # Create the bases and the preimage
    index = np.arange(0,len(preimage),c['ksteps']+1)
    lbasis2 = lbasis[:,:,index]
    rbasis2 = rbasis[:,:,index]
    preimage2 = preimage[index]
    out = np.zeros((len(preimage2)), dtype='complex');

    # Makes sure the contour can be sucessfully computed close enough to the
    # origin to satisfy tolerance before computing everything
    if ('check' in c) and c['check'] == 'on':
        try:
            out[0] = c['evans'](lbasis2[:,:,0],rbasis2[:,:,0],preimage2[0],s,p,m,e);
            near_origin = c['evans'](lbasis2[:,:,-1],rbasis2[:,:,-1],preimage2[-1],s,p,m,e);
            out[-1] = near_origin/out[0]; #matlab first index is 1.
            if abs(np.conj(out[-1])-out[-1])/abs(out[-1]) > c['tol']:
                raise ValueError('The Evans function does not satisfy tolerance at the endpoint of the contour');
            print('The Evans function successfully computed at the end point.')
        except ValueError:
            print('The Evans function failed to compute at the end point.')
            print("\tFirst end point: ", out[0])
            print("\tLast end point: ", out[-1])
            print("\tEnd point error: ", abs(np.conj(out[-1])-out[-1])/abs(out[-1]))
            print()

    # Compute the evans function on the contour, using multiprocessing for
    #  parallelization.

    # Non-parallel computation -- this is non-default
    if 'parallel' in c and c['parallel'] == 'off':
        for j in range(len(index)):
            out[j] = c.evans(lbasis2[:,:,j],
                                   rbasis2[:,:,j],
                                   preimage2[j],s,p,m,e)
    # Parallel computation -- this is the default
    else:
        num_proc = None
        if 'parallel' in c:
            num_proc = int(c['parallel'])
        with Pool(processes=num_proc) as pool:
            # chunksize determines the number of values of preimage2 that will
            # be used in each calculation per process. For chunksize = n, each
            # process will perform the calculations on n values per iteration.
            # chunksize can be finetuned to find optimal speed. 2 seems good.
            chunksize = 2
            out = np.fromiter(pool.starmap(c['evans'],[(lbasis2[:,:,j],
                rbasis2[:,:,j],preimage2[j],dict(s),dict(p),dict(m),dict(e))
                for j in range(len(index))],chunksize=chunksize),
                dtype=np.complex)

    # Check if c.refine has been set to 'on'
    if (c['refine'] not in c) or (c['refine'] != 'on'):
        return out, preimage2

    # Refine the mesh on which the Evans function is computed until requested
    #  tolerance is achieved using the Kato steps as needed.

    rel_error = relative_error(out)

    if rel_error > c['tol']:
        if ('stats' in c) and (c['stats'] == 'print'):
            print("Refining to rel_error < ",c['tol'])
        preimage2,out,index = refine_contour(index, preimage, lbasis, rbasis,
                                        preimage2, out, s, p, m, c, e, cstats)

    # Use lambda_steps to further refine the mesh as needed
    rel_error = relative_error(out)

    if rel_error > c['tol']:
        # Test if lambda_steps have been specified.
        if c['lambda_steps'] == 0:
               raise ValueError("Not enough lambda_steps specified to obtain"+
                                " desired relative error")
        print("Using lambda steps to refine further.")
        final_out_index = 0
        final_out = out
        final_preimage2 = preimage2

        # Compute the Evans function on additional points in the problem areas
        #  using lambda_steps
        for j in range(len(out)-1):
            if abs(out[j+1]-out[j])/min(abs(out[j]),abs(out[j+1])) > c['tol']:

                # Find the needed points, basis, etc. between Kato steps where
                # tolerance is too large
                temp_preimage = pre_preimage[(index[j]-1)*c['lambda_steps']
                    +index[j]:(index[j+1]-1)*c['lambda_steps']+index[j+1]]

                temp_lbasis, temp_lproj = c['basisL'](c['Lproj'],c['L'],
                        temp_preimage,s,p,c['LA'],1,c['epsl'],
                        lbasis[:,:,index[j]],lproj[:,:,index[j]])

                temp_rbasis, temp_rproj = c['basisR'](c['Rproj'],c['R'],
                        temp_preimage,s,p,c['RA'],-1,c['epsr'],
                        rbasis[:,:,index[j]],rproj[:,:,index[j]])
                temp_index = [1,len(temp_preimage)]
                temp_preimage2 = [temp_preimage[1],temp_preimage[-1]]
                temp_out = [out[j], out[j+1]]

                # compute the Evans function
                c['fail'] = 'on'
                if ('best_refine' in c) and (c['best_refine'] == 'on'):
                    c['fail'] = 'off'

                temp_preimage2,temp_out,temp_index = refine_contour(
                    temp_index,temp_preimage,temp_lbasis,temp_rbasis,
                    temp_preimage2,temp_out,s,p,m,c,e,cstats)

                # merge the new computations with old ones
                final_out = np.hstack((final_out[0:j+final_out_index],
                        temp_out[0:],final_out[j+final_out_index:]))
                final_preimage2 = np.hstack((
                        final_preimage2[1:j+final_out_index],
                        temp_preimage2[2:-1]))
                final_preimage2 = np.hstack((final_preimage2,
                        final_preimage2[j+final_out_index+1:-1]))
                final_out_index = final_out_index + len(temp_preimage2) - 2

    else:
        return out, final_preimage2

    # assign merged data to output variables
    out = final_out
    preimage2 = final_preimage2
    return out, preimage2

def refine_contour(index,preimage,lbasis,rbasis,preimage2,out,s,p,m,c,e,
                    cstats):
    """
    returns preimage2, out, index

    Compute the Evans function on additional points as needed to achieve
    requested relative error.
    """
    break_while = False
    rel_error = c.tol + 1
    while (rel_error > c['tol']):
        k = 0
        refine_index = np.ones((0),np.int32)
        # Find new points that need to be computed
        for j in range(len(out)-1):
           if abs(out[j+1]-out[j])/min(abs(out[j]),abs(out[j+1])) > c['tol']:
              if index[j+1]-index[j] > 1:
                 if len(refine_index) == k:
                    refine_index = np.append(refine_index,[0])
                 refine_index[k] = round(0.5*(index[j+1]+index[j]+.1))
              else:
                 print("Warning: points did not reach specified tolerance "+
                       "(see evans.refine_contour)")
                 if 'fail' in c and c['fail'] == 'on':
                    raise ValueError("Not enough contour points to meet "+
                                     "specified tolerance in refine_contour")
                 break_while = True
                 break
              k += 1

        if break_while:
           break

        refine_lbasis = lbasis[:,:,refine_index]
        refine_rbasis = rbasis[:,:,refine_index]
        refine_preimage = preimage[refine_index]

        # Compute the Evans function on the needed entries
        refine_out = np.zeros(len(refine_index),dtype = 'complex')
        # Non-parallel computation -- this is non-default
        if 'parallel' in c and c['parallel'] == 'off':
            for j in range(len(refine_index)):
               refine_out[j] = c.evans(refine_lbasis[:,:,j],
                                       refine_rbasis[:,:,j],
                                       refine_preimage[j],s,p,m,e)
        # Parallel computation -- this is the default
        else:
            num_proc = None
            if 'parallel' in c:
                num_proc = int(c['parallel'])
            with Pool(processes=num_proc) as pool:
                # chunksize determines the number of values of preimage2 that
                # will be used in each calculation per process. For
                # chunksize = n, each process will perform the calculations on
                # n values per iteration. chunksize can be finetuned to find
                # optimal speed. 2 seems good.
                chunksize = 2
                out = np.fromiter(pool.starmap(c['evans'],[
                    (refine_lbasis[:,:,j], refine_rbasis[:,:,j],
                    refine_preimage[j],dict(s),dict(p),dict(m),dict(e))
                    for j in range(len(index))], chunksize=chunksize),
                    dtype=np.complex)

        #merge the two vectors to obtain the refined mesh
        preimage2,out,index = merge(preimage2,out,index,refine_preimage,
                                    refine_out,refine_index)

        rel_error = relative_error(out)
        print("\t(rel_error = ",rel_error,")")
    print()

    return preimage2,out,index

def merge(pre1,post1,index1,pre2,post2,index2):
    """
    returns [mergepre,mergepost,mergeindex]

    Takes as input preimages pre1 and pre2 and corresponding function output
    output post1 and post2 (which may contain output for more than one
    parameter as in solving the periodic Evans function), and index1 and
    index2 describing how the two sets of data should be merged together.
    Returns the two sets of data merged together.
    """
    # Fix the shape of pre and post variables.
    if len(np.shape(pre1)) < 2:
        pre1 = np.expand_dims(pre1, axis=0)
    if len(np.shape(post1)) < 2:
        post1 = np.expand_dims(post1, axis=0)
    if len(np.shape(pre2)) < 2:
        pre2 = np.expand_dims(pre2, axis=0)
    if len(np.shape(post2)) < 2:
        post2 = np.expand_dims(post2, axis=0)

    # Determine the number of data sets in output contained in variable post
    n,sy = np.shape(post1)

    # Initialize variables
    mergepre = np.zeros((n,len(pre1[0])+len(pre2[0])), dtype=np.complex)
    mergepost = np.zeros((n,np.size(mergepre,1)), dtype=np.complex)
    mergeindex = np.zeros(np.size(mergepre,1), dtype=int)

    k = 0
    r = 0
    for j in range(len(mergepre[0])): #Loop through n
        # Attach the last entries if merging has occurred up to the last
        #  value of one variable.
        """if k >= len(index1): # k loops through index1
            mergepre=[mergepre[1:j-1], pre2[r:]];
            mergepost[0:n,:]=[mergepost[:,0:j-1], post2[:,r:]];
            mergeindex=[mergeindex[1:j-1], index2[r:]];
            break
        if r >= len(index2): # r loops through index2
            mergepre=[mergepre[0:j-1], pre1[k:]];
            mergepost[0:n,:]=[mergepost[0:n,1:j-1], post1[0:n,k:]];
            mergeindex=[mergeindex[0:j-1], index1[k:]];
            break
         """ # Adding this commented portion back in may speed up the merging.

        # Merge entries where the variables have intersecting values
        if r >= len(index2) or (k < len(index1) and index1[k] < index2[r]):
            mergepre[0,j] = pre1[0,k] # May have to be changed from zero.
            mergepost[0:n,j] = post1[0:n,k]
            mergeindex[j] = index1[k]
            k = k + 1
        else:
           mergepre[0,j] = pre2[0,r] # May have to be changed from zero.
           mergepost[0:n,j] = post2[0:n,r]
           mergeindex[j] = index2[r]
           r = r + 1
    #return mergepre[0],mergepost[0],mergeindex
    return mergepre, mergepost, mergeindex

def emcset(s, shock_type, eLR, Evan_type="default", evans_matrix=None, compound_evans_matrix=None):
    """
    def emcset(s,shock_type,eL,eR,Evan_type):

    Sets the values of the STABLAB structures e, m, and c to
    default values. Takes as input a string, shock_type, which is either
    "front" or "periodic". The input eL and eR are respectively the
    dimension of the left and right eigenspaces of the Evans matrix.
    The input Evan_type is an optional string. If not specified, Evan_type
    will be assigned the most advantageous polar coordinate method.

    Evan_type has the following options when shock_type = 'front':

    reg_reg_polar
    reg_adj_polar
    adj_reg_polar
    reg_adj_compound
    adj_reg_compound
    reg_reg_cheby

    when shock_type = 'periodic', the choices are:

    regular_periodic
    balanced_periodic
    balanced_polar_scaled_periodic
    balanced_polar_periodic
    balanced_scaled_periodic
    """
    if evans_matrix is None and 'A' in s:
        evans_matrix = s['A']
    else:
        s['A'] = evans_matrix

    if compound_evans_matrix is None and 'Ak' in s:
        compound_evans_matrix = s['Ak']
    else:
        s['Ak'] = compound_evans_matrix

    # Go through the shock.
    if shock_type == "front":
        e,m,c = initialize_front(s,eLR[0],eLR[1],Evan_type,evans_matrix,
                                 compound_evans_matrix)
        return s,e,m,c
    elif shock_type == "periodic":
        s,e,m,c = initialize_periodic(s,eLR[0],eLR[1],Evan_type,evans_matrix)
        return s,e,m,c
    elif shock_type == "lopatinski":
        e,m,c = initialize_lopatinski(evans_matrix,s,shock_type) # changed func to evans_matrix
        return s,e,m,c
    elif shock_type == "lopatinski2":
        e,m,c = initialize_lopatinski(evans_matrix,s,shock_type) # changed func to evans_matrix
        return s,e,m,c
    else:
        raise ValueError("user must specify which type of traveling wave "+
                         "is being studied -- front, periodic, lopatinski, "+
                         "or lopatinski2")

def initialize_periodic(s,eL,eR,Evan_type,evans_matrix):
    s = Struct(s)
    e = Struct()
    c = Struct()
    m = Struct()

    # Find center of the wave
    xdom = np.linspace(s['sol']['x'][0],s['sol']['x'][-1],1000);
    yran = np.zeros((len(xdom),1));
    for j in range(1,len(xdom)):
        temp = s['sol']['sol'](xdom[j-1])
        yran[j-1] = temp[0]

    maxval = yran[0]
    xind = 1
    for j in range(1,len(yran)+1):
        if yran[j-1] > maxval:
            maxval = yran[j-1]
            xind = j
    s['center'] = xdom[xind-1]

    # set default structure values
    n = eL+eR;
    c['ksteps'] = 2**18
    c['lambda_steps'] = 0
    c['refine'] = 'off'
    c['tol'] = 0.2
    c['evans'] = evans
    e['A'] = Aper
    e['Li'] = [-s['X']/2,0]
    e['Ri'] = [s['X']/2,0]
    e['kl'] = n
    e['kr'] = n
    e['dim_eig_L'] = eL
    e['dim_eig_R'] = eR
    s['A'] = evans_matrix # e['A']
    m['options'] = {'AbsTol': 10**(-8), 'RelTol': 10**(-6)}
    m['A'] = e['A']
    m['method'] = drury
    m['damping'] = 0
    m['n'] = 2*n

    if Evan_type == 'regular_periodic':
        e['evans'] = 'regular_periodic'
        e['Li'] = [0,s['X']]
        e['kl'] = n;
    elif Evan_type == 'balanced_scaled_periodic':
        e['evans'] = 'balanced_scaled_periodic'
    elif Evan_type == 'balanced_periodic':
        e['evans'] = 'balanced_periodic'
    elif Evan_type == 'balanced_polar_periodic':
        e['evans'] = 'balanced_polar_periodic'
    elif Evan_type == 'balanced_polar_scaled_periodic':
        e['evans'] ='balanced_polar_scaled_periodic'
    elif Evan_type == 'default':
        e['evans'] = 'balanced_polar_periodic'

    return s,e,m,c

def initialize_lopatinski(func, s, shock_type):
    s = Struct(s)
    e = Struct()
    c = Struct()
    m = Struct()
    
    e['evans'] = shock_type
    
    # set default structure values
    c['LA'] = func
    c['RA'] = func
    
    c['stats'] = 'off'
    c['refine'] = 'off'
    c['tol'] = 0.1
    c['ksteps'] = 2**5
    c['lambda_steps'] = 0
    c['basisL'] = analytic_basis
    c['basisR'] = analytic_basis
    c['evans'] = evans
    
    c['epsl'] = 0
    c['epsr'] = 0
    c['Lproj'] = projection2
    c['Rproj'] = projection2
    
    # dependent structure variables
    e['Li'] = [s['L'],0]
    e['Ri'] = [s['R'],0]
    c['L'] = s['L']
    c['R'] = s['R']
    
    return e,m,c

def initialize_front(s,kL,kR,Evan_type,evans_matrix,compound_evans_matrix):
    s = Struct(s)
    e = Struct()
    c = Struct()
    m = Struct({'n':kL+kR})

    if Evan_type == 'default':
        if kL > m['n']/2:
            e['evans'] = 'adj_reg_polar'
        elif kL < m['n']/2:
            e['evans']='reg_adj_polar'
        else:
            e['evans'] ='reg_reg_polar'
    else:
        e['evans'] = Evan_type

    if e.evans == 'reg_adj_polar':
        c.update({
            'LA': evans_matrix,
            'RA': Aadj,
        })
        e.update({
            'LA': c.LA,
            'RA': c.RA,
            'kl': kL,
            'kr': m.n-kR,
        })
    elif e.evans == 'reg_reg_polar':
        c.update({
            'LA': evans_matrix,
            'RA': evans_matrix,
        })
        e.update({
            'LA': c.LA,
            'RA': c.RA,
            'kl': kL,
            'kr': kR,
        })
    elif e.evans == 'adj_reg_polar':
        c.update({
            'LA': Aadj,
            'RA': evans_matrix,
        })
        e.update({
            'LA': Aadj,
            'RA': evans_matrix,
            'kl': m.n-kL,
            'kr': kR,
        })
    elif e.evans == 'reg_adj_compound':
        c.update({
            'LA': evans_matrix,
            'RA': Aadj,
           })
        e.update({
            'LA': compound_evans_matrix,
            'RA': Akadj,
            'kl': kL,
            'kr': kR,
           })
    elif e.evans == 'adj_reg_compound':
        c.update({
            'LA': Aadj,
            'RA': evans_matrix,
        })
        e.update({
            'LA': Akadj,
            'RA': compound_evans_matrix,
            'kl': kL,
            'kr': kR,
        })
    elif e.evans == 'reg_reg_cheby':
        c.update({
            'LA': evans_matrix,
            'RA': evans_matrix,
        })
        e.update({
            'LA': c.LA,
            'RA': c.RA,
            'kl': kL,
            'kr': kR,
            'NL': 60,
            'NR': 60,
        })

    c.update({'stats':'off',
              'refine':'off',
              'tol':0.2,
              'ksteps':2**5,
              'lambda_steps':0,
              'basisL':analytic_basis,
              'basisR':analytic_basis,
              'evans':evans})

    c.update({'epsl':0,
              'epsr':0,
              'Lproj':projection2,
              'Rproj':projection2 })

    m.update({'damping':0,
              'method':drury })

    # Dependent structure variables
    e.update({'Li':[s['L'], 0],
              'Ri':[s['R'], 0] })
    c.update({'L':s['L'],
              'R':s['R'] })

    return e,m,c

def Aadj(x, lamda, s, p):
    """
    Returns the conjugate transpose of the matrix, s.A(x,lamda,s,p)
    """
    out = -np.conj(s['A'](x,lamda,s,p).T)
    return out

def Akadj(x, lamda, s, p):
    """
    Returns the conjugate transpose of the matrix, s.Ak(x,lamda,s,p)
    """
    out = -np.conj(s['Ak'](x,lamda,s,p).T)
    return out

def LdimRdim(Amat, s, p):
    """
    Returns an array (out) with 2 elements, such that out[0] is the number of
    eigenvalues of Amat with positive real part, and out[1] is the number of
    eigenvalues of Amat with negative real part
    """
    out = np.zeros((2),dtype=int)
    egs = np.linalg.eigvals(Amat(s.L,1,s,p))
    indx, = np.where(np.real(egs) > 0)
    out[0] = len(indx)
    indx, = np.where(np.real(egs) < 0)
    out[1] = len(indx)
    return out

def radius(c,s,p,m,e):
    """
    # FIXME: This function does not yet operate identically to the MATLAB
    #        version. Do not use it yet. This function probably needs just a
    #        little bit of work to be operational.
    #
    # Find radius of a semicircle enclosing possible unstable eigenvalues of
    # D(lambda) for rNS. Fit with $C_1 e^{C_2\sqrt{\lambda}}$
    """
    c.lambda_steps = 0
    c.stats = 'off'
    c.refine = 'off'
    err = 1
    mult = 1
    prev_err = 10**(-12)

    while err > c.tol:
        R = 2**mult

        mult = mult+1
        if R > c.max_R:
            raise ValueError('Needed radius for convergence exceeds tolerance')

        line = np.linspace(R,min(2*R,R+50),2+c.ksteps)
        D, domain = Evans_compute(line,c,s,p,m,e) # FIXME: when comparing this function to the matlab version, this line is where the two start to differ
        D = np.real(D)

        # Curve fitting with C1*np.exp(C2*sqrt(lambda)).
        C2 = (np.log(np.abs(D[-1]))-np.log(np.abs(D[0])))/(np.sqrt(line[-1])
                    -np.sqrt(line[0]))
        C1 = D[-1]*np.exp(-C2*np.sqrt(line[-1]))

        theta = np.linspace(0,np.pi/2,2+c.ksteps)
        curve = R*np.exp(1j*theta)

        DRi,domain = Evans_compute(curve,c,s,p,m,e)
        err1 = abs(DRi[-1]-C1*np.exp(C2*np.sqrt(curve[-1])))/abs(DRi[-1])
        err2 = (abs(np.conj(DRi[-1])-C1*np.exp(C2*np.sqrt(curve[-1])))
                    /abs(DRi[-1]))
        err = min(err1,err2)

        if err/prev_err > 0.8:
            prev_err = err
            err = 1
        else:
            prev_err = err

        # Compute and compare the fit on more points of the semicircle
        if err <= c.tol:
            theta = np.linspace(0,np.pi/2,7*(c.ksteps)+8)
            curve = R*np.exp(1j*theta)
            DR = Evans_compute(curve,c,s,p,m,e)
            index = np.arange(0,len(curve),c.ksteps)
            curve2 = curve[index]
            half_err = 0
            for j in range(len(DR)):
                err1 = abs(DR[j]-C1*np.exp(C2*np.sqrt(curve2[j])))/abs(DR[j])
                err2 = (abs(np.conj(DR[j])-C1*np.exp(C2*np.sqrt(curve2[j])))
                            /abs(DR[j]))
                half_err = max(half_err,min(err1,err2))

            if half_err > c.tol:
                err = 1

    return R

def power_expansion2(R_lambda,k_radii,lambda_powers,k_powers,s,p,m,c,e):
    """
    # function out = power_expansion2(R_lambda,k_radii,lambda_powers,k_powers,s,p,m,c,e)
    #
    # Returns the double contour integral of $D(lambda,kappa)/(lambda^r
    # kappa^s)$ evaluated on a contour in lambda of radius R_lambda and in
    # kappa of radius k_radii(j) for the jth power in kappa, s = k_powers(j).
    #
    #
    # Example: St. Venant's equation
    #
    # [s,e,m,c] = emcset(s,'periodic',[2,1],'balanced_polar_periodic','Aper');
    #
    # m.k_int_options = odeset('AbsTol',10^(-10), 'RelTol',10^(-8));
    # m.lambda_int_options = odeset('AbsTol',10^(-8), 'RelTol',10^(-6));
    #
    # st.k_int_options = m.k_int_options;
    # st.lambda_int_options = m.lambda_int_options;
    #
    # R_lambda = 0.01;
    # k_radii = 9*[0.001, 0.001, 0.001 0.001];
    # k_powers = [0 1 2 3 ];
    # lambda_powers = [0 1 2 3];
    #
    # st.R_lambda = R_lambda;
    # st.k_radii = k_radii;
    #
    # s_time = tic;
    # out = power_expansion2(R_lambda,k_radii,lambda_powers,k_powers,s,p,m,c,e);
    # st.time = toc(s_time);
    #
    # coef = out.';
    #
    # aa=coef(3,1);
    # bb=coef(2,2);
    # cc=coef(1,3);
    # dd=coef(4,1);
    # ee=coef(3,2);
    # ff = coef(2,3);
    # gg = coef(1,4);
    #
    # alpha1=(-bb+sqrt(bb^2-4*aa*cc))/(2*aa);
    # alpha2=(-bb-sqrt(bb^2-4*aa*cc))/(2*aa);
    # beta1=-(dd*alpha1^3+alpha1^2*ee+alpha1*ff+gg)/(2*aa*alpha1+bb);
    # beta2=-(dd*alpha2^3+alpha2^2*ee+alpha2*ff+gg)/(2*aa*alpha2+bb);
    """
    # input error checking
    if len(k_radii) != len(k_powers):
       raise ValueError("Input k_radii must have the same number of entries as k_powers")

    # HELPER FUNCTIONS:

    #--------------------------------------------------------------------------
    # lambda_ode
    #--------------------------------------------------------------------------
    def lambda_ode(t,y,k_radii,k_powers,lambda_powers,R_lambda,p,s,e,m,c):

        lamda = R_lambda*np.exp(1j*t)

        # get manifolds from Evans solver
        e.evans = 'bpspm'
        mani = c.evans(1,0,lamda,s,p,m,e)

        # $\int_{|z| = R_k}D(lamda,k)/k^{r+1} dk$, r = k_powers
        temp = k_int(mani,k_radii,k_powers,p,m,e)

        len_lambda_powers = len(lambda_powers)
        totlen = len_lambda_powers*len(k_powers)

        mat = np.zeros((len(k_powers),len_lambda_powers),dtype=np.complex)

        for j in range(len_lambda_powers): # 1:len
            mat[:,j] = (1j*temp*lamda**(-lambda_powers[j]))/(2*np.pi*1j)

        pre_out = np.reshape(mat,(totlen))

        out = np.zeros((2*totlen),dtype=np.complex)
        out[:totlen] = np.real(pre_out)
        out[totlen:2*totlen] = np.imag(pre_out)
        return out

    #--------------------------------------------------------------------------
    # k_int
    #--------------------------------------------------------------------------
    def k_int(mani,k_radii,k_powers,p,m,e):

        # time interval
        tspan = [0,2*np.pi]
        # initial condition
        ynot = np.array([0,0])

        out = np.zeros((len(k_powers)),dtype=np.complex)

        for j in range(len(k_powers)): # 1:length(k_powers)

            pre_k_ode = lambda t,y: k_ode(t,y,mani,k_radii[j],k_powers[j],p,e)
            # solve ode
            integrator = complex_ode(pre_k_ode).set_integrator('dopri5',
                            atol=m['k_int_options']['AbsTol'],
                            rtol=m['k_int_options']['RelTol'])
            integrator.set_initial_value(ynot,tspan[0])
            integrator.integrate(tspan[-1])
            Y = integrator.y
            Y = np.array([Y.T]).T

            # gather output
            out[j] = Y[0,-1]+Y[1,-1]*1j

        return out

    #--------------------------------------------------------------------------
    # k_ode
    #--------------------------------------------------------------------------
    def k_ode(t,y,mani,R_k,kpow,p,e):

        # Floquet parameter
        kappa = R_k*np.exp(1j*t)

        # Evans function
        D = np.linalg.det(np.vstack([
            np.hstack([mani.sigh[:e.kl,:e.kl], np.exp(1j*kappa*p.X)*mani.phi[:e.kl,:e.kr]]),
            np.hstack([mani.sigh[e.kl:2*e.kl,:e.kl], mani.phi[e.kl:2*e.kl,:e.kl]])
            ]))

        # integrand
        temp = (1j*D*np.exp(-1j*kpow*t)/R_k**kpow)/(2*np.pi*1j)
        out = np.zeros((2))
        # split into real and imaginary parts
        out[0] = np.real(temp)
        out[1] = np.imag(temp)
        return out

    # Begin power_expansion2

    len_lambda_powers = len(lambda_powers)
    totlen = len_lambda_powers*len(k_powers)

    # time interval
    tspan = [0,2*np.pi]

    # initial condition
    ynot = np.zeros((2*totlen),dtype=np.complex)

    pre_lambda_ode = lambda t,y: lambda_ode(t,y,k_radii,k_powers,lambda_powers,
                                            R_lambda,p,s,e,m,c)

    integrator = complex_ode(pre_lambda_ode).set_integrator('dopri5',
                    atol=m['lambda_int_options']['AbsTol'],
                    rtol=m['lambda_int_options']['RelTol'])
    integrator.set_initial_value(ynot,tspan[0])
    integrator.integrate(tspan[-1])
    Y = integrator.y
    Y = np.array([Y.T]).T

    out = np.reshape(Y[:totlen,-1]+1j*Y[totlen:2*totlen,-1],(len(k_powers),
                                                            len_lambda_powers))
    return out

def Aper(xx,lamda,s,p):
    x = xx + s['center']
    if x > p['X']:
        x = (x-p['X'])
    elif x < 0:
        x = p['X'] + x

    out = s['A'](x,lamda,s,p);
    return out

def reflect_image(w):
    """Reflects the image of the Evans function across the imaginary axis,
    returning a numpy array that contains both the old points and the
    reflected points.
    """
    return np.concatenate((w[:-1],np.flipud(np.conj(w))))
