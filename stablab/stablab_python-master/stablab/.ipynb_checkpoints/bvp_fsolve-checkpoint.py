import numpy as np
import matplotlib.pyplot as plt
from dmsuite import chebdif
from scipy.optimize import fsolve

def bvp_fsolve(ode,bc,solinit,options):

    #left and right end points of the interval [a,b]
    sol = {}
    sol['a'] = solinit['boundaries'][0]
    sol['b'] = solinit['boundaries'][-1]
    sol['ode'] = ode
    sol['bc'] = bc

    #number of interpolation nodes for Chebyshev polynomial
    sol['N'] = options['N']

    #In addition to returning parameters in a backwards order, this function returns them 10e4 larger than matlab.
    [x0, DM] = chebdif(sol['N']-1, 1)

    sol['xtilde'] = x0 #xtilde \in [-1,1]
    sol['x'] = 0.5*(sol['a']+sol['b'])+0.5*(sol['a']-sol['b'])*sol['xtilde'] # x \in [a,b]


    sol['Tx'] = (2/(sol['a']-sol['b']))*DM[0,:,:]# /(10e3) #derivative of interpolated function, shape correct.

    # dimension of the first order ODE system. solinit['guess'] is the initial guess for the system.
    # It can be a lambda function returning just a number or a list of numbers.  For this reason we check if it returns
    # a list.
    temp = solinit['guess'](sol['a'])
    sol['dim'] = np.shape(temp)[0]

    # initial guess of solution
    y0 = np.zeros((sol['dim'],sol['N']))
    for j in range(0,sol['N']):
        y0[:,j] = solinit['guess'](sol['x'][j])

    #Set up the params variable.
    params = []
    if 'params' in solinit:
        params = solinit['params']
        if params.shape[1] > params.shape[0]:
            params = params.T #Code block works when params is an empty list.

    #Create the list of indices as well as a value Tcf used for chebychev polynomials.
    ind = np.array([range(0,sol['N'])]) #Correct
    Tcf = (2/(sol['N']-1))*np.cos((np.pi/(sol['N']-1))*(ind.T@ind)) #Probably correct. at least it's the correct shape.


    #Set all the values on edges to half their value.
    Tcf[0,:] = 0.5*Tcf[0,:]
    Tcf[-1,:] = 0.5*Tcf[-1,:]
    Tcf[:,0]  = 0.5*Tcf[:,0]
    Tcf[:,-1] = 0.5*Tcf[:,-1]
    sol['Tcf'] = Tcf
    #sol['BClen'] = len(bc(solinit['guess']))

    #Get the F and U0 to be used in the fsolve function.
    u0 = np.reshape(y0.T,(sol['N']*sol['dim'],1))
    u0 = np.vstack((u0, np.zeros((solinit['BClen'],1)))) #Stack BC
    F = lambda u, a: ode_fun(u,ode,bc,sol)

    # Do the Fsolve
    u = fsolve(F, u0, options['algorithm']) #[u,f,exitflag] = fsolve(lammy,u0,options['algorithm'])
    sol['y'] = np.real(np.reshape(u[0:sol['dim']*sol['N']],(sol['N'],sol['dim'])).T) #Correct.
    sol['params'] = np.real(u[sol['dim']*sol['N']+1:])

    # Set up sol with a 'deval' key.
    sol['cf'] = Tcf@sol['y'].T
    sol['cf'] = np.reshape(sol['cf'], (len(sol['cf']),sol['dim']))
    sol['der_cf'] = Tcf.T @ (sol['Tx'] @ sol['y'].T)
    sol['deval'] = lambda x: local_deval(x,sol)

    # Set up other keys in sol.
    sol['check_err'] = lambda num_pts: check_err(num_pts,sol)
    sol['plot'] = lambda num_pts: plot_profile(num_pts,sol)
    sol['solver'] = 'bvp_fsolve';
    return sol

""""-------------------------------------------------------------------------
% plot
-------------------------------------------------------------------------"""
def plot_profile(num_pts,sol):
    dom = np.linspace(sol['a'],sol['b'],num_pts)
    y = sol.deval(dom);
    plt.plot(dom,y,'LineWidth',2)
    #h = legend('R','S','Nr','Ni','Location','Best');
    #set(h,'FontSize',18);
    #h = xlabel('x');
    #set(h,'FontSize',18);
    #h = gca;
    #set(h,'FontSize',18);
    plt.show()


""" -------------------------------------------------------------------------
check_err
-------------------------------------------------------------------------"""
def check_err(num_pts, sol):

    #check residual error of the profile ODE and BCs
    ode_err = 0;
    dom = np.linspace(sol['a'],sol['b'],num_pts)
    [f,f_der] = sol['deval'](dom)

    #Fix the dimensions if incorrect.
    if (len(np.shape(f)) == 1):
        f = np.array([f])

    if (len(np.shape(f_der)) == 1):
        f_der = np.array([f_der])

    #Check the error.
    sol['ode_err_full'] = []
    #if not sol['params']: #FIXME commented out params.
    for j in range(0,len(dom)):

        """"#Test block delete me.
        print(np.shape(f_der))
        print(np.shape(f))
        print(np.shape(sol['ode'](dom[j],f)))
        plt.plot(sol['ode'](dom[j],f)[0])
        plt.plot(sol['ode'](dom[j],f)[1])
        plt.show()
        """
        """
        print(np.shape(f_der[:,j]))
        print(np.shape(sol['ode'](dom[j],f[:,j])))
        print(np.shape(abs(f_der[:,j]-sol['ode'](dom[j],f[:,j]))))
        print()
        """

        if sol['dim'] > 1:
            arr = abs(f_der[:, j] - sol['ode'](dom[j], f[:, j])[:, 0])
            sol['ode_err_full'].append(arr)
            for i in range(sol['dim']):
                ode_err = max(ode_err,max(arr))

        else:
            arr = abs(f_der[:, j] - sol['ode'](dom[j], f[:, j]))
            sol['ode_err_full'].append(arr)
            ode_err = max(ode_err,max(abs(f_der[:,j]-sol['ode'](dom[j],f[:,j]))))

    bc_err = max(abs(sol['bc'](sol['deval'])))
    #else:
    #    for j in range(len(dom)):
    #        ode_err = max(ode_err,max(abs(f_der[:,j]-sol['ode'](dom[j],f[:,j],sol['params']))))
    #    bc_err = max(abs(sol['bc'](sol.deval,sol.params)))

    return [ode_err,bc_err]



"""-------------------------------------------------------------------------
eval
-------------------------------------------------------------------------"""
def local_deval(dom, sol):

    #evaluate the profile and its derivative
    dom_tilde = (2/(sol['a']-sol['b']))*(dom-0.5*(sol['a']+sol['b']))
    theta_dom = np.arccos(dom_tilde)

    ind = np.linspace(0,sol['N']-1,sol['N'])
    T = np.cos(np.outer(theta_dom,ind))

    f = (T@sol['cf']).T
    f_der = (T@sol['der_cf']).T

    # Fix the shape of the outputs so if vectors, they return just vectors.
    if (np.shape(f)[0] == 1 and not np.shape(f)[1] == 1):
        f = f[0,:]
    if np.shape(f_der)[0] == 1 and len(np.shape(f_der)) > 1 and not np.shape(f_der)[1] == 1:
        f_der = f_der[0,:]
    return (f,f_der)


"""-------------------------------------------------------------------------
ode_fun
-------------------------------------------------------------------------"""
def ode_fun(u,ode,bc,sol):

    # ODE for fsolve.  Takes in a vector, u, and returns a vector of the same size.
    # fsolve will try to find the roots of this function.


    sol['y'] = np.real(np.reshape(u[0:sol['dim']*sol['N']],(sol['N'],sol['dim'])).T)
    sol['params'] = np.real(u[sol['dim']*sol['N']:])
    sol['cf'] = sol['Tcf']@sol['y'].T
    sol['der_cf'] = sol['Tcf']@(sol['Tx']@sol['y'].T) #sol['Tx'] is one size too big.

    #Unpack y and params from u. Params are values appended to the end.
    current_deval = lambda x:local_deval(x,sol)
    y = np.reshape(u[0:sol['dim']*sol['N']],(sol['N'],sol['dim'])).T
    params = u[sol['dim']*sol['N']+1:]
    #params = []

    #Deal with the parameters.
    if not params:
        out_bc = bc(current_deval)
    else:
         out_bc = bc(current_deval) #FIXME params was an argument
    tempVar = len(out_bc)
    out = np.zeros((tempVar+sol['dim']*sol['N'],1))


    ders = np.zeros((sol['dim'],sol['N']))
    for j in range(0,sol['N']):
        if not params:

            if sol['dim'] > 1:
                ders[:,j] = ode(sol['x'][j],y[:,j])[:,0]
            else:
                ders[:,j] = ode(sol['x'][j],y[:,j])[0]
        else:
            if sol['dim'] > 1:
                ders[:,j] = ode(sol['x'][j],y[:,j])[:,0]
            else:
                ders[:,j] = ode(sol['x'][j],y[:,j])[0] #FIXME same as without params.




    for j in range(0,sol['dim']):
        out[(j)*sol['N']:(j+1)*sol['N'], 0] = (sol['Tx']@y[j].T-ders[j].T)

    #Plug in the boundary condition.
    out[sol['dim']*sol['N']:,0] = np.real(out_bc)

    return out[:,0]


""""-------------------------------------------------------------------------
 set options
-------------------------------------------------------------------------"""


def bvp_fsolve_options(algorithm_stats='off', Display='off', Jacobian='off', Algorithm='Levenberg-Marquardt',
                       TolFun=1e-10):
    # Default options
    options = {}
    options['algorithm_options'] = algorithm_stats
    options['Display'] = Display
    options['Jacobian'] = Jacobian
    options['Algorithm'] = Algorithm
    options['TolFun'] = TolFun

    optimset = lambda *x: 0  # print('optimset', *x)
    options['algorithm'] = optimset('Display', 'off', 'Jacobian', 'off',
                                    'Algorithm', 'Levenberg-Marquardt', 'TolFun', 1e-10)
    options['N'] = 2 ** 8 # 2**8

    return options

def set_options(options, property, choice):
    optimset = lambda *x: 0  # print('optimset',*x)
    if property == 'algorithm_stats':
        if choice == 'on':
            options.algorithm = optimset(options.algorithm, 'Display', 'iter')
        else:
            options.algorithm = optimset(options.algorithm, 'Display', 'off')

    return options
