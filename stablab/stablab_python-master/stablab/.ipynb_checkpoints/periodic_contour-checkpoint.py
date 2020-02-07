import numpy as np
from stablab.evans import relative_error, merge

def periodic_contour(kappa,preimage,c,s,p,m,e):
    """
    # Returns the periodic Evans function output for the given input.
    #
    # Here c,s,p,m,e are structures described in the
    # STABLAB documentation. The input preimage is the contour on which the
    # Evans function is computed. The input kappa is a 1-D array containing the
    # Floquet parameters for which the Evans function is evaluated. It is
    # relatively cheap to evaluate the periodic Evans function for additional
    # Floquet parameters.

    # The following is not yet implemented:
    # If c.stats = 'on', then a waitbar showing computation
    # progress is displayed. If c.stats = 'print', then computation progress is
    # printed to the command window instead of to a waitbar (use this option
    # for parallel computing)
    """

    # initialize variables
    index = np.arange(1,len(preimage),c['ksteps']+1)
    index = np.append(index,len(preimage))
    preimage2 = preimage[index-1]
    out = np.zeros((len(kappa),len(preimage2)),dtype=np.complex)
    if 'stats' in c and c['stats'] == 'on':
        #h = waitbar(0,'Iniital. ');
        #time1 = tic;
        for j in range(1,len(index)+1):
            out[:,j-1] = c['evans'](1,kappa,preimage2[j-1],s,p,m,e)
            #curr_time = (time.time()-time1);
            #time_left = curr_time*(len(index)/j)-curr_time;
            #waitbar(j/len(index),h,['Initial. Est time left: ',str(time_left), 'sec']);
        #close(h);
    else:
        for j in range(1,len(index)+1):
            out[:,j-1] = c['evans'](1,kappa,preimage2[j-1],s,p,m,e)
    # Debug position 5
    #print("out",out[0,:])
    #input('here is out')
    #print(preimage2)
    if 'stats' in c and c['stats'] == 'on':
        pass

    # determine if user requested a tolerance on relative error
    if not c['refine'] == 'on':
        return out, preimage2

    # refine output to achieve desired relative error
    if 'return_condition' in c and c['return_condition'] == 'if unstable':
        rel_error = 0
        for j in range(1,len(kappa)+1):
            j_err = relative_error(out[j-1,:]);
            if j_err < c['tol']:
                wnd = winding_number(out[j-1,:]);
                if wnd > 0:
                    out = 'unstable';
                    preimage2 = kappa[j-1]
                    return
            rel_error = max(j_err,rel_error)
    else:
        rel_error = 0
        for j in range(1,len(kappa)+1):
            rel_error = max(relative_error(out[j-1,:]),rel_error);

    if 'stats' in c and c['stats'] == 'print':
        print('\nRelative Error: ',rel_error)

    while rel_error > c['tol']:
        k = 1

        # determine which points need to be evaluated to achieve relative
        # tolerance error
        refine_index = []
        sx,sy = np.shape(out)
        for j in range(1,sy):
            local_rel_err = 0
            for t in range(1,len(kappa)+1):
                local_rel_err = max(abs(out[t-1,j]-out[t-1,j-1])
                                    /min(abs(out[t-1,j-1]),abs(out[t-1,j])),
                                    local_rel_err)
            if local_rel_err > c['tol']:
                if index[j]-index[j-1]>1:
                    if len(refine_index) < k:
                        refine_index.append(0)
                    refine_index[k-1] = int(round(.5*(index[j]+index[j-1])))
                else:
                    raise ValueError("The domain contour mesh was not fine "+
                                     "enough to meet requested relative error")
                k = k + 1

        refine_preimage = preimage[refine_index]

        if 'stats' in c and c['stats'] == 'on':
            pass
            #h = waitbar(0,['Refining Evans function, rel error = ',num2str(rel_error),' > ',num2str(c.tol)]);

        # compute the Evans function on new points.
        refine_out = np.zeros((len(kappa),len(refine_index)),dtype=np.complex)

        if 'stats' in c and c['stats'] == 'on':
            #h = waitbar(0,['Refine. rel error = ',num2str(rel_error),' > ',num2str(c.tol)])
            #time1 = tic;
            for j in range(1,len(refine_index)+1):
                refine_out[:,j-1] = c['evans'](1,kappa,refine_preimage[j-1],s,p,m,e)
                #curr_time = (time.time()-time1)
                #time_left = curr_time*(len(refine_index)/j)-curr_time
                #waitbar(j/len(refine_index),h,['Refine. rel error = ',num2str(rel_error),'. Est time left: ',num2str(time_left),' sec'])
            #close(h);
        else:
            for j in range(1,len(refine_index)+1):
                refine_out[:,j-1] = c['evans'](1,kappa,refine_preimage[j-1],s,p,m,e)

        # merge old and new points together
        preimage2,out,index = merge(preimage2,out,index,refine_preimage,refine_out,refine_index)

        # rel_error=0;
        #  for j=1:len(kappa)
        #   rel_error=max(relative_error(out[j-1,:]),rel_error);
        #  end
        if 'return_condition' in c and c['return_condition'] == 'if unstable':
            rel_error = 0
            for j in range(1,len(kappa)+1):
                j_err = relative_error(out[j-1,:])
                if j_err < c['tol']:
                    wnd = winding_number(out[j-1,:])
                    if wnd > 0:
                        out = 'unstable'
                        preimage2 = kappa[j-1]
                        return out, preimage2
                rel_error = max(j_err,rel_error)

        else:
            rel_error = 0
            for j in range(1,len(kappa)+1):
                rel_error = max(relative_error(out[j-1,:]),rel_error);

        if 'stats' in c and c['stats'] == 'print':
            print('\nRelative Error: ',rel_error)

    return out, preimage2[0,:]
