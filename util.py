import numpy as np
import random as rm
#import matplotlib.pyplot as plt
rm.seed(0)


def read_data(bd, files, ntimes, dt, tmax):
    
    ndata = len(files)
    
    rhos = np.zeros((ndata,ntimes,2,2))
    
    for indf,fd in enumerate(files):
        filename = bd + str(fd) +".s"
        f = open(filename,"r")
        time = 0
        for line in f:
            line2 = line.split()
            rhos[indf,time,0,0] = float(line2[1])
            rhos[indf,time,0,1] = float(line2[3])
            rhos[indf,time,1,0] = float(line2[4])
            rhos[indf,time,1,1] = float(line2[7])
            time += 1
            if time == ntimes:
                break
        
        f.close()
        
    tgrid = np.linspace(0,tmax,ntimes,True)
    
    return rhos, tgrid

def split_input(data, mem):
    
    nel    = data.shape[3]*data.shape[2]
    ntimes = data.shape[1]
    ndata  = data.shape[0]
    
    nsegments = ntimes - int(mem) - 1
    
    x = np.zeros((ndata*nsegments,mem,2,2,1))
    y = np.zeros((ndata*nsegments,4))
    
    idata = 0
    for n in range(ndata):
        vec = data[n,:,:,:]
        for m in range(nsegments):
            x[idata,:,:,:,0] = vec[m:m+mem,:,:]
            y[idata,:]     = vec[m+mem,:,:].flatten('F')
            idata += 1
    
    return x, y

def plot_dms(base_dir, tgrid, r00, _r00, r01r, _r01r, r01i, _r01i, r11, _r11, ids, step):
    
    fontsize=16
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.axis([0,1,-1,1])
    
    ax.plot(tgrid, r00, color='g')
    ax.plot(tgrid, _r00, color='g', ls=':')
    
    ax.plot(tgrid, r01r, color='r')
    ax.plot(tgrid, _r01r, color='r', ls=':')
    
    ax.plot(tgrid, r01i, color='b')
    ax.plot(tgrid, _r01i, color='b', ls=':')
    
    ax.plot(tgrid, r11, color='k')
    ax.plot(tgrid, _r11, color='k', ls=':')
    
    plt.tight_layout()
    ax.set_xlabel("Time",fontsize=fontsize)
    ax.set_ylabel("Rho",fontsize=fontsize)

    fn = base_dir + "/" + str(ids) + "_" + str(step) +".pdf"
    ax.savefig(fn,dpi=1200,bbox_inches='tight')

def read_data_skip(base_dir, files, ntimes, dt, dskip, tmax, 
                   ntest, n_files_to_run):
    
    test_ind = [0, 66, 132, 198, 264, 330, 396, 462, 528, 594, 660, 726, 792, 858, 924, 990, 1056, 1122, 1188, 1254, 1320, 1386, 1452, 1518, 1584, 1650, 1716, 1782, 1848, 1914] 

    ndata = len(files)
    raw, tg = read_data(base_dir, files, ntimes, dt, tmax)
    
    ntimenew = int(ntimes/dskip)
    
    rho = np.zeros((ndata,ntimenew,2,2))
    tgrid = np.zeros((ntimenew))
    
    for s in range(ndata):
        for t in range(ntimenew):
            rho[s,t,:,:] = raw[s,t*dskip,:,:]
            
    for t in range(ntimenew):
        tgrid[t] = tg[t*dskip]

    # get subset of files to be used in the run
    
    run_ind = list(range(0,ndata))
    run_ind = np.delete(run_ind, test_ind)      
    

    run_ind=  rm.sample(run_ind.tolist() ,n_files_to_run-ntest )
    

    rho_run = np.take(rho, run_ind, axis=0)

    ntval = n_files_to_run - ntest

    # choose test indices
    #tst = int(n_files_to_run/ntest)
    #test_ind = list(range(0,ntest))
    #test_ind = [x*tst for x in test_ind]
       

    rho_test = np.take(rho_run, test_ind, axis=0)
    rho_tval = np.delete(rho_run, test_ind, axis=0)    

    print (" test indices ", test_ind)
    print (" train indices ", run_ind)

    return rho_tval, rho_test, tgrid, ntimenew, test_ind
    

def save(cur_dir, tgrid, traj, traj1, ti, step):

    fname = cur_dir + "/" + str(step) + "_" + str(ti) + ".dat"
    f = open(fname, "w")
  
    ntime = tgrid.shape[0]

    for n in range(ntime):
       f.write(" %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f \n"%
               (tgrid[n], traj[n,0,0], traj1[n,0,0], traj[n,0,1], traj1[n,0,1], traj[n,1,0], traj1[n,1,0], traj[n,1,1], traj1[n,1,1]))

    f.close()
