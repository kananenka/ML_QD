import numpy as np
import random as rm
#import matplotlib.pyplot as plt
rm.seed(0)   # Random seed to choose the training files 


def read_data(bd, files, ntimes, dt, tmax):           

#First fucntion to read the data. 
#bd=base directory, where the files are. 
#files: is a list with the numbes of files to use, range(0,4367)
#ntimes: Number of time steps in the evolution. This comes from  HEOM. in this case 10000
#dt: time step or time resolution, fixed by the HEOM output. In this case 0.0001 
#tmax: Maximun time in the evolution, given by HEOM, in this case 1.000. Should be equal to ntimes X dt
                                                      
    ndata = len(files)                       #We get the number of total files:  4367           
    rhos = np.zeros((ndata,ntimes,2,2))      #0-Matrix, that will represents rho, is 2x2 (2 lvl sysytem), for           
                                             #each time, and we have 4367 files.   
    print("ndata",ndata)

       
    for indf,fd in enumerate(files):                #Loop to read the files data.indf is the index and fd is the number      
        filename = bd + str(fd) +".s"               #  
        f = open(filename,"r")                      #Now for 1 file we do: 
        time = 0                                    #for echa line in the file:
        for line in f:                              #we split the lines, should be like [0,1,0,0,0,0,0,0,0]
            line2 = line.split()                    #first number is time, other are real and imag part of the elements of 
            rhos[indf,time,0,0] = float(line2[1])   #rho. 
            rhos[indf,time,0,1] = float(line2[3])   #We put the selected features, in this case are Re(p00) , Re(p01),
            rhos[indf,time,1,0] = float(line2[4])   # Im(p01), Re(p11)
            rhos[indf,time,1,1] = float(line2[7])
            time += 1
            if time == ntimes:                      #stop if we are in the maxtime, 
                break
        
        f.close()
        
    tgrid = np.linspace(0,tmax,ntimes,True)        #make the vector with time
    
    print("rho_0 ",rhos[0,:,:,:])
    return rhos, tgrid



def read_data_skip(base_dir, files, ntimes, dt, dskip, tmax, 
                   ntest, n_files_to_run):


#function to skip time points, in the rhos matrix. 
#base_dir=base directory, where the files are. 
#files: is a list with the numbes of files to use, range(0,4367)
#ntimes: Number of time steps in the evolution. This comes from  HEOM. in this case 10000
#dt: time step or time resolution, fixed by the HEOM output. In this case 0.0001 
#dskik: number of time stesp we wan to skip, in this case 10 so we only take the time one each 10 time steps
#tmax: Maximun time in the evolution, given by HEOM, in this case 1.000. Should be equal to ntimes X dt
#ntest: Number of files to test the trained machine in this case 30
#n_files_to_run: Number files to train, could be diferent to "files", since files are the files imported, but   
# n_files_to_run are the one to train. 
    
    test_ind = [0, 66, 132, 198, 264, 330, 396, 462, 528, 594, 660, 726, 792, 858, 924, 990, 1056, 1122, 1188, 1254, 1320, 1386, 1452, 1518, 1584, 1650, 1716, 1782, 1848, 1914]  #test files fixed.

    ndata = len(files)						#We get the number of total files:  4367 
    raw, tg = read_data(base_dir, files, ntimes, dt, tmax)      #get rho with all files and the time vector from the  
    								# function read_data
    print("rho_0",raw[0,:,:,:])
    print("time", tg)
    
    ntimenew = int(ntimes/dskip)				#new time length, in this case                      
    
    print("ntimenew",ntimenew)						#should be 10000/10=1000
    rho = np.zeros((ndata,ntimenew,2,2))                        #new rho matrix
    tgrid = np.zeros((ntimenew))                                #new time vector
    
    for s in range(ndata):                                      #for each file imported 
        for t in range(ntimenew):                               #for each new time 
            rho[s,t,:,:] = raw[s,t*dskip,:,:]                   #write the new rho, skiping times  
            
    for t in range(ntimenew):                                   #write the new time vector skipped.
        tgrid[t] = tg[t*dskip]

    print("rho_0_skiped",rho[0,:,:,:])
    print("time_skiped",tgrid)

    # get subset of files to be used in the run

    #at this point rho has all the files to run, we need to takes just the number we want for training,
    #and also take out the ones for testing the trained machine. 

    run_ind = list(range(0,ndata))                  #Get the index for all the files
    run_ind = np.delete(run_ind, test_ind)          #deletee de index of the testing files 
    
    #Get random sample of size n_files_to_run from a list of the run_ind

    #run_ind=  rm.sample(run_ind.tolist() ,n_files_to_run-ntest ) #Use this if nfiles_to_run=ndata 
    run_ind=  rm.sample(run_ind.tolist() ,n_files_to_run )       #Use this if ndata-nfiles_to_run is more than 30 (ntest) 

    rho_run = np.take(rho, run_ind, axis=0)                     #take only the files needed to run. 

    print("rho_1",rho[1,:,:,:])	
    print("rho_to_run_0",rho_run[0,:,:,:])

    ntval = n_files_to_run - ntest

    # choose test indices
    #tst = int(n_files_to_run/ntest)
    #test_ind = list(range(0,ntest))
    #test_ind = [x*tst for x in test_ind]
       

    rho_test = np.take(rho, test_ind, axis=0)           #create the rho wiht the files to test the trained machine. 
    print("rho_0",rho[0,:,:,:])	
    print("rho_test_0",rho_test[0,:,:,:]) 


    rho_tval=rho_run 
    #rho_tval = np.delete(rho_run, test_ind, axis=0)    
 
    print (" test indices ", test_ind)
    print (" train indices ", run_ind)

    return rho_tval, rho_test, tgrid, ntimenew, test_ind



def split_input(data, mem):			#Function to split the data in x, y for the training and testint 
						# x as the imput of the machineand y the output.    
						# 


    nel    = data.shape[3]*data.shape[2]	# nel is the dimesnion of rho, in this case 2x2=4
    ntimes = data.shape[1]   			# ntimes in the dimesion of the time vector, i.e. how many times steps
    ndata  = data.shape[0]                      # Number of files 
    
    nsegments = ntimes - int(mem) - 1		#Form one file we get several cuts of size (memory) as x and the output 
    						# y will be the point at at next time. The this is the number of segmests 
    						# or partitions 
    x = np.zeros((ndata*nsegments,mem,2,2,1))   # Since we hace ndata files, and for each file we have  nsegments  
                                                # the number of x sampls is ndata*nsegments, for con3D we have a volume 
                                                # of memoeryx2x2 and just one channel 
    y = np.zeros((ndata*nsegments,4))           # y will have the 4 features for the next time 
    
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


    

def save(cur_dir, tgrid, traj, traj1, ti, step):

    fname = cur_dir + "/" + str(step) + "_" + str(ti) + ".dat"
    f = open(fname, "w")
  
    ntime = tgrid.shape[0]

    for n in range(ntime):
       f.write(" %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f \n"%
               (tgrid[n], traj[n,0,0], traj1[n,0,0], traj[n,0,1], traj1[n,0,1], traj[n,1,0], traj1[n,1,0], traj[n,1,1], traj1[n,1,1]))

    f.close()
