import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Activation, Dense, Dropout, GRU #, ConvLSTM2D
from keras.layers import Flatten, LeakyReLU, MaxPooling3D, TimeDistributed
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import util


########################################################################

data_dir = "/work/akanane/users/ML_QD/data/"
cur_dir  = os.getcwd()
files      = range(0,4367) 
n_files_to_run = 2000
ntimes     = 10000
tmax       = 1.000
dt         = 0.0001
dskip      = 10
n_features = 4
   
n_test     = 30
   
memory     = 200
   
test_size  = 0.1
validation_size = 0.2

batch_size = 32
epoch      = 2
lr         = 0.0001
   
print (" memory = %d \n"%(memory),flush=True)
print (" batch_size = %d \n"%(batch_size),flush=True)
print (" epochs = %d \n"%(epoch),flush=True)
print (" lr = %9.7f \n"%(lr),flush=True)

 
rhos_tval, rhos_test, tgrid, ntimes_new, test_ind = util.read_data_skip(data_dir, files, ntimes, dt, dskip, tmax, n_test, n_files_to_run)

x, y = util.split_input(rhos_tval, memory)
x_val,y_val=util.split_input(rhos_test, memory)

indices = np.arange(x.shape[0])
split = train_test_split(x, y, indices, test_size=test_size, random_state=999)
x_train, x_test, y_train, y_test, idx1, idx2 = split

   





##########################################################################



model = Sequential()
model.add(Conv3D(filters=2, kernel_size=(16,1,1), activation='relu', input_shape=(memory,2,2,1), data_format='channels_last'))
model.add(Conv3D(filters=15, kernel_size=(48,1,1), activation='relu', data_format='channels_last'))
model.add(MaxPooling3D(pool_size=(10,1,1), data_format='channels_last'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(4,  activation='linear'))


   
model.add(Dense(4,  activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
                 optimizer=optimizers.Adam(lr=lr),
                 metrics=['mse'])
   
print(model.summary())

result = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epoch,
                       verbose=True,
                       #callbacks=callbacks,
                       validation_split=validation_size)

##############################################################################
# Plot statistics here
acc      = result.history['mean_squared_error']
val_acc  = result.history['val_mean_squared_error']
loss     = result.history['loss']
val_loss = result.history['val_loss']

nps = range(1, len(acc)+1)

f,((ax1),(ax2)) = plt.subplots(2,1)
plt.subplots_adjust(hspace=0.15,wspace=0)

ax1.plot(nps, acc, 'bo', label='Training acc')
ax1.plot(nps, val_acc, 'b', label='Validation acc')
ax1.legend()

ax2.plot(nps, loss, 'bo', label="Training loss")
ax2.plot(nps, val_loss, 'b', label='Validation loss')
ax2.legend()

fname = cur_dir + "/stat.pdf"
f.savefig(fname,dpi=1200,bbox_inches='tight')

################################################################################3
   
yhat = model.predict(x_test)

traj   = np.zeros((ntimes_new,2,2))
traj1  = np.zeros_like(traj)
yout = np.zeros((2,2))

 

aerror = 0.0
    
for ti in range(n_test):
    error0 = 0.0
    error1 = 0.0
    error2 = 0.0
    error3 = 0.0
       
    traj[:,:]  = 0.0
    traj1[:,:] = 0.0
    traj[:,:,:]  = rhos_test[ti,:,:,:]
    traj1[:,:,:] = rhos_test[ti,:,:,:]
       
    for n in range(ntimes_new-memory):
        x_inp = traj[n:n+memory,:,:].reshape(1,memory,2,2,1)
        yhat = model.predict(x_inp, verbose=False)
        yout = np.transpose(yhat.reshape(2,2))
        traj[n+memory,:,:] = yout[:,:]

        #util.plot_dms(cur_dir, tgrid, traj[:,0,0], traj1[:,0,0],traj[:,0,1], traj1[:,0,1], traj[:,1,0], traj1[:,1,0],traj[:,1,1], traj1[:,1,1],ti)
    util.save(cur_dir, tgrid, traj, traj1, ti, 0)

    error0 += np.sum(np.abs(traj[memory:,0,0] - traj1[memory:,0,0]))/len(np.abs(traj[memory:,0,0]))
    error1 += np.sum(np.abs(traj[memory:,0,1] - traj1[memory:,0,1]))/len(np.abs(traj[memory:,0,1]))
    error2 += np.sum(np.abs(traj[memory:,1,0] - traj1[memory:,1,0]))/len(np.abs(traj[memory:,1,0]))
    error3 += np.sum(np.abs(traj[memory:,1,1] - traj1[memory:,1,1]))/len(np.abs(traj[memory:,1,1]))
    
    error0 /= len(test_ind)
    error1 /= len(test_ind)
    error2 /= len(test_ind)
    error3 /= len(test_ind)
    terror = error0 + error1 + error2 + error3

    aerror += terror

    print (" Errors %d : %10.5f %10.5f %10.5f %10.5f total = %10.5f"%(test_ind[ti],error0,error1,error2,error3,terror))
  
print (" Total and average errors %10.5f %10.5f \n"%(aerror,aerror/n_test))
