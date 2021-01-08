import numpy as np
import pandas as pd
from tensorflow import keras
import time
from keras_tqdm import TQDMNotebookCallback
import seaborn as sns


import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

import tensorflow as tf
from obspy.io.segy.segy import _read_segy
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

patch_size = 64 
batch_size = 256
num_channels = 1
num_classes = 9
all_examples = 158812
num_examples = 7500
epochs = 1
steps=450
sampler = list(range(all_examples))

opt = Adam(lr=0.001) 
lossfkt = ['categorical_crossentropy']
metrica = ['mae', 'acc']

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.test.is_gpu_available)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
filename = 'data/Dutch Government_F3_entire_8bit seismic.segy'

t0=time.time()
stream0 = _read_segy(filename, headonly=True)
print('--> data read in {:.1f} sec'.format(time.time()-t0))

t0=time.time()

labeled_data = np.stack(t.data for t in stream0.traces if t.header.for_3d_poststack_data_this_field_is_for_in_line_number == 339).T
inline_data = np.stack(t.data for t in stream0.traces if t.header.for_3d_poststack_data_this_field_is_for_in_line_number == 500).T
xline_data = np.stack(t.data for t in stream0.traces if t.header.for_3d_poststack_data_this_field_is_for_cross_line_number == 500).T

print('--> created slices in {:.1f} sec'.format(time.time()-t0))

#%%
def patch_extractor2D(img,mid_x,mid_y,patch_size,dimensions=1):
    try:
        x,y,c = img.shape
    except ValueError:
        x,y = img.shape
        c=1
    patch= np.pad(img, patch_size//2, 'constant', constant_values=0)[mid_y:mid_y+patch_size,mid_x:mid_x+patch_size] #because it's padded we don't subtract half patches all the tim
    if c != dimensions:
        tmp_patch = np.zeros((patch_size,patch_size,dimensions))
        for uia in range(dimensions):
            tmp_patch[:,:,uia] = patch
        return tmp_patch
    return patch
image=np.random.rand(10,10)//.1
print(image)

patch_extractor2D(image,10,10,4,1)

def acc_assess(data,loss=['categorical_crossentropy'],metrics=['acc']):
    if not isinstance(loss, list):
        try:
            loss = [loss]
        except:
            raise("Loss must be list.")
    if not isinstance(metrics, list):
        try:
            metrics = [metrics]
        except:
            raise("Metrics must be list.")
    out='The test loss is {:.3f}\n'.format(data[0])
    for i, metric in enumerate(metrics):            
        if metric in 'mae':
            out += "The total mean error on the test is {:.3f}\n".format(data[i+1])
        if metric in 'accuracy':
            out += "The test accuracy is {:.1f}%\n".format(data[i+1]*100)
    return out
print(acc_assess([1,2,3],'bla',["acc", "mae"]))

#%%
labels = pd.read_csv('data/classification.ixz', delimiter=" ", names=["Inline","Xline","Time","Class"])
labels.describe()

labels["Xline"]-=300-1
labels["Time"] = labels["Time"]//4
labels.describe()
#%%
fig2 = plt.figure(figsize=(15.0, 10.0))
vml = np.percentile(labeled_data, 99)
img1 = plt.imshow(labeled_data, cmap="Greys", vmin=-vml, vmax=vml, aspect='auto')
plt.yticks(np.arange(0, 462, 100), np.arange(0, 462*4, 400))
plt.xlabel('Trace Location')
plt.ylabel('Time [ms]')
plt.show()

#%%
train_data, test_data, train_samples, test_samples = train_test_split(
    labels, sampler, random_state=42)
print(train_data.shape,test_data.shape)

class SeismicSequence(Sequence):
    def __init__(self, img, x_set, t_set, y_set, patch_size, batch_size, dimensions):
        self.slice = img
        self.X,self.t = x_set,t_set
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.dimensions = dimensions
        self.label = y_set
    
    def __len__(self):
        return len(self.X) // self.batch_size
    
    def __getitem__(self,idx):
        sampler = np.random.permutation(len(self.X))
        samples = sampler[idx*self.batch_size:(idx+1)*self.batch_size]
        labels = keras.utils.to_categorical(self.label[samples], num_classes=9)
        if self.dimensions == 1:
            return np.expand_dims(np.array([patch_extractor2D(self.slice,self.X[x],self.t[x],self.patch_size,self.dimensions) for x in samples]), axis=3), labels
        else:
            return np.array([patch_extractor2D(self.slice,self.X[x],self.t[x],self.patch_size,self.dimensions) for x in samples]), labels
        
#%%
earlystop1 = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')

earlystop2 = keras.callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')

checkpoint = keras.callbacks.ModelCheckpoint('tmp.h5', 
                                     monitor='val_loss', 
                                     verbose=0, 
                                     save_best_only=False, 
                                     save_weights_only=False, 
                                     mode='auto', 
                                     period=1)

                                     
def schedule(epoch):
    """Defines exponentially decaying learning rate."""
    
    initial_learning_rate =  0.00005  #0.0001 0.00005
    
    lr_decay_start_epoch =  30

    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return 0.00005
        
scheduler = keras.callbacks.LearningRateScheduler(schedule)  

csv_fn        = os.path.join('check', 'log_bnn.csv')

csv_logger  = keras.callbacks.CSVLogger(csv_fn, append=True, separator=';')
        
callbacklist = [TQDMNotebookCallback(leave_inner=True, leave_outer=True),  csv_logger, scheduler]# earlystop1, earlystop2, scheduler]

#%%

kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(450, dtype=tf.float32))
                            
                            
def normal_prior(prior_std):
    """Defines normal distribution prior for Bayesian neural network."""

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
#        tfd = tfp.distributions
        dist = tfd.Normal(loc=tf.zeros(shape, dtype),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn     
    
prior_std = 10 #1 10  30
prior_fn = normal_prior(prior_std)    
    
model_vanilla = Sequential()
model_vanilla.add(tfp.layers.Convolution2DFlipout(50, (5, 5), padding='same', input_shape=(patch_size,patch_size,1), strides=(4, 4),kernel_divergence_fn=kl_divergence_function, kernel_prior_fn=prior_fn, data_format="channels_last",name = 'conv_0'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(tfp.layers.Convolution2DFlipout(50, (3, 3), strides=(2, 2), padding = 'same',kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn,name = 'conv_1'))
#model_vanilla.add(Dropout(0.2))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(tfp.layers.Convolution2DFlipout(50, (3, 3), strides=(2, 2), padding= 'same',kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn,name = 'conv_2'))
#model_vanilla.add(Dropout(0.2))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(tfp.layers.Convolution2DFlipout(50, (3, 3), strides=(2, 2), padding= 'same',kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn,name = 'conv_3'))
#model_vanilla.add(Dropout(0.2))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(tfp.layers.Convolution2DFlipout(50, (3, 3), strides=(2, 2), padding= 'same',kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn,name = 'conv_4'))
model_vanilla.add(Flatten())
model_vanilla.add(tfp.layers.DenseFlipout(50,kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn, name = 'dense_0'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(tfp.layers.DenseFlipout(10,kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn, name = 'dense_1'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(tfp.layers.DenseFlipout(num_classes, kernel_divergence_fn=kl_divergence_function,kernel_prior_fn=prior_fn, name = 'dense_2'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('softmax'))

model_vanilla.summary(line_length=100)

#%%

model_vanilla.compile(loss=lossfkt,
                  optimizer=opt,
                  metrics=metrica, experimental_run_tf_function=False)

t0=time.time()

hist_vanilla = model_vanilla.fit_generator(
    SeismicSequence(
        labeled_data,
        train_data["Xline"].values,
        train_data["Time"].values,
        train_data["Class"].values,
        patch_size,
        batch_size,
        1),
    steps_per_epoch=steps,
    validation_data = SeismicSequence(
        labeled_data,
        test_data["Xline"].values,
        test_data["Time"].values,
        test_data["Class"].values,
        patch_size,
        batch_size,
        1),
    validation_steps = len(test_samples)//batch_size,
    epochs = epochs,
    verbose = 1,
    callbacks = callbacklist)

print('--> Training for Waldeland CNN took {:.1f} sec'.format(time.time()-t0))

#%%

def plot_weight_posteriors(names, qm_vals, qs_vals):

  fig = plt.figure(figsize=(6, 3))

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names[0:2], qm_vals[0:2]):
    sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
  ax.set_title('weight means')
  ax.set_xlim([-1.5, 1.5])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names[0:2], qs_vals[0:2]):
    sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax, label=n)
  ax.set_title('weight stddevs')
  ax.set_xlim([0, 1.])
  ax.legend()

  fig.tight_layout()

names = [layer.name for layer in model_vanilla.layers
                    if 'conv' in layer.name]
qm_vals = [layer.kernel_posterior.mean()
                      for layer in model_vanilla.layers
                      if 'conv' in layer.name]
qs_vals = [layer.kernel_posterior.stddev()
                      for layer in model_vanilla.layers
                      if 'conv' in layer.name]
plot_weight_posteriors(names, qm_vals, qs_vals)




















