import os, sys
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
print("TensorFlow version is ", tf.__version__)
deb = False
deb = True
x = tf.constant([1, 2, 3])
y = tf.broadcast_to(x, [3, 3])
print(y)
d=2 #dim
k=3 # codebook size
n=5 # no of signals

cb = tf.random.normal([k,d])
print(cb)
print(cb.shape)
print(tf.broadcast_to(cb,[3,k,d]))
def sub1():
    tf.random.set_seed(2)

    cols = 100
    cb = tf.random.normal([k,d])
    cb3 = tf.broadcast_to(cb,[n,k,d])
    signals = tf.random.normal([n,d])
    signals3 = tf.broadcast_to(signals,[k,n,d])
    print("cb3",cb3)
    print("signals3",signals3)
    #quit()
    sys.exit
    #
    # determine for each signal the closest codebook vector
    # 
    for i in range(20):
        #x = tf.random.normal([lines,cols])
        v = x[0]
        if deb: print("v=",v)
        diff = x-v
        if deb: print("diff=",diff)
        sqdiff=tf.math.square(diff)
        if deb: print("sqdiff=",sqdiff)
        dists = tf.reduce_sum(sqdiff,axis=1)
        if deb: print("dists=",dists)
        target = tf.math.argmin(dists)
        print(target)
sub1()