r'''
==================================
Kalman Filter tracking a sine wave
==================================

This example shows how to use the Kalman Filter for state estimation.

In this example, we generate a fake target trajectory using a sine wave.
Instead of observing those positions exactly, we observe the position plus some
random noise.  We then use a Kalman Filter to estimate the velocity of the
system as well.

The figure drawn illustrates the observations, and the position and velocity
estimates predicted by the Kalman Smoother.
'''
import numpy as np
import pylab as pl
import tensorflow as tf

from tfkalman import filters

rnd = np.random.RandomState(0)

# generate a noisy sine wave to act as our fake observations
n_timesteps = 100
x_axis = np.linspace(0, 3 * np.pi, n_timesteps)
observations = 20 * (np.sin(x_axis) + 0.5 * rnd.randn(n_timesteps))

n = 1
m = 1
l = 1
x = np.ones([1, 1])

A = np.ones([1, 1])

B = np.zeros([1, 1])

P = np.ones([1, 1])

Q = np.array([[0.005]])

H = np.ones([1, 1])

u = np.zeros([1, 1])

R = np.array([[0.01]])

predictions = []
with tf.Session() as sess:
    kf = filters.KalmanFilter(m=m, n=n, l=l, x=x, A=A, B=B, P=P, Q=Q, H=H)
    predict = kf.predict()
    correct = kf.correct()
    tf.global_variables_initializer().run()
    for i in range(0, n_timesteps):
        print i
        x_pred, _ = sess.run(predict, feed_dict={kf.u: u})
        # print x_pred, p_pred
        predictions.append(x_pred[0, 0])
        sess.run(correct, feed_dict={kf.z:np.array([observations[i]]), kf.R:R})
    # predictions = sess.run(predictions)
    sess.close()

pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x_axis, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x_axis, np.array(predictions),
                        linestyle='-', marker='o', color='r',
                        label='position est.')
# velocity_line = pl.plot(x, states_pred[:, 1],
#                         linestyle='-', marker='o', color='g',
#                         label='velocity est.')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=x_axis.max())
pl.xlabel('time')
pl.show()
