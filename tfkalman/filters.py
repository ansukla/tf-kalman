import tensorflow as tf
import numpy as np

# X : State at step k - 1 (apriori) [n]
# P : state error covariance at step k - 1 (apriori) [n, n]
# A : transition matrix [n, n]
# Q : The process noise covariance matrix.
# B : The input effect matrix.
# U : The control input.

class KalmanFilter(object):

    def __init__(self, m=1, n=1, l=1, x=None, A=None, P=None, B=None, H=None, Q=None):
        """Initialize a filter

        Parameters
        ----------
        m : int - measurement size
        n : int - state size
        l : int - control input size
        x : float32 [n, 1]  - initial state
        A : float32 [n, n] -  state transition matrix
        Q : float32 [n, n] -  process noise covariance
        u : float32 [l, 1] -  control input
        B : float32 [n, l] -  control input transition matrix
        z : float32 [m, 1] -  measurement
        R : float32 [m, m] -  measurement noise covariance
        H : float32 [m, n] -  measurement transition matrix
        """

        self._m = m
        self._n = n
        self._l = l
        self._x = tf.Variable(x, dtype=tf.float32, name="x")
        self._A = tf.constant(A, dtype=tf.float32, name="A")
        self._P = tf.Variable(P, dtype=tf.float32, name="P")
        self._B = tf.constant(B, dtype=tf.float32, name="B")
        self._Q = tf.constant(Q, dtype=tf.float32, name="Q")
        self._H = tf.constant(H, dtype=tf.float32, name="H")

    # returns
    # X = a priori projected state at step k
    # P = projected error covariance at step k
    def predict(self, u):
        x = self._x
        A = self._A
        P = self._P
        B = self._B
        u = tf.constant(u, dtype=tf.float32, name="u")
        Q = self._Q
        self._x = tf.matmul(A, x) + tf.matmul(B, u)
        self._P = tf.matmul(A, tf.matmul(P, A, transpose_b=True)) + Q
        return self._x, self._P

    def correct(self, z, R):
        x = self._x
        P = self._P
        H = self._H
        if not tf.is_numeric_tensor(z):
            z = tf.constant(z, dtype=tf.float32, name="u")
        if not tf.is_numeric_tensor(R):
            R = tf.constant(R, dtype=tf.float32, name="z")

        K = tf.matmul(P, tf.matmul(tf.transpose(H), tf.matrix_inverse(tf.matmul(H, tf.matmul(P, H, transpose_b=True)) + R)))
        x = x + tf.matmul(K, z - tf.matmul(H, x))
        P = tf.matmul((1 - tf.matmul(K, H)), P)
        return K, x, P

    @property
    def x(self):
        return self._x

    @property
    def A(self):
        return self._A

    @property
    def P(self):
        return self._P

    @property
    def Q(self):
        return self._Q

    @property
    def B(self):
        return self._B

    @property
    def u(self):
        return self._u

    @property
    def H(self):
        return self._H

    @property
    def R(self):
        return self._R

    @property
    def z(self):
        return self._z
