import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

t_min, t_max = 0, 30
resolution = 0.1
def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps,1),ys[:,1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max,int((t_max-t_min)/resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series(generated)", fontsize=14)

plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2. \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]),"b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0,30,-17,13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instnace")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()