from matplotlib import pyplot as plt
import numpy as np

sigma = 0.3
count = 1000

seed = np.random.randint(1000)

np.random.seed(seed)

# Stochastic integration technique
price = 100.0
r1 = price + price * (np.cumsum(np.random.normal(0, 1, count)) * sigma * np.sqrt(1.0 / 252.0))
r2 = np.zeros(count)
r3 = np.zeros(count)

# Log returns
np.random.seed(seed)
returns = np.random.normal(0, 1, count) * sigma
r2[0] = price
for i in range(1, len(r2)):
    r2[i] = r2[i-1] * np.exp((0.0 - 0.5*sigma**2)*(1.0 / 252.0) + returns[i-1] * np.sqrt(1.0 / 252.0))

# Real returns - very close on
r3[0] = price
for i in range(1, len(r3)):
    r3[i] = r3[i-1] * returns[i-1] * np.sqrt(1.0/252.0) + r3[i-1]


np.random.seed(seed)
T = 365
mu = 0.0
sigma = 0.3
S0 = price
dt = 1.0 / 252.0
t = np.linspace(0, (1.0 / 252.0) * count, count)
W = np.random.standard_normal(size=count)
W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
X = (mu-0.5*sigma**2)*t + sigma*W
S = S0*np.exp(X) ### geometric brownian motion ###
plt.plot(S, label='S')
# plt.show()



plt.plot(r1, label='r1')
plt.plot(r2[1:], label='r2')
plt.plot(r3[1:], label='r3')
plt.legend()
plt.show()