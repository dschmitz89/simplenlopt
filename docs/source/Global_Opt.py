import numpy as np
import simplenlopt
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import rosen, rosen_der

x0 = [-1., 2.]

res = simplenlopt.minimize(rosen, x0, method='neldermead')
print(res.x)
print(res.nfev)

res = simplenlopt.minimize(rosen, x0, jac = rosen_der, method = 'slsqp')
print(res.x)
print(res.nfev)
'''
def styblinski_tang(p):
    x, y = p
    return 0.5 * (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y)

xgrid = np.linspace(-4, 4, 100)
ygrid = np.linspace(-4, 4, 100)

X, Y = np.meshgrid(xgrid, ygrid)
Z = 0.5 * (X**4 - 16 * X**2 + 5 * X + Y**4 - 16 * Y**2 + 5 * Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, linewidth = 0, cmap = cm.viridis, antialiased=False)

bounds = [(-5., 5.), (-5., 5.)]
res_direct = simplenlopt.direct(styblinski_tang, bounds)
ax.scatter(res_direct.x[0], res_direct.x[1], res_direct.fun, c='r', s = 70, label='DIRECT')

x0 = np.array([0., 0.])
res_bobyqa = simplenlopt.minimize(styblinski_tang, x0)
ax.scatter(res_bobyqa.x[0], res_bobyqa.x[1], res_bobyqa.fun, c='k', s = 70, label='BOBYQA')
ax.view_init(11, -150)
ax.legend()

plt.savefig("Styblinski_Tang.png", dpi = 300)
'''