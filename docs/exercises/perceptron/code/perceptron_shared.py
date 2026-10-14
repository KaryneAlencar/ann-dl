import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.figsize': (9,6), 'axes.grid': True, 'grid.alpha': .25})

def prepare_sep():
    rng = np.random.default_rng(42)
    mu0, mu1 = np.array([1.5,1.5]), np.array([5.,5.])
    cov = np.diag([.5,.5])
    X0 = rng.multivariate_normal(mu0,cov,size=1000)
    X1 = rng.multivariate_normal(mu1,cov,size=1000)
    X = np.vstack((X0,X1))
    y = np.concatenate((np.zeros(1000,dtype=int),np.ones(1000,dtype=int)))
    return rng,X,y

def prepare_overlap():
    rng, X_sep, y_sep = prepare_sep()
    w_initial_sep = rng.normal(0,.01,size=2)
    cov = np.diag([1.5,1.5])
    X0 = rng.multivariate_normal([3.,3.],cov,size=1000)
    X1 = rng.multivariate_normal([4.,4.],cov,size=1000)
    X = np.vstack((X0,X1))
    y = np.concatenate((np.zeros(1000,dtype=int),np.ones(1000,dtype=int)))
    return rng,X,y

def plot_boundary(ax,w,b,x_limits,label,linestyle='-'):
    x1 = np.linspace(*x_limits,300)
    if abs(w[1]) > 1e-12:
        ax.plot(x1,-(w[0]*x1+b)/w[1],linestyle=linestyle,linewidth=2,label=label)
    elif abs(w[0]) > 1e-12:
        ax.axvline(-b/w[0],linestyle=linestyle,linewidth=2,label=label)
