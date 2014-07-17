import numpy as np
from matplotlib import pyplot as plt


def classification(model_func,
                   X=None, Y=None,
                   xlim=None, ylim=None,
                   res=None, xres=100, yres=100):
    if xlim is None:
        assert X is not None
        xmin = np.min(X[:, 0])
        xmax = np.max(X[:, 0])
        xlim = (xmin, xmax)
    if ylim is None:
        assert X is not None
        ymin = np.min(X[:, 1])
        ymax = np.max(X[:, 1])
        ylim = (ymin, ymax)
    if res is not None:
        if isinstance(res, tuple):
            xres, yres = res
        else:
            xres = res
            yres = res
    else:
        assert xres is not None and yres is not None
    xx, yy = np.meshgrid(np.linspace(*xlim, num=xres),
                         np.linspace(*ylim, num=yres))
    xx = xx.astype(X.dtype)
    yy = yy.astype(X.dtype)
    zz = model_func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    hfig = plt.figure()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.contourf(xx, yy, zz, cmap=plt.cm.Paired)

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    return hfig
