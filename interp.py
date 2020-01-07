# -*- coding: utf-8 -*-

from scipy.ndimage import map_coordinates
import numpy as np

'''
Modul za interpolacijo večrazsežnih funkcij.
'''

def interp1(xi, x, f, order=1, **kwargs):
    '''
    Interpolacija 1D signalov definiranih na enakomerni mreži točk.

    Parametri
    ---------
    xi : podatkovno polje
        Točke v katerih je potrebno izračunati (interpolirati)
        funkcijske vrednosti.
    x : vektor
        Enakomerno razporejene vzorčne točke v katerih je funkcijska
        vrednost f znana.
    f : podatkovno polje
        Funkcijske vrednosti v vzorčnih točkah x. Velikost podatkovnega
        polja f mora ustrezati (x.size,).
    order : int, neobvezno
        Red interpolacije. 0 - najbližji sosed, 1 - linearna ...
    kwargs : razno, neobvezno
        Preostali poimensko podani parametri se posredujejo funkciji
        map_coordinates.

    Vrne
    ----
    fxi : podatkovno polje
        Funkcijske vrednosti v točkah f[xi].

    Primer
    ------
    >>> import numpy as np
    >>> from matplotlib import pyplot as pp
    >>>
    >>> x = np.linspace(0, np.pi, 5)
    >>> x_ref = np.linspace(x[0], x[-1], 1000)
    >>> f = np.cos(x)
    >>>
    >>> xi = np.linspace(0, np.pi, 50)
    >>> fi_lin = interp1(xi, x, f)
    >>> fi_quad = interp1(xi, x, f, 2)
    >>>
    >>> pp.figure()
    >>> pp.plot(x_ref, np.cos(x_ref), '-k', label='cos(x)')
    >>> pp.plot(x, f, 'or', label='vzorčne točke', markersize=6)
    >>> pp.plot(xi, fi_lin, 'xg', label='linearna interpolacija', markersize=6)
    >>> pp.plot(xi, fi_quad, 'xb', label='kvadratična interpolacija', markersize=6)
    >>> pp.legend()
    '''
    if x is not None:
        x = np.asarray(x).flatten()
        if x.size != f.size:
            raise IndexError('Length of vector x and f must be the same.')
        indx = (xi - x[0])*((x.size - 1)/(x[-1] - x[0]))
    else:
        indx = xi

    return map_coordinates(np.asarray(f), np.asarray([indx]),
                           order=order, **kwargs)

def interp2(xi, yi, x, y, f, order=1, **kwargs):
    '''
    Interpolacija 2D funkcij definiranih na enakomerni mreži točk. Pri tem
    se smatra, da prva razsežnost pripada y koordinatni osi, druga razsežnost
    pa x koordinatni osi, in sicer kot f[y, x].

    Parametri
    ---------
    xi : podatkovno polje
        X komponente točk v katerih je potrebno izračunati (interpolirati)
        funkcijske vrednosti.
    yi : podatkovno polje
        Y komponente točk v katerih je potrebno izračunati (interpolirati)
        funkcijske vrednosti.
    x : vektor
        Enakomerno razporejene vzorčne točke vzdolž x koordinatne osi.
    y : vektor
        Enakomerno razporejene vzorčne točke vzdolž y koordinatne osi.
    f : podatkovno polje
        Funkcijske vrednosti na 2D mreži vzorčnih točkah, ki jo napenjata
        vektorja x in y. Velikost podatkovnega polja f
        mora ustrezati (y.size, x.size).
    order : int, neobvezno
        Red interpolacije. 0 - najbližji sosed, 1 - linearna ...
    kwargs : razno, neobvezno
        Preostali poimensko podani parametri se posredujejo funkciji
        map_coordinates.

    Vrne
    ----
    fxyi : podatkovno polje
        Funkcijske vrednosti v točkah f[yi, xi].

    Primer
    ------
    >>> import numpy as np
    >>> from matplotlib import pyplot as pp
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>>
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 10)
    >>> Y, X = np.meshgrid(y, x, indexing='ij')
    >>> f = 1.0/(X**2 + Y**2 + 1)
    >>>
    >>> xi = np.linspace(0, 1, 30)
    >>> yi = np.linspace(0, 1, 30)
    >>> Yi, Xi = np.meshgrid(yi, xi, indexing='ij')
    >>> fi = interp2(Xi, Yi, x, y, f)
    >>>
    >>> fig = pp.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.plot_wireframe(X, Y, f, color='r', label='vzorčne točke')
    >>> ax.plot_wireframe(Xi, Yi, fi, color='g', label='interpolirane vrednosti')
    >>> ax.legend()
    '''

    f = np.asarray(f)
    xi, yi = np.asarray(xi), np.asarray(yi)

    if x is not None:
        x = np.asarray(x)
        if x.size != f.shape[1]:
            raise IndexError('Length of vector x must match ' \
                'the number of columns of f.')
        indx = (xi - x[0])*((x.size - 1)/(x[-1] - x[0]))
    else:
        indx = xi

    if y is not None:
        y = np.asarray(y)
        if y.size != f.shape[0]:
            raise IndexError('Length of vector y must match ' \
                'the number of rows of f.')
        indy = (yi - y[0])*((y.size - 1)/(y[-1] - y[0]))
    else:
        indy = yi

    return map_coordinates(f, np.asarray([indy, indx]),
                           order=order)

def interp3(xi, yi, zi, x, y, z, f, order=1, **kwargs):
    '''
    Interpolacija 3D funkcij definiranih na enakomerni mreži točk. Pri tem
    se smatra, da prva razsežnost pripada z koordinatni osi, druga razsežnost
    y koordinatni osi, tretja razsežnost pa x koordinatni osi, in sicer kot
    f[z, y, x].

    Parametri
    ---------
    xi : podatkovno polje
        X komponente točk v katerih je potrebno izračunati (interpolirati)
        funkcijske vrednosti.
    yi : podatkovno polje
        Y komponente točk v katerih je potrebno izračunati (interpolirati)
        funkcijske vrednosti.
    zi : podatkovno polje
        Z komponente točk v katerih je potrebno izračunati (interpolirati)
        funkcijske vrednosti.
    x : vektor
        Enakomerno razporejene vzorčne točke vzdolž x osi.
    y : vektor
        Enakomerno razporejene vzorčne točke vzdolž y osi.
    z : vektor
        Enakomerno razporejene vzorčne točke vzdolž z osi.
    f : podatkovno polje
        Funkcijske vrednosti na 3D mreži vzorčnih točkah, ki jo napenjajo
        vektorji x, y in z. Velikost podatkovnega polja f
        mora ustrezati (z.size, y.size, x.size).
    order : int, neobvezno
        Red interpolacije. 0 - najbližji sosed, 1 - linearna ...
    kwargs : razno, neobvezno
        Preostali poimensko podani parametri se posredujejo
        funkciji map_coordinates.

    Vrne
    ----
    fxyzi : podatkovno polje
        Funkcijske vrednosti v točkah f[zi, yi, xi].

    Primer
    ------
    >>> import numpy as np
    >>>>
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 10)
    >>> z = np.linspace(-1, 1, 10)
    >>> Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    >>> f = 1.0/(X**2 + Y**2 + Z**2 + 1)
    >>>
    >>> xi = np.linspace(0, 1, 30)
    >>> yi = np.linspace(0, 1, 30)
    >>> zi = np.linspace(0, 1, 30)
    >>> Zi, Yi, Xi = np.meshgrid(zi, yi, xi, indexing='ij')
    >>>
    >>> fi = interp3(Xi, Yi, Zi, x, y, z, f)
    '''
    f = np.asarray(f)
    xi, yi = np.asarray(xi), np.asarray(yi)

    if x is not None:
        x = np.asarray(x)
        if x.size != f.shape[2]:
            raise IndexError('Length of vector x must match ' \
                'the number of columns in f.')
        indx = (xi - x[0])*((x.size - 1)/(x[-1] - x[0]))
    else:
        indx = xi

    if y is not None:
        y = np.asarray(y)
        if y.size != f.shape[1]:
            raise IndexError('Length of vector y must match ' \
                'the number of rows in f.')
        indy = (yi - y[0])*((y.size - 1)/(y[-1] - y[0]))
    else:
        indy = yi

    if z is not None:
        z = np.asarray(z)
        if z.size != f.shape[0]:
            raise IndexError('Length of vector z must match ' \
                'the number of slices in f.')
        indz = (zi - z[0])*((z.size - 1)/(z[-1] - z[0]))
    else:
        indy = zi

    return map_coordinates(f, np.asarray([indz, indy, indx]),
                           order=order, **kwargs)

def interpn(ti, t, f, order=1, **kwargs):
    '''
    Interpolacija ND funkcij definiranih na enakomerni mreži točk.

    Parametri
    ---------
    ti : podatkovno polje
        Seznam komponent točk v katerih je potrebno izračunati
        (interpolirati) funkcijske vrednosti.
    t : podatkovno polje
        Seznam vektorjev enakomerno razporejenih vzorčnni točk vzdolž vseh
        koordinatnih osi (razsežnosti).
    f : podatkovno polje
        Funkcijske vrednosti na ND mreži vzorčnih točkah, ki jo napenjajo
        vektorji v t. Velikost podatkovnega polja f mora
        ustrezati (t[0].size, t[1].size, ..., t[-1].size).
    order : int, neobvezno
        Red interpolacije. 0 - najbližji sosed, 1 - linearna ...
    kwargs : razno, neobvezno
        Preostali poimensko podani parametri se posredujejo funkciji
        map_coordinates.

    Vrne
    ----
    fti : podatkovno polje
        Funkcijske vrednosti v točkah f[t[0], t[1], ..., t[-1]].

    Primer
    ------
    >>> import numpy as np
    >>> from matplotlib import pyplot as pp
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>>
    >>> x = np.linspace(-1, 1, 15)
    >>> y = np.linspace(-1, 1, 10)
    >>> Y, X = np.meshgrid(y, x, indexing='ij')
    >>> f = 1.0/(X**2 + Y**2 + 1)
    >>>
    >>> xi = np.linspace(0, 1, 30)
    >>> yi = np.linspace(0, 1, 30)
    >>> Yi, Xi = np.meshgrid(yi, xi, indexing='ij')
    >>> fi = interpn([Yi, Xi], [y, x], f)
    >>>
    >>> fig = pp.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.plot_wireframe(X, Y, f, color='r', label='vzorčne točke')
    >>> ax.plot_wireframe(Xi, Yi, fi, color='g', label='interpolirane vrednosti')
    >>> ax.legend()
    '''

    f = np.asarray(f)

    if t is not None:
        N = len(t) # space dimensionality
        if N != len(ti):
            raise IndexError('Dimensions of the coordinates must agree!')

        tind = []
        for i in range(N):
            if t[i] is not None:
                tmp = t[i].flatten()
                tind.append((ti[i] - tmp[0])*
                            ((tmp.size - 1)/(tmp[-1] - tmp[0])))
            else:
                tind.append(ti[i])
    else:
        tind = ti
    return map_coordinates(f, np.asarray(tind), order=order, **kwargs)
