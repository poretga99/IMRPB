import numpy as np
import interp
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def im_load_raw_2d(filename, height, width, dtype_=np.uint8):
    data = np.fromfile(filename, dtype = dtype_)
    slika = np.reshape(data, (height, width))
    return slika

def im_save_raw_2d(filename, data):
    data.tofile(filename)


def im_scale(img, slope, intersection):
    img = np.asarray(img, dtype=np.float)
    return img*slope + intersection


def im_window(img, center, width, ls=255):
    c, w = center, width
    oimg = np.zeros(img.shape)
    mask1 = img>(c+w/2)
    mask2 = np.logical_and(img>=(c - w/2), img <= (c + w/2))
    oimg[mask1]=ls
    oimg[mask2]=ls/w*(img[mask2] - (c-w/2))
    return oimg


def im_threshold(img, threshold, ls=255):
    oimg = np.zeros(img.shape)
    mask = img >= threshold
    oimg[mask] = ls
    return oimg

def im_load_raw_3d(filename, width, height, depth, dtype=np.uint8):
    data = np.fromfile(filename, dtype = dtype)
    slika = np.reshape(data, (depth, height, width))
    return slika


def gaussian_kernel_2d(sigma, truncate=5):
    n = np.ceil(sigma*truncate/2)*2+1
    u = np.arange(n, dtype = np.float)
    u = u - np.mean(u)
    v = u
    U, V = np.meshgrid(u, v, indexing='ij')
    K = 1.0/(2*np.pi*sigma**2)*np.exp(-(U**2+V**2)/(2.0*sigma**2))
    K = K/K.sum()
    
    return K
    
def im_sharpen_2d(img, kind='mask', c=1.0, sigma=1.0, **kwargs):
    img = np.asarray(img, dtype = np.float)
    if kind=="mask":
        M = img - gaussian_filter(img, sigma=sigma, **kwargs)
        S = img + c*M
    elif kind=="laplace":
        K = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float)
        S = img - c*convolve(img, K)
    else:
        raise ValueError('Napacna vrednost parametra')
    return S


def im_shading_calibrate(img, dark, bright, d=0, b=255):
    img = np.asarray(img, dtype=np.float)
    dark = np.asarray(dark, dtype=np.float)
    bright = np.asarray(bright, dtype=np.float)
    
    ofm = (bright - dark)/(b-d)
    ofa = (b*dark - d*bright)/(b-d)
    
    oimg = (img - ofa)/ofm
    
    return oimg, ofa, ofm


def standardiziraj(img, ref):
    img = np.asarray(img, dtype=np.float)
    ref = np.asarray(ref, dtype=np.float)
    return (img - img.mean())*ref.std()/img.std() + ref.mean();


def im_shading_filter(img, sigma):
    img = np.asarray(img, dtype=np.float)
    of = gaussian_filter(img, sigma=sigma)
    oimg = img - of
    return oimg, of

def im_shading_homomorphic(img, sigma):
    img = np.asarray(img, dtype=np.float)
    log_img = np.log(img + 1.0)
    log_of = gaussian_filter(log_img, sigma=sigma)
    #log_oimg = log_img - log_of
    #oimg = np.exp(log_oimg)-1.0
    of = np.exp(log_of) - 1.0
    oimg = img/of
    return oimg, of


def im_shading_calibrate_dodatek(img, dark, bright, *args, **kwargs):
    # v args poberemo vrednost sigma
    # v kwargs poberemo vrednosti a in b
    if dark is None:
        return im_shading_homomorphic(img, *args)
    elif bright is None:
        return im_shading_filter(img, *args)
    else:
        return im_shading_calibrate(img, dark, bright, **kwargs)

    
def transform_affine_2d(scale=(1.0, 1.0), trans=(0.0, 0.0), rot=0.0, shear=(0.0, 0.0), proj=(0.0, 0.0)):
    # kot podan v radianih!!!!!!!
    Tscale = np.array([[scale[0], 0.0, 0.0],
                       [0.0, scale[1], 0.0],
                       [0.0, 0.0 , 1.0]])
    Trot = np.array([[np.cos(rot), -np.sin(rot), 0.0],
                       [np.sin(rot), np.cos(rot), 0.0],
                       [0.0, 0.0, 1.0]])
    Ttrans = np.array([[1.0, 0.0, trans[0]],
                       [0.0, 1.0, trans[1]],
                       [0.0, 0.0, 1.0]])
    Tshear = np.array([[1.0, shear[0], 0.0],
                       [shear[1], 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    Tproj = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [proj[0], proj[1], 1]])
    
    return Tproj@Tshear@Trot@Ttrans@Tscale

def pt_transform_2d(T, x, y, inverse=False):
    T = np.asarray(T, dtype=np.float)
    x = np.asarray(x, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    xy_in = np.ones((3, x.size), dtype=np.float)
    xy_in[0] = x.flatten()
    xy_in[1] = y.flatten()
    
    if not inverse:
        xy_out = T@xy_in
    else:
        #lahko tudi np.linalg.inv(T)@xy_in
        xy_out = np.linalg.solve(T, xy_in)
    
    x_out = xy_out[0]/xy_out[2]
    y_out = xy_out[1]/xy_out[2]
    return x_out.reshape(x.shape), y_out.reshape(y.shape)


def im_transform_2d(img, t, x=None, y=None, center=(0.0, 0.0)):
    img = np.asarray(img, dtype=np.float)
    H, W = img.shape
    if x is None:
        x = np.arange(W, dtype = np.float) - center[0]
    if y is None:
        y = np.arange(H, dtype = np.float) - center[1]
    
    Y, X = np.meshgrid(y, x, indexing='ij')
    Xt, Yt = pt_transform_2d(t, X, Y, inverse=True)
    oimg = interp.interp2(Xt, Yt, x, y, img)
    return oimg

import numpy as np

from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy import optimize
from scipy.signal import convolve

import interp

################################################################################

#%% Splošna orodja

################################################################################

#%% Splošna orodja

class getpts:
  def __init__(self, n=1, marker='xr', verbose=False):
    '''
    Omogoča izbiranje/označevanje točk na trenutnem grafu.

    Parametri
    ---------
    n: int
        Število točk, ki jih želimo označiti.
    marker: str
        Grafična podoba izrisa označenih točk.
    verbose: bool
        Če je vrednost True, izpiše podrobno informacijo o označeni ali
        izbrisani točki.

    Opombe
    ------
    Za označbo nove točke je potrebno aktivirati Shift tipko in nato pritisniti
    levi gumb miške.
    Za izbris zadnje označene točke je potrebno aktivirati Ctrl tipko in nato
    pritisniti levi gumb miške.
    Zadnjo označeno točko je mogoče izbrisati tudi s pritiskom na tipko
    Backspace.
    '''
    self._verbose = bool(verbose)
    self._xy = []
    self._pt_items = []
    self._marker = marker
    self._n = int(n)
    self._fig = pp.gcf()
    self._axis = pp.gca()
    self._click_handler = self._fig.canvas.mpl_connect(
      'button_press_event', self._onclick)
    self._key_handler = self._fig.canvas.mpl_connect(
      'key_press_event', self._onkey)

  def xy(self):
    return self._xy

  def _onclick(self, event):
    if self._verbose:
      print('Clicked at ({}, {}), with key modifier "{}"\n'.format(
        event.xdata, event.ydata, event.key))

    if event.key and event.key.lower() == 'shift':
      # add point
      if event.button == 1:
        if len(self._pt_items) < self._n:
          x, y = event.xdata, event.ydata
          self._xy.append((x, y))
          pt_item = self._axis.plot(x, y, self._marker)
          self._pt_items.append(pt_item)
          self._fig.canvas.draw()

    elif event.key and event.key.lower() in ('control', 'ctrl'):
      # remove point
      if event.button == 1:
        # remove point
        self._delete_point()

  def _onkey(self, event):
    if self._verbose:
      print('Key press event on "{}"'.format(event.key))

    if event.key and event.key.lower() == 'backspace':
      self._delete_point()
    else:
      return event

  def _delete_point(self):
    if len(self._pt_items) > 0:
      self._xy.pop(-1)
      lines = self._pt_items.pop(-1)
      for item in lines:
        item.remove()
      self._fig.canvas.draw()
      del lines

  def __del__(self):
    self._fig.canvas.mpl_disconnect(self._click_handler)
    self._fig.canvas.mpl_disconnect(self._key_handler)

################################################################################

#%% vaja 7

def transform_estimate_2d(x, y, u, v, kind='rigid', nr=1, 
                          animate=False):
  '''
  Določi izbrano intrerpolacijsko ali aproksimacjsko preslikavo na
  podlagi podanih korespondečnih točk (x, y) -T-> (u,v).
  
  Parametri
  ---------
  x, y: np.ndarray vektor
    Koordinate korespondečnih točk v referenčni sliki.
  u, v: np.ndarray vektor
    Koordinate korespondenčnih točk v zajeti sliki.
  kind: str
    Model preslikave 'rigid', 'affine', 'projective', ali radial.
  nr: int
    Red radialnih distorzij.
  animate: bool
    Ko je vrednost parametra True, se prikazujejo vmesni rezultati 
    optimizacijskega postopka.

  Vrne
  ----
  t: np.ndarray
    Transformacijska matrika velikosti 3 x 3.
  tr: np.ndarray vektor
    Parametri radialnih distrorzij. Funkcija vrne vrednost le takrat, ko
    je vrednost parametra kind enaka "radial".
  err: float
    Napaka poravnave med korespondenčnimi točkami.

  Primeri uporabe
  ---------------
  >>> import numpy as np
  >>>
  >>> x, y = np.random.rand(10), np.random.rand(10)
  >>> t = transform_affine_2d(scale=[2.0, 0.5], trans=[5.0, 2.0], rot=np.pi/4, shear=[0.2, 0.5])
  >>> u, v = pt_transform_2d(t, x, y)
  >>> te, err = transform_estimate_2d(x, y, u, v, kind='affine')
  >>> t, te
  >>>
  '''
  x = np.asarray(x).astype(np.float).flatten()
  y = np.asarray(y).astype(np.float).flatten()
  u = np.asarray(u).astype(np.float).flatten()
  v = np.asarray(v).astype(np.float).flatten()
  t = tr = err = None

  if kind == 'rigid':
    xm = x.mean()
    um = u.mean()
    ym = y.mean()
    vm = v.mean()
    yum = np.mean(y*u)
    xvm = np.mean(x*v) 
    xum = np.mean(x*u)
    yvm = np.mean(y*v)
    a = -np.arctan((yum - xvm - ym*um + xm*vm)/
      (xum +  yvm - xm*um - ym*vm))
    tx = um - xm*np.cos(a) + ym*np.sin(a)
    ty = vm - xm*np.sin(a) - ym*np.cos(a)
    t = transform_affine_2d(trans=[tx, ty], rot=a)
    tx, ty = pt_transform_2d(t, x, y)
    err = (((u - tx)**2 + (v - ty)**2).mean())**0.5
    return t, err

  elif kind == 'affine':
    xxm = np.mean(x*x)
    xym = np.mean(x*y)
    yym = np.mean(y*y)
    xm = x.mean()
    ym = y.mean() 
    uxm = np.mean(x*u)
    uym = np.mean(u*y) 
    vxm = np.mean(v*x)
    vym= np.mean(y*v)
    um = u.mean()
    vm = v.mean()
    t = np.zeros([3,3])
    t[-1, -1] = 1.0
    tvec = np.linalg.solve(
      np.array(
        [[xxm, xym, xm, 0, 0, 0],
        [xym, yym, ym, 0, 0, 0],
        [xm, ym, 1, 0, 0, 0],
        [0, 0, 0, xxm, xym, xm],
        [0, 0, 0, xym, yym, ym],
        [0, 0, 0, xm, ym, 1]]),
      np.array([uxm, uym, um, vxm, vym, vm]))
    t[0] = tvec[:3]
    t[1] = tvec[3:]
    tx, ty = pt_transform_2d(t, x, y)
    err = (((u - tx)**2 + (v - ty)**2).mean())**0.5
    return t, err

  elif kind == 'projective':
    if animate:
      pp.figure()
    n_iter = np.zeros((1,), dtype=np.int)
    t0, err = transform_estimate_2d(x, y, u, v, 'affine')
    t0 = t0.flatten()[:-1]
    topt = optimize.fmin(
      lambda t: _k_fun_projective_2d(t, x, y, u, v, animate, n_iter), t0)
    t = np.ones([9], dtype=np.float)
    t[:8] = topt
    t.shape = [3, 3]
    err = _k_fun_projective_2d(topt, x, y, u, v, False)
    return t, err

  elif kind == 'radial':
    if animate:
      pp.figure()
    n = np.zeros((1,), dtype=np.int)
    Tproj = np.zeros([3, 3])
    xc = x.mean()
    yc = y.mean()
    tr0 = np.zeros([nr + 2])
    tr0[0] = xc
    tr0[1] = yc
    tropt = optimize.fmin(
      lambda tr: _k_fun_projective_radial_2d(
        tr, x, y, u, v, Tproj, animate, n), tr0)
    tr = tropt
    err = _k_fun_projective_radial_2d(tropt, x, y, u, v, Tproj, False, 0)
    return Tproj, tr, err

  return t, tr, err

def _k_fun_projective_2d(t, x, y, u, v, animate, n=0):
  '''
  Kriterijska funkcija za določevanje parametrov projektivne preslikave na
  podlagi podanega seta korespondenčnih točk (x,y) ter (u,v).

  Parametri
  ---------
  t: np.ndarray vektor
    Trenutna vrednost parametrov projektivne preslikave, ki so zapisani v
    vektorju (po vrsticah).
  x, y: np.ndarray
    Koordinate korespondečnih točk v referenčni sliki.
  u, v: np.ndarray
    Koordinate korespondenčnih točk v zajeti sliki.
  animate: bool
    Ko je vrednost parametra True, se prikazujejo vmesni rezultati 
    optimizacijskega postopka.
  n: np.int
    Števec iteracij.

  Vrne
  ----
  err: float
      Vrne koren srednje kvadratične razdalje med preslikanimi referenčnimi 
      točkami in točkami v zajeti sliki.
  '''
  T = np.zeros([9])
  T[:8] = t
  T[-1] = 1.0
  T.shape = (3,3)
  ue, ve = pt_transform_2d(T, x, y)
  err = (((u - ue)**2 + (v - ve)**2).mean())**0.5

  if animate and int(n) % 10 == 0:
    ax = pp.gca()
    ax.title.set_text('Koren povp. kvadr. razdalje: {:.3f} slik. el. '
                      '@iter. {}'.format(err, int(n) + 1))
    if ax.lines:
        ax.lines[0].set_data(u, v)
        ax.lines[1].set_data(ue, ve)
    else:
        pp.plot(u, v, 'xr')
        pp.plot(ue, ve, 'xb')
    pp.gcf().canvas.draw()
    pp.pause(0.1)

  n += 1
  return err




def im_sm(imga, imgb, sm, nb=64, nb_ab=16, span=(0,255)):
    imga = np.asarray(imga, dtype=np.float)
    imgb = np.asarray(imgb, dtype=np.float)
    N = imga.size
    
    sm = str(sm).lower()
    
    if sm == 'mae':
        sm_f = np.abs(imga-imgb).mean()
    elif sm == 'mse':
        sm_f = ((imga-imgb)**2).mean()
    elif sm == 'cc':
        ap = imga.mean()
        bp = imgb.mean()
        
        sm_f = (((imga-ap)*(imgb-bp)).sum())/(((imga-ap)**2).sum()*((imgb-bp)**2).sum())**0.5
    elif sm == 'mi':
        pa = np.histogram(imga, bins=nb, range=span)
        pb = np.histogram(imgb, bins=nb, range=span)
        pa_val = pa[0]/imga.size
        pb_val = pb[0]/imgb.size
        hist_idx = pa[1]
        
        Ha = 0
        Hb = 0
        for i in range(len(pa_val)):
            if pa_val[i] > 0:
                Ha -= pa_val[i]*np.log(pa_val[i])
            if pb_val[i] > 0:
                Hb -= pb_val[i]*np.log(pb_val[i])
        
        
        # Komentar
        # Vsi bini so inclusive-exclusive razen zadnji je inclusive-inclusive, zato je par if-ov da pregledujejo kakšno je stanje
        pab = np.zeros((nb_ab, nb_ab), dtype=np.float)
        hab_idx = np.linspace(span[0], span[1], nb_ab+1)
        for i in range(nb_ab):
            for j in range(nb_ab):
                if i == nb_ab-1:
                    tmp_a = np.multiply(imga<hab_idx[j+1], imga>=hab_idx[j])
                    tmp_b = np.multiply(imgb<=hab_idx[i+1], imgb>=hab_idx[i])
                    pab[i,j] = np.sum(np.multiply(tmp_a, tmp_b))
                elif j == nb_ab - 1:
                    tmp_a = np.multiply(imga<=hab_idx[j+1], imga>=hab_idx[j])
                    tmp_b = np.multiply(imgb<hab_idx[i+1], imgb>=hab_idx[i])
                    pab[i,j] = np.sum(np.multiply(tmp_a, tmp_b))
                elif i == nb_ab - 1 and j == nb_ab - 1:
                    tmp_a = np.multiply(imga<=hab_idx[j+1], imga>=hab_idx[j])
                    tmp_b = np.multiplty(imgb<=hab_idx[i+1], imgb>=hab_idx[i])
                    pab[i,j] = np.sum(np.multiply(tmp_a, tmp_b))
                else:
                    tmp_a = np.multiply(imga<hab_idx[j+1], imga>=hab_idx[j])
                    tmp_b = np.multiply(imgb<hab_idx[i+1], imgb>=hab_idx[i])
                    pab[i,j] = np.sum(np.multiply(tmp_a, tmp_b))
        pab = pab / imga.size
        Hab = 0
        for i in range(nb_ab):
            for j in range(nb_ab):
                if pab[i, j] > 0:
                    Hab -= pab[i, j]*np.log(pab[i, j])
        MI = Ha + Hb - Hab
        #raise ValueError('Implementiraj me')
        sm_f = MI
    #print(sm_f)
    return sm_f

def im_rigid_register(imga, imgb, sm, x0=[0.0, 0.0, 0.0], animate=False):
    imga = np.asarray(imga, dtype=np.float)
    imgb = np.asarray(imgb, dtype=np.float)
    
    sm = str(sm).lower()
    
    if animate:
        plt.figure()
    n_iter = np.zeros([1,], dtype = np.int)
    ox = optimize.fmin(lambda x: kfun_rigid_register(x, imga, imgb, sm, animate, n_iter), x0)
    of = kfun_rigid_register(ox, imga, imgb, sm, animate, n_iter)
    ot = transform_affine_2d(trans=ox[:2], rot=ox[2])
    
    
    return ox, ot, of, n_iter
    
def kfun_rigid_register(ox, imga, imgb, sm, animate, n_iter):
    T = transform_affine_2d(trans=ox[:2], rot=ox[2])
    float_img = im_transform_2d(imgb, T)
    
    f = im_sm(imga, float_img, sm)
    if sm=='cc':
        # poravnani sliki -1 ali 1
        f = 1.0 - np.abs(f)
    elif sm == 'mi':
        f = 1/f
        #raise RuntimeError('Implementiraj me!')
    
    if animate and n_iter % 10 == 0:
        ax = plt.gca()
        if ax.images:
            ax.images[0].set_data(imga-float_img)
        else:
            plt.imshow(np.clip(imga - float_img + 127.0 , 0.0, 255.0), cmap='gray')
        plt.gcf().canvas.draw()
        plt.pause(0.5)
    
    n_iter += 1
    return f
