import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import PIL.Image as im
import funkcije as fun


def normalizeImage(iImage, type='whitening'):
    if type=='whitening':
        oImage = (iImage - np.mean(iImage)) / np.std(iImage)
    elif type=='range':
        oImage = (iImage - np.min(iImage)) / (np.max(iImage) - np.min(iImage))
    return oImage


def getChessBoardImage(iImageSize, iArraySize=10, dtype='uint8'):
    dy = int(np.ceil(iImageSize[0] / iArraySize)) + 1
    dx = int(np.ceil(iImageSize[1] / iArraySize)) + 1

    A = [255 * np.ones(shape=(iArraySize, iArraySize)),
         np.zeros(shape=(iArraySize, iArraySize))]
    board = np.array(np.vstack([np.hstack([A[(i + j) % 2] \
                                           for i in range(dx)]) \
                                for j in range(dy)]), dtype=dtype)
    return board[:iImageSize[0], :iImageSize[1]]


def swirlControlPoints(iCPx, iCPy, a=2.0, b=100.0):
    oCPx = np.array(iCPx)
    oCPy = np.array(iCPy)
    xc = np.mean(oCPx[1:-3,1:-3])
    yc = np.mean(oCPy[1:-3,1:-3])
    rx1 = oCPx[1:-3,1:-3] - xc
    ry1 = oCPy[1:-3,1:-3] - yc
    angle = a*np.exp(-(rx1*rx1+ry1*ry1)/(b*b))
    oCPx[1:-3,1:-3] = np.cos(angle)*rx1 + np.sin(angle)*ry1 + xc
    oCPy[1:-3,1:-3] = -np.sin(angle)*rx1 + np.cos(angle)*ry1 + xc
    return oCPx, oCPy


def B0(u):
    # YOUR CODE HERE
    return (1 - u) ** 3 / 6


def B1(u):
    # YOUR CODE HERE
    return (3 * u ** 3 - 6 * u ** 2 + 4) / 6


def B2(u):
    # YOUR CODE HERE
    return (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6


def B3(u):
    # YOUR CODE HERE
    return u ** 3 / 6


B = [B0, B1, B2, B3]

def getCubicBSpline2DGrid(iImageSize, iStep):
    # YOUR CODE HERE
    dy, dx = iImageSize
    if not isinstance(iStep, (tuple, list)):
        iStep = (iStep, iStep)
    oCPx, oCPy = np.meshgrid(
        np.arange(-iStep[0], (np.floor(dx/iStep[0]+3)*iStep[0]), iStep[0]),
        np.arange(-iStep[1], (np.floor(dy/iStep[1]+3)*iStep[1]), iStep[1]))
    return oCPx, oCPy


def getCubicBSpline2DDeformation(iImageSize, iCPx, iCPy, iStep):
    dy, dx = iImageSize
    gx, gy = np.meshgrid(np.arange(dx), np.arange(dy))
    gx, gy = np.array(gx, dtype='float64'), np.array(gy, dtype='float64')
    oGx, oGy = np.zeros_like(gx), np.zeros_like(gy)
    for l in (0, 1, 2, 3):
        for m in (0, 1, 2, 3):
            i, j = np.array(np.floor(gx/iStep[0]), dtype='int64'), np.array(np.floor(gy/iStep[1]), dtype='int64')
            u, v = np.array(gx/iStep[0], dtype='float64')-i, np.array(gy/iStep[1], dtype='float64')-j
            oGx += B[l](u)*B[m](v)*iCPx[j+m, i+l]
            oGy += B[l](u)*B[m](v)*iCPy[j+m, i+l]
    return oGx, oGy


def deformImageBSpline2D(iImage, iCPx, iCPy, iStep):
    dy, dx = iImage.shape
    oGx, oGy = getCubicBSpline2DDeformation(iImage.shape, iCPx, iCPy, iStep)
    gx, gy = np.meshgrid(np.arange(dx), np.arange(dy))
    oGx = 2*gx - oGx # inverz preslikave
    oGy = 2*gy - oGy # inverz preslikave
    oImage = interpn((np.arange(dy), np.arange(dx)), iImage.astype('float'), np.dstack((oGy, oGx)), method='linear', bounds_error=False, fill_value=0)
    return oImage




###########################################
### implementacija genetskega algoritma


def generate_cromosomes(num, init_locx, init_locy, rng=2):
    # number of cromosomes, initial location of points, random number range
    # returns array [num_of_cromosomes, initial_loc.shape[0], initial_loc.shape[1]]
    tmp = np.random.random((2, num, init_locx.shape[0], init_locx.shape[1]))*2*rng - rng
    tmp[0, :, :, :] += init_locx
    tmp[1, :, :, :] += init_locy
    return tmp


def select_mating(population, fit, num_parents):
    # inicializacija prazne matrike staršev
    # indeksi najmanjsih elementov kritejrijske funkcije
    min_idxs = abs(fit).argsort(axis=0)[:num_parents]
    # vrnemo 4 najboljse primerke za starse
    return population[:, min_idxs, :, :].reshape(2, num_parents, population.shape[2], population.shape[3]), fit[min_idxs].reshape(num_parents, 1)


def fitness(source_img, floating_img, cromosomes, iStep):
    fit = np.zeros((cromosomes.shape[1], 1), dtype=np.float)
    for i in range(cromosomes.shape[1]):
        tmp_out = deformImageBSpline2D(floating_img, cromosomes[0, i, :, :], cromosomes[1, i, :, :], iStep)
        fit[i] = 1 - np.abs(fun.im_sm(source_img, tmp_out, 'cc', nb=64, nb_ab=16))
    return fit


def mutate(dst_points, rng, fit_f):
    mutation = np.random.random((2, dst_points.shape[1], dst_points.shape[2], dst_points.shape[3]))*2*rng - rng

    for i in range(dst_points.shape[1]):
        mutation[0, i, :, :] *= fit_f[i]
        mutation[1, i, :, :] *= fit_f[i]

    mutation_locations = np.random.randint(0, 2, (2, dst_points.shape[1], dst_points.shape[2], dst_points.shape[3]))
    return dst_points + np.multiply(mutation, mutation_locations)


def crossover(parents):
    children = np.zeros((2, int(parents.shape[1]/2), parents.shape[2], parents.shape[3]), dtype=np.float)
    n_parents, genes_rows, genes_cols = parents.shape[1], parents.shape[2], parents.shape[3]
    parents_ids = np.arange(n_parents)
    np.random.shuffle(parents_ids)
    swap_ids = np.random.randint(0, 1, (parents.shape[1], parents.shape[2], parents.shape[3]))
    for i in range(int(n_parents / 2)):
        children[0, i, :, :] = parents[0, i * 2, :, :] * swap_ids[i * 2, :, :] + (
                    parents[0, i * 2 + 1, :, :] * (1 - swap_ids[i * 2 + 1, :, :]))
        children[1, i, :, :] = parents[1, i * 2, :, :] * swap_ids[i * 2, :, :] + (
                    parents[1, i * 2 + 1, :, :] * (1 - swap_ids[i * 2 + 1, :, :]))
    return children


################################################
# optimizacija
fixed = np.array(im.open('mr1.png'))
moving = np.array(im.open('mr5.png'))

iStep = ((fixed.shape[0]/8), (fixed.shape[1]/8))

oCPx, oCPy = getCubicBSpline2DGrid(fixed.shape, iStep)

plt.figure()
plt.imshow(fixed, cmap='gray')
plt.plot(oCPx, oCPy, marker='o', color='r', linewidth=1)
plt.plot(oCPx.transpose(), oCPy.transpose(), marker='o', color='r',
         linewidth=1)
plt.xlim([-50,250])
plt.ylim([250,-50])
plt.show()

'''
# premakni kontrolne točke glede na funkcijo vrtinca
oCPx_swirl, oCPy_swirl = swirlControlPoints(oCPx, oCPy, a=2.0, b=30.0)

# izračunaj polje deformacij
oGx, oGy = getCubicBSpline2DDeformation(
    fixed.shape, oCPx_swirl, oCPy_swirl, iStep)

# prevzorči sliko glede na novo lego kontrolnih točk
cbImageDeformed = deformImageBSpline2D(
    fixed, oCPx_swirl, oCPy_swirl, iStep)
'''

count = 0
fmin = 10
f_hist = []
cromosomes = generate_cromosomes(40, oCPx, oCPy, rng=2)
fit = fitness(fixed, moving, cromosomes, iStep)
while count < 3000 and fmin > 0.05:
    selected_cromosomes, selected_fit = select_mating(cromosomes, fit, 20)
    offspring = crossover(selected_cromosomes)

    cromosomes = mutate(cromosomes, 2, fit)

    new_gen = np.append(cromosomes, offspring, axis=1)
    new_gen = np.append(new_gen, selected_cromosomes, axis=1)

    fit = fitness(fixed, moving, new_gen, iStep)
    cromosomes, fit = select_mating(new_gen, fit, num_parents=20)

    count +=1
    fmin = fit.min()
    f_hist.append(fmin)
    idx = np.where(fit[:, 0] == fmin)[0][0]
    T_optx = cromosomes[0, idx, :, :]
    T_opty = cromosomes[1, idx, :, :]
    print("Count: ", count, " fmin: ", fmin)

cbImageDeformed = deformImageBSpline2D(moving, T_optx, T_opty, iStep)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(fixed, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(moving, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(cbImageDeformed, cmap='gray')
plt.show()

plt.figure()
plt.plot(f_hist)
plt.show()

T_optx.tofile('test.dat')
T_optx.tofile('test2.dat')
f_hist = np.array(f_hist, dtype=np.float)
f_hist.tofile('fhist.dat')