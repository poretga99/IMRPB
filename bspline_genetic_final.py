import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import PIL.Image as im
import funkcije as fun
from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage.measure import compare_ssim as ssim

def filter_image(img, sigma):
    img = np.asarray(img, dtype=np.float)
    out_img = gaussian_filter(img, sigma)
    # standardiziramo na interval [0, 1]
    out_img = (out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img))
    return out_img


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
    return (1 - u) ** 3 / 6


def B1(u):
    return (3 * u ** 3 - 6 * u ** 2 + 4) / 6


def B2(u):
    return (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6


def B3(u):
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


def generate_points(iCPx, iCPy):
    '''
    Function generates inner field of image, crops top left and bottom two right rows/columns, returns array of
    [n_of_pts, 2] where rows consists of [row_i, column_i]
    :param iCPx:
    :param iCPy:
    :return:
    '''
    src = np.dstack([iCPx[1:-2,1:-2].flat, iCPy[1:-2,1:-2].flat])[0]
    src = np.asarray(src, dtype=np.uint8)
    return src


def modify_filtered_vals(vals):
    tmp = np.zeros((vals.shape[0], vals.shape[1]), dtype=np.float)
    for i in range(1, vals.shape[0]-1):
        for j in range(1, vals.shape[1]-1):
            if np.abs(vals[i, j] - np.max(vals[i-1:i+2, j-1:j+2]))>0.5:
                tmp[i, j] = np.average(vals[i-1:i+2, j-1:j+2])
                #tmp[i, j] = np.max(vals[i - 1:i + 2, j - 1:j + 2]) / 2.0
            else:
                tmp[i, j] = vals[i, j]
    return tmp


def get_filtered_vals(cromosomes, im_filtered, filter_pts):
    tmp = np.zeros((cromosomes.shape[2], cromosomes.shape[3]), dtype=np.float)
    tmp_vals = im_filtered[[filter_pts[:, 0], filter_pts[:, 1]]].reshape((cromosomes.shape[2]-3, cromosomes.shape[3]-3), order='F')
    tmp_vals = (tmp_vals - tmp_vals.min())/(tmp_vals.max()-tmp_vals.min())*0.8+0.2
    tmp[1:-2, 1:-2] = im_filtered[[filter_pts[:, 0], filter_pts[:, 1]]].reshape((cromosomes.shape[2]-3, cromosomes.shape[3]-3), order='F')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tmp = modify_filtered_vals(tmp)
    return tmp


def mi(im1, im2, bins):
    hist_2d, e_edges, y_edges = np.histogram2d(im1.ravel(), im2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

###########################################
### implementacija genetskega algoritma


def generate_cromosomes(num, init_locx, init_locy, rng=2):
    # number of cromosomes, initial location of points, random number range
    # returns array [num_of_cromosomes, initial_loc.shape[0], initial_loc.shape[1]]
    tmp = np.random.random((2, num, init_locx.shape[0], init_locx.shape[1]))*2*rng - rng
    tmp[0, :, :, :] += init_locx
    tmp[1, :, :, :] += init_locy
    return tmp


def select_mating(population, fit, num_parents, deformed):
    # inicializacija prazne matrike staršev
    # indeksi najmanjsih elementov kritejrijske funkcije
    min_idxs = abs(fit).argsort(axis=0)[:num_parents]
    # vrnemo x najboljse primerke za starse
    return population[:, min_idxs, :, :].reshape(2, num_parents, population.shape[2], population.shape[3]), fit[min_idxs].reshape(num_parents, 1), deformed[min_idxs.reshape(min_idxs.shape[0]), :, :], min_idxs


def fitness(source_img, floating_img, cromosomes, iStep):
    source_img = np.asarray(source_img, dtype=np.float)
    floating_img = np.asarray(floating_img, dtype=np.float)
    fit = np.zeros((cromosomes.shape[1], 1), dtype=np.float)
    filtered_ims = np.zeros((cromosomes.shape[1], source_img.shape[0], source_img.shape[1]), dtype=np.float)
    deformed_ims = np.zeros((cromosomes.shape[1], source_img.shape[0], source_img.shape[1]), dtype=np.float)
    for i in range(cromosomes.shape[1]):
        tmp_out = deformImageBSpline2D(floating_img, cromosomes[0, i, :, :], cromosomes[1, i, :, :], iStep)
        fit[i] = 1.0 / mi(source_img, tmp_out, 16)
        #fit[i] = 1 - ssim(source_img, tmp_out)
        #fit[i] = 1 - np.abs(fun.im_sm(source_img, tmp_out, 'cc', nb=64, nb_ab=16))
        filtered_ims[i, :, :] = filter_image(np.abs(source_img - tmp_out), 20) #!¨!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        deformed_ims[i, :, :] = tmp_out
    return fit, filtered_ims, deformed_ims


def mutate(dst_points, rng, fit_f, flag, filt_pts, fixed, filtered):
    #rng *= flag
    mutation = np.random.random((2, dst_points.shape[1], dst_points.shape[2], dst_points.shape[3]))*2*rng - rng

    for i in range(dst_points.shape[1]):
        mutation[0, i, :, :] = np.multiply(mutation[0, i, :, :], 2*get_filtered_vals(cromosomes, filtered[i], filt_pts))
        mutation[1, i, :, :] = np.multiply(mutation[0, i, :, :], 2*get_filtered_vals(cromosomes, filtered[i], filt_pts))
        #mutation[0, i, :, :] *= 1#fit_f[i]
        #mutation[1, i, :, :] *= 1#fit_f[i]


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




def normalizeImage(iImage, type='whitening'):
    if type=='whitening':
        oImage = (iImage - np.mean(iImage)) / np.std(iImage)
    elif type=='range':
        oImage = (iImage - np.min(iImage)) / (np.max(iImage) - np.min(iImage))
    return oImage


# optimizacija
###########################################################
# Nalozimo slike

fixed = normalizeImage(np.array(im.open('dojka1.png')))
moving = normalizeImage(np.array(im.open('dojka2.png')))


##############################################################
fixed = np.asarray(fixed, dtype=np.float)
moving = np.asarray(moving, dtype=np.float)
#fixed[fixed > 5.0] = 254.0
#moving[moving > 5.0] = 254.0

#edges1 = feature.canny(fixed, sigma=3)
#edges2 = feature.canny(moving, sigma=3)

#fixed = np.asarray(edges1, dtype=np.float)
#moving = np.asarray(edges2, dtype=np.float)



iStep = (((fixed.shape[0]-1)/5), ((fixed.shape[1]-1)/5))

oCPx, oCPy = getCubicBSpline2DGrid(fixed.shape, iStep)

plt.figure()
plt.imshow(moving, cmap='gray')
plt.plot(oCPx, oCPy, marker='o', color='r', linewidth=1)
plt.plot(oCPx.transpose(), oCPy.transpose(), marker='o', color='r',
         linewidth=1)
plt.xlim([-50,250])
plt.ylim([250,-50])
plt.show()

count = 0
flag = 1
fmin = 10
flag_count = 0
f_hist = np.ones(1, dtype=np.float)
cromosomes = generate_cromosomes(40, oCPx, oCPy, rng=2)
fit, filtered, deformed = fitness(fixed, moving, cromosomes, iStep)
filt_pts = generate_points(oCPx, oCPy) # points in rectangle for filtered image value return
deformed = np.zeros((cromosomes.shape[1], fixed.shape[0], fixed.shape[1]), dtype=np.float)

for i in range(cromosomes.shape[1]):
    deformed[i, :, :] = fixed


ax = plt.subplot(2,3,1)
ax.title.set_text('Referencna')
ax.imshow(fixed, cmap='gray')

ax1 = plt.subplot(2,3,2)
ax1.imshow(moving, cmap='gray')
ax1.title.set_text('Plavajoca')

ax2 = plt.subplot(2,3,3)
ax2.title.set_text('fmin = ')
ax2.imshow(np.abs(fixed - deformed[0, :, :]), cmap='gray')

ax3 = plt.subplot(2,3,4)
ax3.imshow(moving, cmap='gray')
ax3.title.set_text('Deformirana')

ax4 = plt.subplot(2,3,5)
ax4.imshow(deformed[0, :, :], cmap='gray')
ax4.title.set_text('Deformirana + tocke')
l1 = ax4.plot(oCPx, oCPy, marker='o', color='r', linewidth=1)
l2 = ax4.plot(oCPx.transpose(), oCPy.transpose(), marker='o', color='r', linewidth=1)

ax5 = plt.subplot(2,3,6)
ax5.imshow(filter_image(np.abs(fixed - moving), 10), cmap='gray')
ax5.title.set_text('Zglajena')

plt.gcf().canvas.draw()
plt.pause(0.5)

while count < 2000: #and fmin > 1.5:
    selected_cromosomes, selected_fit, deformed, min_idxs = select_mating(cromosomes, fit, 30, deformed)
    offspring = crossover(selected_cromosomes)

    cromosomes = mutate(cromosomes, 2, fit, flag, filt_pts, fixed, filtered)

    new_gen = np.append(cromosomes, offspring, axis=1)
    new_gen = np.append(new_gen, selected_cromosomes, axis=1)
    fit, filtered, deformed = fitness(fixed, moving, new_gen, iStep)
    cromosomes, fit, deformed, min_idxs = select_mating(new_gen, fit, 40, deformed)

    filtered2 = np.array(filtered)
    filtered2 = filtered2[min_idxs.reshape(min_idxs.shape[0]), :, :]

    count +=1
    f_hist = np.append(f_hist, fit.min())
    fmin = fit.min()
    idx = np.where(fit[:, 0] == fmin)[0][0]
    T_optx = cromosomes[0, idx, :, :]
    T_opty = cromosomes[1, idx, :, :]
    def_opt = deformed[idx, :, :]
    filt_opt = filtered2[idx, :, :]
    print("Count: ", count, " fmin: ", fmin, " flag:", flag, " fhist std", f_hist[-10:].var())
    if (f_hist[-10:].max() - f_hist[-10:].min()) < 1**-3 and len(f_hist) > 50 and flag_count >= 10:
        flag *= 1.0
        flag_count = 0
    flag_count += 1

    ax3.images[0].set_data(def_opt)
    ax4.clear()
    ax4.imshow(def_opt, cmap='gray')
    ax4.title.set_text('Deformirana + tocke')
    ax4.plot(T_optx, T_opty, marker='o', color='r', linewidth=1)
    ax4.plot(T_optx.transpose(), T_opty.transpose())
    ax2.clear()
    ax2.imshow(np.abs(def_opt - fixed), cmap='gray')
    ax2.title.set_text('Fmin: {:.3f} @iter. {}'.format(float(fmin), int(count)))
    ax5.images[0].set_data(filt_opt)
    plt.gcf().canvas.draw()
    plt.pause(0.5)





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