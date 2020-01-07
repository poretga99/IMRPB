import numpy as np
import funkcije as fun
import PIL.Image as im
import matplotlib.pyplot as plt
import iteround
import random


mr1 = np.array(im.open('mr1.png'))
mr2 = np.array(im.open('mr2.png'))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(mr1, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(mr2, cmap='gray')
plt.show()


def normalizeImage(iImage, type = 'whitening'):
    if type == 'whitening':
        oImage = (iImage - np.mean(iImage)) / np.std(iImage)
    elif type == 'range':
        oImage = (iImage - np.min(iImage)) / (np.max(iImage) - np.min(iImage))
    return oImage

# gen oblike [Tx, Ty, rot]

def show_im(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()


def generate_cromosomes(num, rng=3):
    a = np.random.random((num, 3)) * 2 * rng - rng
    return a


def select_mating(population, fit, num_parents):
    # inicializacija prazne matrike staršev
    # indeksi najmanjsih elementov kritejrijske funkcije
    min_idxs = abs(fit).argsort(axis=0)[:num_parents]
    # vrnemo 4 najboljse primerke za starse
    return population[min_idxs].reshape(num_parents, 3), fit[min_idxs].reshape(num_parents, 1)


def fitness(cromosomes):
    st_krom, st_param = cromosomes.shape
    val = np.zeros((st_krom, 1), dtype=np.float)
    for i in range(0, st_krom):
        tmpT = fun.transform_affine_2d(trans=(cromosomes[i, 0], cromosomes[i, 1]), rot=np.deg2rad(cromosomes[i, 2]))
        im_t = fun.im_transform_2d(mr2, tmpT)
        val[i, 0] = 1 - abs(fun.im_sm(mr1, im_t, 'cc', nb=64, nb_ab=16))
    return val


def roulette(population, fit, num_parents=4):
    # izracunamo P_i
    fitness_rel = (fit) / np.sum(fit) * 100
    # zaokrozujemo tako, da je vsota se vedno enaka 100!!!
    fitn_rel_round = iteround.saferound(list(fitness_rel[:, 0]), 0)
    # naredimo array s stotimi elementi, glede na P_i priredimo vrednosti
    # Npr. Če je P_i za 1 kromosom enak 40, potem bo imel array 40 elementov z vrednostjo 1... To naredimo za vse trenutne kromosome
    # najboljsi kromosom bo imel najvec vnosov -> najvecja verjetnost, da bo izbran kot starš
    a = []
    for i in range(num_parents):
        a = a + [i] * int(fitn_rel_round[i])
    a = np.array(a)
    # nakljucno premešamo array
    np.random.shuffle(a)
    # izberemo prve 4 unikatne vrednosti
    # podoben princip kot da vrtimo kolo, le da ga ne vrtimo n-krat, ampak samo naključno premešamo vrednosti
    # in poiščemo prve 4 unikatne
    unique = []
    for el in a:
        if not el in unique:
            unique.append(el)

    return population[unique[:num_parents]], fit[unique[:num_parents]]


def crossover(parents):
    children = np.zeros((int(parents.shape[0] / 2), 3), dtype=np.float)
    # koliko genov zamenjamo
    n_parents, n_genes = parents.shape
    parents_ids = np.arange(n_parents)
    # nakljucni vrstni red starsev
    np.random.shuffle(parents_ids)
    for i in range(int(n_parents / 2)):
        #shape, max val
        n_swap = np.random.randint(1, 3, 1)[0]
        # toDo nakljucna zamenjava
        # nakljucne lokacije za zamenjavo IDja
        idx_swap = np.array(random.sample(range(n_genes), n_swap))
        children[i, :] = parents[i*2, :]
        children[i, idx_swap] = parents[i*2 + 1, idx_swap]
    return children


def mutate(population, max_mut, fit):
    mutate_binary = np.random.randint(0, 2, population.shape)
    mutate_range = np.ones(population.shape)
    for i in range(population.shape[0]):
        mutate_range[i, :] = mutate_range[i, :] * max_mut#* fit[i, 0]
    mutation = np.multiply(mutate_binary, mutate_range)
    mutation = np.multiply(mutation, np.random.randn(mutation.shape[0], mutation.shape[1]) )
    return population + mutation



def optimize():
    max_mut = 5
    flag_count = 0
    count = 0
    fmin = 10
    f_hist = []
    population = generate_cromosomes(40)
    fit = fitness(population)
    while count < 150:# and fmin > 0.1:
        parents, tmp = select_mating(population, fit, num_parents=20)
        offspring = crossover(parents)
        # appendamo novo generacijo
        population = mutate(population, 1, fit)
        new_gen = np.append(population, offspring, axis=0)
        new_gen = np.append(new_gen, parents, axis=0)
        fit = fitness(new_gen)
        population, fit = select_mating(new_gen, fit, num_parents=40)
        count += 1
        fmin = fit.min()
        f_hist.append(fmin)
        idx = np.where(fit[:, 0] == fmin)[0][0]
        T_opt = population[idx, :]
        print("Count: ", count, " fmin: ", fmin, " maxmut ", max_mut)

        if (np.array(f_hist[-10:]).max() - np.array(f_hist[-10:]).min()) < 1 ** -3 and len(f_hist) > 50 and flag_count >= 10:
            max_mut *= 1.0
            flag_count = 0
        flag_count += 1

    return T_opt, f_hist


opt_crom, f_hist = optimize()

tmpT = fun.transform_affine_2d(trans=(opt_crom[0], opt_crom[1]), rot=np.deg2rad(opt_crom[2]))
im_t = fun.im_transform_2d(mr2, tmpT)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(mr1, cmap='gray')
plt.title('Referencna')
plt.subplot(1,3,2)
plt.imshow(mr2, cmap='gray')
plt.title('Plavajoca')
plt.subplot(1,3,3)
plt.imshow(im_t, cmap='gray')
plt.title('Poravnana')
plt.show()

plt.figure()
plt.imshow(np.abs(mr1 - im_t), cmap='gray')
plt.show()

plt.figure()
plt.plot(f_hist)
plt.show()


####################################################################################

