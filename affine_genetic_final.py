import numpy as np
import funkcije as fun
import PIL.Image as im
import matplotlib.pyplot as plt
import iteround
import random

# Nalozimo slike, mr1 predstavlja referencno, mr2 pa plavajoco sliko
mr1 = np.array(im.open('mr1.png'))
mr2 = np.array(im.open('mr2.png'))

# Prikaz slik na grafu
plt.figure()
plt.subplot(1,2,1)
plt.imshow(mr1, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(mr2, cmap='gray')
plt.show()


def normalizeImage(iImage, type = 'whitening'):
    '''
    Funkcija za normalizacijo vhodne slike.
    :param iImage: vhodna slika za normalizacijo
    :param type: tip standardizacije
    :return: standardizirana slika
    '''
    if type == 'whitening':
        oImage = (iImage - np.mean(iImage)) / np.std(iImage)
    elif type == 'range':
        oImage = (iImage - np.min(iImage)) / (np.max(iImage) - np.min(iImage))
    return oImage


def show_im(image):
    '''
    Funkcija za prikaz slike v oknu
    :param image: slika za prikaz
    :return: None
    '''
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()


# gen oblike [Tx, Ty, rot]
def generate_cromosomes(num, rng=3):
    '''
    Funkcija za nakljucno generacijo kromosomov
    :param num: stevilo kromosomov
    :param rng: interval zacetnih vrednosti kromosomov, uniformna distribucija
    :return: matrika kromosomov
    '''
    a = np.random.random((num, 3)) * 2 * rng - rng
    return a


def select_mating(population, fit, num_parents):
    '''
    Funkcija za izbor starsev iz populacije preko najboljsega fitnesa.
    :param population: kromosomi v trenutni populaciji
    :param fit: vektor fitnesov vhodnih kromosomov
    :param num_parents: stevilo kromosomov, ki jih bomo izlocili kot starse
    :return: kromosomi, ki so izbrani kot starsi
    '''
    # indeksi najmanjsih elementov kritejrijske funkcije
    min_idxs = abs(fit).argsort(axis=0)[:num_parents]
    # vrnemo N najboljsih primerkov za starse
    return population[min_idxs].reshape(num_parents, 3), fit[min_idxs].reshape(num_parents, 1)


def fitness(cromosomes):
    '''
    Funkcija za izracun vrednosti fitnesa vhodnih kromosomov
    :param cromosomes: vhodni kromosomi v dani iteraciji
    :return: vektor z vrednostmi fitnesa
    '''
    st_krom, st_param = cromosomes.shape
    val = np.zeros((st_krom, 1), dtype=np.float)
    for i in range(0, st_krom):
        tmpT = fun.transform_affine_2d(trans=(cromosomes[i, 0], cromosomes[i, 1]), rot=np.deg2rad(cromosomes[i, 2]))
        im_t = fun.im_transform_2d(mr2, tmpT)
        val[i, 0] = 1 - abs(fun.im_sm(mr1, im_t, 'cc', nb=64, nb_ab=16))
    return val


# Ruletno kolo kot altnernativa N najboljsih po fitnesu, se ne uporablja, vseeno pa sem pustil funckijo
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
    '''
    Funkcija za operacijo krizanja kromosomov. Iz nabora kromosomov nakljucno izbere 2, nato jima izmenja gene,
    kjer je lokacija genov za vsak par kromosomov nakljucna.
    :param parents: kromosomi starsev
    :return: krizani kromosomi starsev -> potomci
    '''
    # inicializacija matrike potomcev
    children = np.zeros((int(parents.shape[0] / 2), 3), dtype=np.float)
    # oblika matrike kromosomov starsev
    n_parents, n_genes = parents.shape
    # vektor indeksov starsev
    parents_ids = np.arange(n_parents)
    # nakljucni vrstni red matrike indeksov starsev
    np.random.shuffle(parents_ids)
    for i in range(int(n_parents / 2)):
        # vektor nakljucnih indeksov kromosomov, ki jih bomo zamenjali. Imamo 3 parametre, torej je tudi ta vektor
        # dolzine 3, kjer 1 predstavlja lokacijo za imenjavo, 0 pa ne izmenja gena na dani lokaciji.
        # nakljucno izbranima starsema izmenjamo gene in rezultat shranimo v matriko potomcev
        n_swap = np.random.randint(1, 3, 1)[0]
        idx_swap = np.array(random.sample(range(n_genes), n_swap))
        children[i, :] = parents[i*2, :]
        children[i, idx_swap] = parents[i*2 + 1, idx_swap]
    return children


def mutate(population, max_mut, fit):
    '''
    Funkcija za mutacijo danih kromosomov, kjer ima vsak gen v kromosomu verjetnost mutacije 1/2, interval mutacije pa
    je definiran preko vhodnega parametra max_mut, ki se z vecanjem stevila iteracij ter v primeru enakih rezultatov
    manjsa za fino poravnavo.
    :param population: dana populacija kromosomov v trenutni iteraciji
    :param max_mut: absolutna vrednost maksimalne mutacije gena
    :param fit: fitnes funkcija vhodnih kromosomov
    :return: mutirana populacija
    '''
    # matrika nakljucnih lokacij mutacije za celotno populacijo
    mutate_binary = np.random.randint(0, 2, population.shape)
    mutate_range = np.ones(population.shape)
    # malce zastarela implementacija, ki sicer omogoca neposredni vpliv fitnesa izbranega kromosoma na mutacijo
    # koncna implementacija ne spreminja mutacije glede na fitnes kromosoma.
    for i in range(population.shape[0]):
        mutate_range[i, :] = mutate_range[i, :] * max_mut#* fit[i, 0]
    mutation = np.multiply(mutate_binary, mutate_range)
    mutation = np.multiply(mutation, np.random.randn(mutation.shape[0], mutation.shape[1]) )
    return population + mutation


def optimize():
    '''
    Glavna funkcija za optimizacijo poravnave referencne ter plavajoce slike.
    :return: matrika optimalnih parametrov, vektor minimalnih fitnesov v vsaki iteraciji za namene graficnega prikaza
    '''
    # absolutna vrednost maksimalne mutacije genov
    max_mut = 5
    # spremenljivka, ki spremlja kdaj spremenimo max_mut, da se ne zgodi, da bi jo spremenili neposredno v dveh iteracijah
    flag_count = 0
    # stevec iteracij
    count = 0
    # izmisljena zacetna vrednost kriterijske funkcije, ce zelimo optimizirati dokler ni vrednost k.f. < zeljena k.f.
    fmin = 10
    # vektor minimalnih vrednosti kriterijske funkcije v vsakem koraku
    f_hist = []
    # inicializacija kromosomov
    population = generate_cromosomes(40)
    fit = fitness(population)
    while count < 150:# and fmin > 0.1:
        # genetske operacije na populaciji kromosomov
        parents, tmp = select_mating(population, fit, num_parents=20)
        offspring = crossover(parents)
        population = mutate(population, 1, fit)
        new_gen = np.append(population, offspring, axis=0)
        new_gen = np.append(new_gen, parents, axis=0)
        fit = fitness(new_gen)
        population, fit = select_mating(new_gen, fit, num_parents=40)
        count += 1
        fmin = fit.min()
        f_hist.append(fmin)
        # index kromosoma z najboljsim fitnesom
        idx = np.where(fit[:, 0] == fmin)[0][0]
        # parametri kromosoma z najboljsim fitnesom
        T_opt = population[idx, :]
        print("Count: ", count, " fmin: ", fmin, " maxmut ", max_mut)
        # spreminjanje abs. vrednosti maksimalne mutacije gena glede na zadnjih 10 meritev.
        if (np.array(f_hist[-10:]).max() - np.array(f_hist[-10:]).min()) < 1 ** -3 and len(f_hist) > 50 and flag_count >= 10:
            max_mut *= 0.9
            flag_count = 0
        flag_count += 1

    return T_opt, f_hist


opt_crom, f_hist = optimize()

# Transformacija plavajoce slike glede na izracunane optimalne parametre
tmpT = fun.transform_affine_2d(trans=(opt_crom[0], opt_crom[1]), rot=np.deg2rad(opt_crom[2]))
im_t = fun.im_transform_2d(mr2, tmpT)

# Graficni prikaz rezultatov
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

