"""
Kandidaatintutkielma: Sara Nurminen
----- DOA-estimointisimulaatio -----

KOE 1: RMSE vs SNR

Tässä osassa vertaillaan signaali-kohinasuhteen vaikutusta 
DOA-estimoinnin resoluutioon neljällä eri sensorigeometrialla,
käyttäen MUSIC-algoritmia. Käytetyt geometriat:
-ULA (Uniform Linear Array)
-NLA (Nested Linear Array)
-Restricted MRLA (Minimum Redundancy linear Array)
-General MRLA

"""



import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Perusparametrit
# ============================================================

d = 0.5                                              # sensorien väli λ/2
grid_deg = np.arange(-90, 90, 0.05)                  # keilaus -90 - 90 astetta, 0.05 asteen välein
true_thetas_deg = np.array([-10.001, 10.001])        # todelliset signaalien tulokulmat
snr_list_dB = np.arange(-5, 15, 0.5)                 # läpi käytävät signaali-kohinasuhteen arvot

T = 200                                              # aikanäytteiden lukumäärä
montecarlo = 100                                     # Monte Carlo -kierrosten määrä

M_list = [7]                                          # sensorien lukumäärä M

np.random.seed(0)             


# ============================================================
#  Signaalimalli ja MUSIC
# ============================================================


def deg2rad(deg):
    return deg * np.pi / 180.0



def steering_matrix(thetas_deg, positions):
    """
    Muodostaa ohjausmatriisin antennien paikkojen ja 
    tulevien signaalien perusteella.

    :param thetas_deg: lista tulokulmista asteina
    :param positions: lista antennien paikoista

    :return ohjausmatriisi A (dimensio M x K)
    """
    thetas_rad = deg2rad(np.asarray(thetas_deg))                    # muunnetaan tulokulmat radiaaneiksi ja listasta vektoriksi (1 x M)
    positions = np.asarray(positions).reshape(-1, 1)                               # muunnetaan antennien paikat listasta vektoriksi (M x 1)
    phase = -2j * np.pi * positions * np.sin(thetas_rad)              # M X K
    return np.exp(phase)


def generate_snapshots(positions, thetas_deg, T, snr_dB):
    """
    Muodostaa signaalit X = A S + N siten, että SNR on sama kaikilla geometrioilla.
    SNR määritellään PER SENSORI:
        SNR = (signaalin tehon odotusarvo per sensori) / (kohinateho per sensori)

    :param positions: sensorien paikat
    :param thetas_deg: todelliset DOA:t
    :param T: aikanäytteiden määrä
    :param snr_dB: SNR-arvot

    :return: matriisi X
    """

    positions = np.asarray(positions)
    M = len(positions)
    K = len(thetas_deg)

    # Steering-matriisi (M x K)
    A = steering_matrix(thetas_deg, positions)

    # Satunnaiset signaalit (K x T)
    S = (np.random.randn(K, T) + 1j*np.random.randn(K, T)) / np.sqrt(2)

    # Signaalimatriisi sensoreilla (M x T)
    Y = A @ S

    # Signaaliteho per sensori (keskiarvo |Y|^2)
    signal_power = np.mean(np.abs(Y)**2)

    # SNR per sensori
    snr_lin = 10**(snr_dB / 10.0)

    # Kohinateho, jotta SNR = signal_power / noise_power
    noise_power = signal_power / snr_lin

    # Kohina (M x T)
    N = np.sqrt(noise_power/2) * (
        np.random.randn(M, T) + 1j*np.random.randn(M, T)
    )

    # Havaittu X
    X = Y + N

    return X




def covariance_matrix(X):
    """
    Muodostaa näytekovarianssimatriisin havaittujen signaalien
    X, sekä aikanäytteiden T perusteella

    :param X:   sensorien havaitsemat signaalit

    :return:    kovarianssimatriisi M X M
    """
    return (X @ X.conj().T) / X.shape[1]                    # matriisikertolasku: X x X^H (hermittinen transpoosi)  
                                                            #(M×T × T×M = M×M)   



def music_spectrum(R, positions, grid_deg, K):
    """
    MUSIC-spektri:

        P(theta) = 1 / ( a(theta)^H  U_n U_n^H  a(theta) )

    missä
        R      = kovarianssimatriisi
        U_n    = kohina-avaruuden ominaisvektorit
        a(theta) = ohjausvektori
        grid   = kulmat, joita skannataan
        K      = signaalien lukumäärä
    """

    eigenvals, eigenvecs = np.linalg.eigh(R)            # eigenvals = ominaisarvot (1xM), eigenvecs = matriisi, jonka sarakkeissa ominaisvektorit (MxM)
    order = np.argsort(eigenvals)                       # kertoo ominaisarvojen suuruusjärjestyksen

    # Kohina-avaruus U_n 
    # M antennia → M ominaisvektoria
    # Signaalitila = suurimmat K ominaisarvot
    # Kohinatila   = pienimmät M-K ominaisarvot
    M = len(positions)
    Un = eigenvecs[:, order[:M-K]]                      # kohina-alitila U_n, otetaan M-K pienintä ominaisarvoa  (M × (M-K))

    Agrid = steering_matrix(grid_deg, positions)        # ohjausmatriisi kaikille gridin kulmille

    # Projektion normi kohina-avaruuteen
    UnH_A = Un.conj().T @ Agrid                         # koko (M-K) × 721 (gridin kulmien määrä)

    norm = np.sum(np.abs(UnH_A)**2, axis=0)             # otetaan euklidinen normi UnH_A kaikille theta

    #  P(theta) = 1 / ||U_n^H a(theta)||^2
    P = 1.0 / (norm)        

    return P



def estimate_doa(P, grid_deg, K):
    from scipy.signal import find_peaks
    step = grid_deg[1] - grid_deg[0]
    min_distance_points = int(1.0 / step)                   # 1 astetta piikkien väli (8 pistettä koska yksi askel 0.25)

    peaks, _ = find_peaks(P, distance=min_distance_points)  #scipy find_peaks etsii music-spektrin piikit
    if len(peaks) < K:
        idx = np.argsort(P)[::-1][:K]                       # jos MUSIC-spektro ei löytänyt riittävästi piikkejä, otetaam K suurinta koko spektristä
        return np.sort(grid_deg[idx])
    top = peaks[np.argsort(P[peaks])[::-1][:K]]             # katsotaan piikkien korkeudet ja lajitellaan suurimmasta pienimpään, näistä K suurinta
    return np.sort(grid_deg[top])                           # palautetaan 

def rmse_deg(est, true):
    """
    Laskee RMSEt arvioitujen ja todellisten tulokulmien perusteella

    :param est: estimoitu tulokulma
    :param true: todellinen tulokulma

    :return: RMSE
    """
    from itertools import permutations
    est = np.asarray(est)
    true = np.asarray(true)
    best = 999.0                                # laitetaan ensin jokin suuri luku
    for p in permutations(est):                 # ei tiedetä mikä piikki on mikin signaali, käydään permutaatiot läpi 
        p = np.array(p)                         
        err = np.sqrt(np.mean((p - true)**2))
        best = min(best, err)
    return best

# ============================================================
#  Antennigeometriat
# ============================================================

def ULA(M):
    """
    Muodostaa ULA-geometrian

    :param M: sensorien määrä

    :return: array sensorien paikoista
    """
    return np.arange(M) * d


def NLA(M):
    """
    Kaksitasoinen nested array (Pal & Vaidyanathan):

      - Sisempi ULA: N1 sensoria välein d  →  0, d, ..., (N1-1)*d
      - Ulompi ULA:  N2 sensoria välein (N1+1)*d  →  (N1+1)*d, 2(N1+1)*d, ...

    Yhteensä M = N1 + N2 anturia

    :param M: sensorien määrä

    :return: array sensorien paikoista
    """
    # jaetaan anturit kahteen tasoon
    N1 = (M + 1) // 2      # pyöristys ylöspäin
    N2 = M - N1

    inner = np.arange(N1) * d
    outer = (N1 ) * np.arange(1, N2 + 1) * d

    return np.concatenate([inner, outer])



#  MRLA-tiedostojen luku  (from chat.gpt)

def load_mrla_distances(filename):
    """
    Lukee tiedoston, jossa rivit ovat muotoa:
      M d1 d2 d3 ...
    ja palauttaa sanakirjan: M -> [d1, d2, ...].

    :param filename: tiedoston nimi
    
    :return: dict
    """
    data = {}
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # ohita tyhjät rivit ja #-kommentit
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                M = int(parts[0])
                distances = [float(x) for x in parts[1:]]
                data[M] = distances
    except FileNotFoundError:
        print(f"Varoitus: tiedostoa '{filename}' ei löytynyt.")
    return data

# Ladataan restricted ja general MRLA -data dicteihin
MRLA_restricted_dist = load_mrla_distances("mrla_restricted.txt")
MRLA_general_dist    = load_mrla_distances("mrla_general.txt")

def distances_to_positions(distances):
    """
    Muuntaa välimatkat [d1, d2, ...] antennien paikoiksi [0, d1, d1+d2, ...].

    :param distances: lista antennien välimatkoista

    :return: array antennien sijainneista *d
    """
    positions = [0.0]
    sum = 0.0
    for step in distances:
        sum += step
        positions.append(sum)
    return np.array(positions, dtype=float) * d


def get_mrla_positions(M, distances_dict):
    """
    Palauttaa antennien paikat annetulla M:llä
    (restricted tai general). Jos M:ää ei löydy, palauttaa None.

    :param M: sensorien määrä
    :param distances_dict: M arvoja vasstaavat MRLA:t

    :return: sensorien paikat
    """
    distances = distances_dict.get(M)
    if distances is None:
        return None
    return distances_to_positions(distances)

def difference_coarray(positions):
    """
    Laskee erotusryhmän

    :param positions: sensorien paikat

    :return: erotusryhmä (coarray)
    """
    pos = np.asarray(positions) / d          # skaalataan d-yksiköihin
    diffs = []
    for i in range(len(pos)):
        for j in range(len(pos)):
            diffs.append(pos[i] - pos[j])
    diffs = np.sort(np.unique(np.round(diffs))).astype(int)
    return diffs


# ============================================================
#  RMSE vs SNR 
# ============================================================


def rmse_vs_snr(positions, snrs):
    """
    Käydään läpi kaikki SNR-arvot ja jokaisen arvon kohdalla
    Monte Carlo -koe jonka jokaiselta kierrokselta DOA-estimointi 
    ja lasketaan RMSE-arvo

    :param positions: sensorien paikat
    :param snrs: läpi käytävät SNR-arvot

    :return: array RMSE-arvoista
    """
    rmse_vals = []
    for snr in snrs:
        errs = []
        for i in range(montecarlo):
            X = generate_snapshots(positions, true_thetas_deg, T, snr)
            R = covariance_matrix(X)
            P = music_spectrum(R, positions, grid_deg, len(true_thetas_deg))
            est = estimate_doa(P, grid_deg, len(true_thetas_deg))
            errs.append(rmse_deg(est, true_thetas_deg))
        rmse_vals.append(np.mean(errs))
    return np.array(rmse_vals)


# ----------------------------------------------------
# KUVAAJAT
# ----------------------------------------------------

def plot_rmse_vs_snr(M, rmse_ula, rmse_nested, rmse_r, rmse_g):
    """
    Piirtää kuvan, jossa vertaillaan jokaisen antennigeometrian
    RMSE-arvoja signaali-kohinasuhdetta vastaan
    """
    plt.figure(figsize=(7, 5))
    plt.plot(snr_list_dB, rmse_ula,    marker="o", color="cyan",        label="ULA")
    plt.plot(snr_list_dB, rmse_nested, marker="o", color="mediumblue",  label="NLA")

    if rmse_r is not None:
        plt.plot(snr_list_dB, rmse_r, marker="o", color="mediumorchid", label="MRLA-restricted")
    if rmse_g is not None:
        plt.plot(snr_list_dB, rmse_g, marker="o", color="hotpink",      label="MRLA-general")


    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE (deg, log-scale)")

    plt.title(f"RMSE vs SNR (M = {M}),     θ = [-10, 10]") # \ntrue DOA = {list(true_thetas_deg)}")

    # LOG-AKSELI
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_antenna_geometry(M, pos_ula, pos_nested, pos_mrla_r, pos_mrla_g):
    """
    Piirtää antennien paikat
    """

    colors = {
        "ULA": "cyan",
        "NLA": "mediumblue",
        "MRLA-restricted": "mediumorchid",
        "MRLA-general": "hotpink"
    }

    plt.figure(figsize=(8, 3))
    geoms = []
    geoms.append(("ULA", pos_ula))
    geoms.append(("NLA", pos_nested))
    if pos_mrla_r is not None:
        geoms.append(("MRLA-restricted", pos_mrla_r))
    if pos_mrla_g is not None:
        geoms.append(("MRLA-general", pos_mrla_g))

    y_positions = np.arange(len(geoms))

    for i, (name, pos) in enumerate(geoms):
        x = pos / d
        y = np.full_like(x, i, float)
        plt.scatter(x, y, s=30, color=colors[name], label=name)

    plt.yticks(y_positions, [name for name, _ in geoms])
    plt.xlabel("Sijainti (· d)")
    plt.title(f"Antennien sijainnit (M = {M})")
    plt.grid(True, axis='x', alpha=0.3)

    max_x = max(np.max(pos) / d for _, pos in geoms)
    if max_x <= 20:
        step = 1
    elif max_x <= 50:
        step = 5
    else:
        step = 10
    plt.xticks(np.arange(0, max_x + 0.001, step))

    plt.tight_layout()
    plt.show()



def plot_coarray(M, pos_ula, pos_nested, pos_mrla_r, pos_mrla_g):
    """
    piirtää erotusryhmän
    """
    colors = {
        "ULA": "cyan",
        "NLA": "mediumblue",
        "MRLA-restricted": "mediumorchid",
        "MRLA-general": "hotpink"
    }

    plt.figure(figsize=(8, 3))

    geoms = []
    geoms.append(("ULA", pos_ula))
    geoms.append(("NLA", pos_nested))
    if pos_mrla_r is not None:
        geoms.append(("MRLA-restricted", pos_mrla_r))
    if pos_mrla_g is not None:
        geoms.append(("MRLA-general", pos_mrla_g))

    y_positions = np.arange(len(geoms))

    all_lags = []

    for i, (name, pos) in enumerate(geoms):
        lags = difference_coarray(pos)
        all_lags.append(lags)

        x = lags
        y = np.full_like(x, i, float)
        plt.scatter(x, y, s=25, color=colors[name], label=name)

    plt.yticks(y_positions, [name for name, _ in geoms])
    plt.xlabel("Erotus (· d)")
    plt.title(f"Erotusryhmä (M = {M})")
    plt.grid(True, axis='x', alpha=0.3)

    all_lags = np.concatenate(all_lags)
    max_lag = int(np.max(np.abs(all_lags)))

    step = 5 if max_lag <= 20 else 10

    n_steps = max_lag // step
    xticks = np.arange(-n_steps*step, (n_steps+1)*step, step)
    if 0 not in xticks:
        xticks = np.sort(np.append(xticks, 0))

    plt.xticks(xticks)

    plt.tight_layout()
    plt.show()


    
# ============================================================
#  Pääohjelma 
# ============================================================

def main():
    # käydään ensin läpi eri antennien määrät
    for M in M_list:
        print(f"\n=== M = {M} ===")
        
        # lasketaan ULA ja NLA geometriat
        pos_ula = ULA(M)
        pos_nested = NLA(M)


        # haetaan tiedostosta tuodusta dictistä MRLA geometriat
        pos_mrla_r = get_mrla_positions(M, MRLA_restricted_dist)
        pos_mrla_g = get_mrla_positions(M, MRLA_general_dist)

        # piirretään kuvaaja antennien paikoista
        plot_antenna_geometry(M, pos_ula, pos_nested, pos_mrla_r, pos_mrla_g)
        # piirretään kuvaaja erotusryhmistä (difference coarray)
        plot_coarray(M, pos_ula, pos_nested, pos_mrla_r, pos_mrla_g)


        # RMSE laskeminen 
        rmse_ula    = rmse_vs_snr(pos_ula,    snr_list_dB)
        rmse_nested = rmse_vs_snr(pos_nested, snr_list_dB)

        # MRLA:t vain jos M löytyy tiedostosta
        rmse_r = rmse_g = None
        if pos_mrla_r is not None:
            rmse_r = rmse_vs_snr(pos_mrla_r, snr_list_dB)
        else:
            print(f"  Ei restricted MRLA:a M={M}:lle.")

        if pos_mrla_g is not None:
            rmse_g = rmse_vs_snr(pos_mrla_g, snr_list_dB)
        else:
            print(f"  Ei general MRLA:a M={M}:lle.")

        plot_rmse_vs_snr(M, rmse_ula, rmse_nested, rmse_r, rmse_g)


if __name__ == "__main__":
    main()
