"""
Kandidaatintutkielma: Sara Nurminen
----- DOA-estimointisimulaatio -----

KOE 2: kulmaseparaatio

Tässä osassa vertaillaan signaalien separointikykyä neljällä eri antennigeometrialla
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

d = 0.5                                                     # sensorien väli λ/2
grid_deg = np.arange(-90, 90, 0.05)                        # keilaus -90 - 90 astetta, 0.05 asteen välein

snr_list_dB = [5]                                           # läpi käytävät signaali-kohinasuhteen arvot
separations_deg = np.arange(2, 10, 0.5)                    # kahden tulevan signaalin välinen separaatio                                             

T = 200                                                     # aikanäytteiden lukumäärä
montecarlo = 100                                            # Monte Carlo -kierrosten määrä
success_tol_deg = 1.0                                       # sallittu keskivirhe (RMSE) asteina, jotta lasketaan onnistumiseksi


M_list = [7]                                          # sensorien lukumäärät

np.random.seed(0)             

# ============================================================
#  MRLA-tiedostojen luku
# ============================================================

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

# Ladataan restricted ja general MRLA -datan sanakirjoihin
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

    # -------- 5. MUSIC-spektri --------
    #  P(theta) = 1 / ||U_n^H a(theta)||^2
    P = 1.0 / (norm)        

    return P




def estimate_doa(P, grid_deg, K):
    """
    tulokulman estimointi
    """
    from scipy.signal import find_peaks
    step = grid_deg[1] - grid_deg[0]
    min_distance_points = int(1.0 / step)  # 1° väliä piikkien välille

    peaks, _ = find_peaks(P, distance=min_distance_points)
    if len(peaks) < K:
        idx = np.argsort(P)[::-1][:K]
        return np.sort(grid_deg[idx])
    top = peaks[np.argsort(P[peaks])[::-1][:K]]
    return np.sort(grid_deg[top])

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

      - Sisempi ULA: N1 anturia välein d  →  0, d, ..., (N1-1)*d
      - Ulompi ULA:  N2 anturia välein (N1+1)*d  →  (N1+1)*d, 2(N1+1)*d, ...

    Yhteensä M = N1 + N2 anturia
    """
    # jaetaan anturit kahteen tasoon
    N1 = (M + 1) // 2      # pyöristys ylöspäin
    N2 = M - N1

    inner = np.arange(N1) * d
    outer = (N1 ) * np.arange(1, N2 + 1) * d

    return np.concatenate([inner, outer])



# ============================================================
#  Separaatio
# ============================================================

def resolution_vs_separation(positions, snr_dB, geom_name):
    """
    Tässä funktiossa on itse koe

    Silmukassa käydään läpi eri separaatioarvot ja 
    jokaisella separaation arvolla suoritetaan
    Monte Carlo -kokeet. Jokaisella monte carlo -kierroksella
    suoritetaan DOA-estimointi ja lasketaan RMSE.

    Funktio laskee myös DOA-estimoinnin onnistumisprosenttia.
    """

    res_prob = []
    rmse_out = []
    
    for dtheta in separations_deg:
        theta0 = -dtheta/2
        theta1 = dtheta/2
        true_thetas = np.array([theta0, theta1])

        successes = 0
        errs = []

        for i in range(montecarlo):
            X = generate_snapshots(positions, true_thetas, T, snr_dB)
            R = covariance_matrix(X)
            P = music_spectrum(R, positions, grid_deg, len(true_thetas))
            est = estimate_doa(P, grid_deg, len(true_thetas))

            # Piirretään havainnollistava MUSIC-spektri vain kerran
            if (dtheta == separations_deg[0]) and (i == 0):
                plot_music_spectrum(
                    P, grid_deg, true_thetas, est,
                    title_extra=(
                        f"{geom_name} – M={len(positions)}, "
                        f"Δθ={dtheta}°, SNR={snr_dB} dB"
                    )
                )


            # Lasketaan RMSE tämän MC-ajon estimoinnille
            err_val = rmse_deg(est, true_thetas)
            errs.append(err_val)

            # SUCCESS-kriteeri: molemmat piikit riittävän lähellä (RMSE <= success_tol_deg)
            if len(est) == 2 and err_val <= success_tol_deg:
                successes += 1


        res_prob.append(successes/montecarlo)
        rmse_out.append(np.mean(errs))

    return np.array(res_prob), np.array(rmse_out)


# ============================================================
#  Kuvaajat
# ============================================================

def plot_music_spectrum(P, grid_deg, true_thetas, est_thetas, title_extra=""):
    plt.figure()
    P_norm = P / np.max(P)
    plt.plot(grid_deg, 10*np.log10(np.maximum(P_norm, 1e-12)))

    # True DOA -viivat
    for th in true_thetas:
        plt.axvline(th, ls="--", color="k", label="True DOA" if th == true_thetas[0] else None)

    # Estimoidut DOA -viivat
    for th in est_thetas:
        plt.axvline(th, ls=":", color="r", label="Est DOA" if th == est_thetas[0] else None)

    plt.xlabel("Kulma (deg)")
    plt.ylabel("MUSIC-spektri (dB, norm.)")
    plt.title(f"MUSIC-spektri {title_extra}")
    plt.grid(True)
    plt.legend()

    # rajataan DOA-akseli, jotta erot näkyvät paremmin
    plt.xlim(-20, 20)

    plt.tight_layout()
    plt.show()


def plot_rmse_vs_separation(separations_deg,
                            rmse_ula, rmse_nla, rmse_mrla_r, rmse_mrla_g,
                            M, snr_dB):
    plt.figure()
    x = separations_deg

    plt.plot(x, rmse_ula,    marker="o", color="cyan",        label="ULA")


    plt.plot(x, rmse_nla,    marker="o", color="mediumblue",  label="NLA")

    if rmse_mrla_r is not None:
        plt.plot(x, rmse_mrla_r, marker="o", color="mediumorchid", label="MRLA-restricted")

    if rmse_mrla_g is not None:
        plt.plot(x, rmse_mrla_g, marker="o", color="hotpink",      label="MRLA-general")

    plt.xlabel("Kulmaseparaatio Δθ (deg)")
    plt.ylabel("RMSE (deg)")

    plt.yscale("log")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.title(
        f"RMSE vs kulmaseparaatio (M = {M}, SNR = {snr_dB} dB)\n"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_success_vs_separation(separations_deg,
                               res_ula, res_nla, res_mrla_r, res_mrla_g,
                               M, snr_dB):
    """
    Piirtää onnistumisprosentin (success rate) jokaiselle geometorialle
    kulmaseparaation funktiona.
    """
    plt.figure()
    x = separations_deg

    geom_labels = []

    plt.plot(x, res_ula,    marker="o", color="cyan",        label="ULA")
    geom_labels.append("ULA")

    plt.plot(x, res_nla,    marker="o", color="mediumblue",  label="NLA")
    geom_labels.append("NLA")

    if res_mrla_r is not None:
        plt.plot(x, res_mrla_r, marker="o", color="mediumorchid", label="MRLA-restricted")
        geom_labels.append("MRLA-restricted")

    if res_mrla_g is not None:
        plt.plot(x, res_mrla_g, marker="o", color="hotpink",      label="MRLA-general")
        geom_labels.append("MRLA-general")

    plt.xlabel("Kulmaseparaatio Δθ (deg)")
    plt.ylabel("Onnistumisprosentti")

    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    geom_text = ", ".join(geom_labels)

    plt.title(
        f"Success% vs kulmaseparaatio (M = {M}, SNR = {snr_dB} dB)\n"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()



# ============================================================
#  Pääfunktio
# ============================================================

def main():
    for M in M_list:

        print(f"\n=== M = {M} ===")

        pos_ula = ULA(M)
        pos_nested = NLA(M)


        # Restricted & general MRLA paikat (tai None jos ei löydy)
        pos_mrla_r = get_mrla_positions(M, MRLA_restricted_dist)
        pos_mrla_g = get_mrla_positions(M, MRLA_general_dist)

        for snr_dB in snr_list_dB:  # nyt snr_dB saa arvon listasta, esim. 10

            res_ula, rmse_ula = resolution_vs_separation(pos_ula, snr_dB, "ULA")
            res_nla, rmse_nla = resolution_vs_separation(pos_nested, snr_dB, "NLA")

            res_mrla_r = rmse_mrla_r = None
            if pos_mrla_r is not None:
                res_mrla_r, rmse_mrla_r = resolution_vs_separation(
                    pos_mrla_r, snr_dB, "MRLA-restricted"
                )

            res_mrla_g = rmse_mrla_g = None
            if pos_mrla_g is not None:
                res_mrla_g, rmse_mrla_g = resolution_vs_separation(
                    pos_mrla_g, snr_dB, "MRLA-general"
                )



            plot_rmse_vs_separation(
                separations_deg,
                rmse_ula, rmse_nla, rmse_mrla_r, rmse_mrla_g,
                M, snr_dB
            )

            plot_success_vs_separation(
                separations_deg,
                res_ula, res_nla, res_mrla_r, res_mrla_g,
                M, snr_dB
            )





if __name__ == "__main__":
    main()
