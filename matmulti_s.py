import os
#Onemogočimo večnitno izvajanje numpy.dot(), da dobimo resničen enojedrni čas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time

#Nastavimo velikost matrike
N = 1000

#Ustvarimo A in B matriki
np.random.seed(1)  #za ponovljivost
A = np.random.rand(N, N)
B = np.random.rand(N, N)

stevilo_ponovitev = 10
casi = []

for i in range(stevilo_ponovitev):
    #Merjenje časa izvajanja
    start =  time.perf_counter()

    #Množenje matrik
    C = A.dot(B)

    end = time.perf_counter()
    cas = end - start
    casi.append(cas)

    print(f"{i+1}. ponovitev: {cas:.6f}s")

povprecje = np.mean(casi)
std_odklon = np.std(casi, ddof = 1)

print(f"\n====== REZULTATI SERIAL ======")
print(f"Velikost matrik: {N} × {N}")
print(f"Število ponovitev: {stevilo_ponovitev}")
print(f"Povprečni čas: {povprecje:.6f}s")
print(f"Standardni odklon:  {std_odklon:.6f}")
print(f"Minimalni čas:      {min(casi):.6f}s")
print(f"Maksimalni čas:     {max(casi):.6f}s")