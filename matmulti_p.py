import os
#Onemogočimo večnitno izvajanje numpy.dot(), da dobimo resničen enojedrni čas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Nastavimo velikost matrike
    N = 1000
    
    rows_per_proc = N // size

    if rank == 0:
        #Na master procesu ustvarimo matriki A in B
        np.random.seed(1)
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
    else:
        A = None
        B = np.empty((N, N), dtype = 'd') #B se bo napolnila preko Bcast

    #Pripravimo lokalni blok matrike A
    local_A = np.empty((rows_per_proc, N), dtype = 'd')
    local_C = np.empty((rows_per_proc, N), dtype = 'd')

    if rank == 0:
        C = np.empty((N, N), dtype = 'd')
    else:
        C = None

    stevilo_ponovitev = 10
    casi = []

    for i in range(stevilo_ponovitev):
        comm.Barrier()
        start = MPI.Wtime()

        #Scatter: razdelimo vrstice A
        comm.Scatter([A, MPI.DOUBLE], [local_A, MPI.DOUBLE], root = 0)

        #Broadcast: razpošljemo matriko B vsem procesom
        comm.Bcast([B, MPI.DOUBLE], root = 0)

        #Vsak proces izvede množenje svojega bloka vrstic z matriko B
        local_C = local_A.dot(B)
    
        #Gather: zberemo vse bloke nazaj v C na korenu
        comm.Gather([local_C, MPI.DOUBLE], [C, MPI.DOUBLE], root = 0)

        comm.Barrier()
        end = MPI.Wtime()

        elapsed = end - start
        max_cas = comm.reduce(elapsed, op = MPI.MAX, root = 0)

        if rank == 0:
            casi.append(max_cas)
            print(f"{i+1}. ponovitev: {max_cas:.6f}s")

    #Master proces izpiše rezultat
    if rank == 0:
        povprecje = np.mean(casi)
        std_odklon = np.std(casi, ddof = 1)

        print(f"\n==== REZULTATI MPI ({size} procesov) ====")
        print(f"Velikost matrik: {N} × {N}")
        print(f"Število ponovitev: {stevilo_ponovitev}")
        print(f"Povprečni čas:     {povprecje:.6f}s")
        print(f"Standardni odklon: {std_odklon:.6f}")
        print(f"Minimalni čas:     {min(casi):.6f}s")
        print(f"Maksimalni čas:    {max(casi):.6f}s")

if __name__ == "__main__":
    main()