import os
#Onemogočimo večnitno izvajanje BLAS knjižnic, da dobimo bolj realne MPI rezultate
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from mpi4py import MPI
from numba import njit
import numpy as np

#Numba funkcija za množenje matrik
@njit
def matmulti_numba(A, B):
    vrstice = A.shape[0]
    stolpci = B.shape[1]
    notri = A.shape[1]

    C = np.zeros((vrstice, stolpci))

    for i in range(vrstice):
        for j in range(stolpci):

            s = 0.0

            for k in range(notri):
                s += A[i, k] * B[k, j]

            C[i, j] = s

    return C

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Velikost matrik
    N = 4000

    rows_per_proc = N // size

    #Master ustvari matriki
    if rank == 0:
        np.random.seed(1)
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)

    else:
        A = None
        B = np.empty((N, N), dtype = 'd')

    local_A = np.empty((rows_per_proc, N), dtype = 'd')
    local_C = np.empty((rows_per_proc, N), dtype = 'd')

    if rank == 0:
        C = np.empty((N, N), dtype=np.float64)
    else:
        C = None

    stevilo_ponovitev = 10
    casi = []

    for i in range(stevilo_ponovitev):
        comm.Barrier()
        start = MPI.Wtime()

        # Scatter
        comm.Scatter([A, MPI.DOUBLE], [local_A, MPI.DOUBLE], root = 0)

        # Broadcast
        comm.Bcast([B, MPI.DOUBLE],
                   root=0)

        # Lokalno množenje
        local_C = matmulti_numba(local_A, B)

        # Gather
        comm.Gather([local_C, MPI.DOUBLE], [C, MPI.DOUBLE], root = 0)

        comm.Barrier()
        end = MPI.Wtime()
        elapsed = end - start

        # Vzamemo najpočasnejši proces
        max_time = comm.reduce(elapsed, op = MPI.MAX, root = 0)

        if rank == 0:
            casi.append(max_time)
            print(f"{i+1}. ponovitev: {max_time:.6f}s")

    if rank == 0:
        povprecje = np.mean(casi)
        std_odklon = np.std(casi, ddof=1)

        print(f"\n==== REZULTATI MPI + NUMBA ({size} procesov) ====")
        print(f"Velikost matrik: {N} x {N}")
        print(f"Število ponovitev: {stevilo_ponovitev}")
        print(f"Povprečni čas:     {povprecje:.6f}s")
        print(f"Standardni odklon: {std_odklon:.6f}")
        print(f"Minimalni čas:     {min(casi):.6f}s")
        print(f"Maksimalni čas:    {max(casi):.6f}s")

if __name__ == "__main__":
    main()