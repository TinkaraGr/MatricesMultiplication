import matplotlib.pyplot as plt
import numpy as np

#Podatki
procesi = [1, 2, 4, 8]

#MPI NumPy
t_numpy = [0.134175, 0.080048, 0.054915, 0.082146]
s_numpy = [1.00, 1.68, 2.44, 1.63]
e_numpy = [0, 0.19, 0.21, 0.56]

#MPI + Numba
t_numba = [0.484491, 0.289672, 0.199429, 0.215861]
s_numba = [1.00, 1.67, 2.43, 2.24]
e_numba = [0, 0.20, 0.22, 0.37]

#Serial NumPy
t_serial = 0.131953

#GRAF - Pospešek
plt.figure(figsize=(8, 5))
plt.plot(procesi, procesi, 'k--', linewidth=2, label='Idealni pospešek')
plt.plot(procesi, s_numpy, 'o-', color="#97A6C4", linewidth=2, markersize=8, label='MPI NumPy')
plt.plot(procesi, s_numba, 's-', color='#384860', linewidth=2, markersize=8, label='MPI + Numba')
plt.xlabel('Število procesov', fontsize=12)
plt.ylabel('Pospešek', fontsize=12)
plt.title('Pospešek glede na število procesov', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(procesi)
plt.ylim(0, 9)
plt.tight_layout()
plt.savefig('slike/pospesek.png', dpi=150, bbox_inches='tight')
plt.close()

#GRAF - Karp-Flatt
plt.figure(figsize=(8, 5))
plt.plot(procesi[1:], e_numpy[1:], 'o-', color='#97A6C4', linewidth=2, markersize=8, label='MPI NumPy')
plt.plot(procesi[1:], e_numba[1:], 's-', color='#384860', linewidth=2, markersize=8, label='MPI + Numba')
plt.xlabel('Število procesov', fontsize=12)
plt.ylabel('Karp-Flatt', fontsize=12)
plt.title('Karp-Flatt glede na število procesov', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(procesi[1:])
plt.tight_layout()
plt.savefig('slike/karp_flatt.png', dpi=150, bbox_inches='tight')
plt.close()

#GRAF - Časi izvajanja
plt.figure(figsize=(8, 5))
plt.plot(procesi, t_numpy, 'o-', color='#97A6C4', linewidth=2, markersize=8, label='MPI NumPy')
plt.plot(procesi, t_numba, 's-', color='#384860', linewidth=2, markersize=8, label='MPI + Numba')
plt.axhline(y=t_serial, color='#757575', linestyle=':', linewidth=2, label=f'Serial NumPy ({t_serial:.4f} s)')
plt.xlabel('Število procesov', fontsize=12)
plt.ylabel('Čas izvajanja', fontsize=12)
plt.title('Časi izvajanja', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(procesi)
plt.tight_layout()
plt.savefig('slike/casi.png', dpi=150, bbox_inches='tight')
plt.close()