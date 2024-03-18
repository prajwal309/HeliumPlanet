import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left


Location = "/media/prajwal/LaCie/CrossSectionFromSuperCloud4JWST/CrossSectionTIERRA"

Wavelength = np.load(Location + "/Wavelength.npy")
Temperature = np.loadtxt(Location + "/Temperature.txt")
Pressure = np.loadtxt(Location + "/Pressure.txt")

TIndex = bisect_left(Temperature, 520)
PIndex = bisect_left(Pressure, 0.01)

CrossSection = np.load("/media/prajwal/LaCie/CrossSectionFromSuperCloud4JWST/CrossSectionTIERRA/CS_1/H2O.npy")[TIndex, PIndex, :]

plt.figure()
plt.plot(Wavelength*1e4, CrossSection, "k-")
plt.xlim(0.5, 5.3)
plt.yscale("log")
plt.xlabel("Wavelength (microns)")
plt.ylabel("Cross-Section [per cm] ")
plt.tight_layout()
plt.savefig("WaterCrossSection.png")
plt.show()