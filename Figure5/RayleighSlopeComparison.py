import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import os

def RayleighScattering(Molecule, Lam):
    '''
    This function calculates the rayleigh scattering for different molecules

    Lam is wavelength in
    '''

    N_0 = 25.47E18
    nu = 1./Lam
    nu_2 = nu*nu
    RS_cst = 24*np.pi**3.*1./(Lam**4*N_0**2)


    if Molecule == "CO2":
        print("Case CO2")
        #From Sneep and Ubachs 2005
        n = 1 + 1.1427E3*((5799.25/((128908.9)**2-nu_2))+     \
                          (120.05/((89223.8)**2-nu_2))+       \
                          (5.3334/((75037.5)**2-nu_2))+       \
                          (4.3244/((67837.7)**2-nu_2)))       \
                          #+(0.1218145/((2418.136)**2-nu_2)))
        F_k = 1.1364 + 25.3E-12*nu_2
        return RS_cst*((n*n-1)/(n*n+2))**2.*F_k

    elif Molecule == "CO":
        #From Sneep and Ubachs 2005
        print("Case CO")
        n = 1 + 1E-8*(22851 + 0.456E14/(71427**2-nu_2))
        rho_p = 0.0048
        F_k = (3+6*rho_p)/(3-4*rho_p)
        return  RS_cst*((n*n-1)/(n*n+2))**2*F_k


    elif Molecule == "H2":
        #From equation 14 from Dalgarno and Williams 1965
        #Converting cm to angstrom
        print("Case H2")
        nu_2 = (Lam*1E8)**(-2)
        sig_H2_R = 8.14E-13*(nu_2**2)*((1+     \
                        (1.572E6*(nu_2))+       \
                        (1.981E12*(nu_2**2))+   \
                        (2.307E18*(nu_2**3))+   \
                        (2.582E24*(nu_2**4))+   \
                        (2.822E30*(nu_2**5))))
        return sig_H2_R

    
    elif Molecule == "He":
        #From Wilmouth 2019
        #Converting cm to angstrom
        print("Case He")
        n = 1 + 1E-8*(2283+1.1802e13/(1.5342e10-nu_2))
        F_k = 1.
        return RS_cst*((n**2-1)/(n**2+2))**2*F_k


    elif Molecule == "CH4":
        #From Sneep and Ubachs 2005\
        print("Case CH4")
        n = 1 + 1E-8*(46662. + 4.02E-6*nu_2)
        F_k = 1
        return RS_cst*((n**2-1)/(n**2+2))**2*F_k

    elif Molecule == "O3":
        print("Case O3")
        print("Not available")

    elif Molecule == "N2":
        #From Sneep and Ubachs 2005
        print("Case N2")
        n = 1 + 1E-8*(6498.2+307.43305E12/(14.4E9-nu_2))
        F_k = 1.034 + 3.17E-12*nu_2
        return RS_cst*((n*n-1)/(n*n+2))**2*F_k

    elif Molecule == "H2O":
        print("Case H2O")
        #From Schiebener 1990
        Lam = Lam*1e4
        n = 1 + 1E-6*(268.036+1.476/(Lam*Lam)+  \
                      0.010803/(Lam*Lam*Lam*Lam))
        F_k = 1.0
        return RS_cst*((n*n-1)/(n*n+2))**2*F_k

    else:
        print("The molecule not found")
        assert 1==2




Lam = np.load("Wavelength.npy")

Rayleigh_H2 = RayleighScattering("H2", Lam)
Rayleigh_He = RayleighScattering("He", Lam)
Rayleigh_N2 = RayleighScattering("N2", Lam)
Rayleigh_H2O = RayleighScattering("H2O", Lam)
Rayleigh_CH4 = RayleighScattering("CH4", Lam)
Rayleigh_CO2 = RayleighScattering("CO2", Lam)
Rayleigh_CO = RayleighScattering("CO", Lam)


colorList = ["#0d0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921"]

plt.figure()
plt.plot(Lam*1e4, Rayleigh_CO2/Rayleigh_N2, color="black", linestyle=":", lw=2.5, label="N$_2$")
plt.plot(Lam*1e4, Rayleigh_H2, color="black", linestyle=":", lw=2.5, label="N$_2$")
#plt.plot(Lam*1e4, Rayleigh_CO2, color="red", linestyle=":", lw=2.5, label="CO$_2$")
plt.xlabel("Wavelength [microns]")
plt.ylabel("Rayleigh Cross Section")
plt.yscale("log")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,8))
plt.plot(Lam*1e4, Rayleigh_H2, color="black", linestyle=":", lw=2.5, label="H$_2$")
plt.plot(Lam*1e4, Rayleigh_He, color=colorList[1], linestyle="--", label="He")
plt.plot(Lam*1e4, Rayleigh_CH4, color=colorList[2], label="CH$_4$")
plt.plot(Lam*1e4, Rayleigh_H2O, color=colorList[3], label="H$_2$O")
plt.plot(Lam*1e4, Rayleigh_CO2, color=colorList[4], label="CO$_2$")
plt.plot(Lam*1e4, Rayleigh_CO, color=colorList[5], label="CO")
plt.plot(Lam*1e4, Rayleigh_N2, color=colorList[6], label="N$_2$")
plt.xlim(min(Lam*1e4), max(Lam*1e4))
plt.yscale("log")
plt.legend(loc=0, fontsize=20)
plt.xlabel("Wavelength [microns]", fontsize=20)
plt.ylabel("Rayleigh Cross Section", fontsize=20)
plt.tight_layout()
plt.savefig("RayleighCrossSection.png")
plt.savefig("RayleighCrossSection.pdf")
plt.show()