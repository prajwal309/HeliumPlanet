import numpy as np
import matplotlib.pyplot as plt
import bisect

import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import matplotlib as mpl
mpl.rc('font',**{'sans-serif':['Helvetica'], 'size':30,'weight':'bold'})
mpl.rc('axes',**{'labelweight':'bold', 'linewidth':1.5})
mpl.rc('ytick',**{'major.pad':30, 'color':'k', 'direction':'in', 'right':'True'})
mpl.rc('xtick',**{'major.pad':30,})
mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
mpl.rc('text', **{'usetex':True})
mpl.rc('contour', **{'negative_linestyle':'solid'})


NIRCAMData = np.loadtxt("HD209458b_NIRCAM.txt")
Wavelength = NIRCAMData[:,0]
WavelengthBin = NIRCAMData[:,1]
TransitDepth = NIRCAMData[:,2]*1e6
TransitDepthErr = NIRCAMData[:,3]*1e6


HSTData = np.loadtxt("HD209458b_NIRCAM_HST.txt", skiprows=1)
WavelengthHST = HSTData[:,0]
WavelengthBinHST = HSTData[:,1]
TransitDepthHST = HSTData[:,2]*1e6
TransitDepthErrHST = HSTData[:,3]*1e6


def GetSO2CrossSection():
    SO2Data = open("CrossSection/SO2_358.0_0.0_23995.0-43985.0_09.xsc", 'r').readlines()
    AllContent = SO2Data[0].split()
    StartWaveNumber = float(AllContent[1])
    StopWaveNumber = float(AllContent[2])
    NumWaveNumber = int(AllContent[3])
    WaveNumber  = np.linspace(StartWaveNumber, StopWaveNumber, NumWaveNumber)
    SO2_CrossSection = []
    for Content in SO2Data[1:]:
        SO2_CrossSection.extend(Content.split())
    SO2_CrossSection = np.array(SO2_CrossSection).astype(float)
    WavelengthSO2 = 10000/WaveNumber
    return WavelengthSO2[::-1], SO2_CrossSection[::-1]


def GetO3CrossSection():
    O3Data = open("CrossSection/O3_293.0K-0.0Torr_28901.0-40999.0_118.xsc", 'r').readlines()
    AllContent = O3Data[0].split()
    StartWaveNumber = float(AllContent[1])
    StopWaveNumber = float(AllContent[2])
    NumWaveNumber = int(AllContent[3])
    WaveNumber  = np.linspace(StartWaveNumber, StopWaveNumber, NumWaveNumber)
    O3_CrossSection = []
    for Content in O3Data[1:]:
        O3_CrossSection.extend(Content.split())
    O3_CrossSection = np.array(O3_CrossSection).astype(float)
    WavelengthO3 = 10000/WaveNumber
    return WavelengthO3[::-1], O3_CrossSection[::-1]




def GetCS2CrossSection():
    CS2Data = open("CrossSection/CS2_323.1K-760.0K_600.0-6500.0_0.11_N2_380_43.xsc", 'r').readlines()
    AllContent = CS2Data[0].split()
    StartWaveNumber = float(AllContent[1])
    StopWaveNumber = float(AllContent[2])
    NumWaveNumber = int(AllContent[3])
    WaveNumber  = np.linspace(StartWaveNumber, StopWaveNumber, NumWaveNumber)
    CS2_CrossSection = []
    for Content in CS2Data[1:]:
        CS2_CrossSection.extend(Content.split())
    CS2_CrossSection = np.array(CS2_CrossSection).astype(float)
    WavelengthCS2 = 10000/WaveNumber
    return WavelengthCS2[::-1], CS2_CrossSection[::-1]


WavelengthO3, O3_CrossSection =  GetO3CrossSection()
WavelengthSO2, SO2_CrossSection =  GetSO2CrossSection()
WavelengthCS2, CS2_CrossSection =  GetCS2CrossSection()
MaxValue = np.max([max(O3_CrossSection), max(SO2_CrossSection)])


MoleculeNames = ["CS2", "S2", "O3", "H2O", "SO2", "H2SO4", "H2S"]



#Use different colors for different type of models.
colorList = ["#0d0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921"]

fig = plt.figure(figsize=(12,6))
gs = GridSpec(1, 1, figure=fig)

# Plot data on the first axis (top)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = ax1.twinx()

#ax1.errorbar(Wavelength, TransitDepth-50, yerr=TransitDepthErr, marker="x", color="black", capsize=3, linestyle="None", label="HD209458b HST Data")
#ax1.errorbar(WavelengthHST, TransitDepthHST-500, yerr=TransitDepthErrHST, marker="d", color="black", capsize=3, linestyle="None", label="HD209458b JWST Data")

#ax2.plot(WavelengthSO2, SO2_CrossSection, label="SO2", color=colorList[0])
#ax2.plot(WavelengthO3, O3_CrossSection, label="O3", color=colorList[2])
ax2.plot(WavelengthCS2, CS2_CrossSection, label="CS2", color=colorList[4])

ax2.legend(loc=0, fontsize=20)

#ax1.set_xlim(0.3, 5.3)
#ax1.set_xlim(0.3, 0.7)
#ax2.set_ylim(1e-21, MaxValue*2)
ax2.set_yscale("log")
#ax1.set_ylim(14400, 15800)
ax1.set_ylabel('Transit Depth [ppm]', fontsize=30)
#ax1.legend(loc=0, ncols=2, fontsize=20)
plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig("Figures/H2SO4_DifferentMolecules.png")
plt.savefig("Figures/H2SO4_DifferentMolecules.pdf")
plt.show()
plt.close()

#Describe tierra and the limitations of the model. 
#Show impact of the molecules for each of the molecules.