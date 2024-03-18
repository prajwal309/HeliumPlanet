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

#Use tierra to generate the model
from tierra.Target import System
from tierra.transmission import  TransmissionSpectroscopy


def BinTheModel(Wavelength, Model, NBins=100):
    '''
    This function bins the model to the given bins
    '''
    BinnedWavelength = []
    BinnedModel = []
    
    Bins = np.linspace(min(Wavelength), max(Wavelength), NBins+1)

    for Counter, Bin in enumerate(Bins[:-1]):
        Index = np.where((Wavelength > Bins[Counter]) & (Wavelength < Bins[Counter+1]))[0]
        BinnedWavelength.append(np.mean(Wavelength[Index]))
        BinnedModel.append(np.mean(Model[Index]))
    return BinnedWavelength, BinnedModel

SystemName = "HD209458"
#Use the values for HD .


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


if SystemName == "HD209458":
    StellarDict = {}
    StellarDict['Mass']= 1.119 # changed for HD209458
    StellarDict['MassErr']=0.033
    StellarDict['Radius']= 1.155
    StellarDict['RadiusErr']=0.016
    
    PlanetaryDict = {}
    PlanetaryDict['Mass']=0.682*317.907
    PlanetaryDict['MassErr']=0.015*317.907
    PlanetaryDict['Radius']=1.359*11.2089
    PlanetaryDict['RadiusErr']=0.019*11.2089

    '''
    Mass:217.5682878147854
    P0:-0.3871393594636018
    T0:799.6542210018674
    Vplanet:-113.10253774821291
    LOG10HeH2Ratio:-0.10862813925864773
    LogMR_CH4:-10.252212191890491
    LogMR_CO:-4.936709539845975
    LogMR_CO2:-5.059153124907283
    LogMR_H2O:-1.3087215284304672
    LogMR_H2S:-3.3988931151404054
    LogMR_HCN:-6.918116300184245
    LogMR_NH3:-4.337008172545981
    LogMR_SO2:-8.330588887655617
    '''


    PlanetaryDict['P0'] = 10.#10**-0.3871
    PlanetaryDict['T0'] = 799.654
    #PlanetaryDict['MR_CH4'] = 0.0#10**-10.25
    #PlanetaryDict['MR_CO'] = 0.0#10**-4.9367
    PlanetaryDict['MR_CO2'] = 10**-5.059
    #PlanetaryDict['MR_H2O'] = 10**-1.3087
    PlanetaryDict['MR_H2O'] = 10**-1.3087
    #PlanetaryDict['MR_H2S'] = 0.0#10**-3.398
    #PlanetaryDict['MR_HCN'] = 0.0#10**-6.918
    #PlanetaryDict['MR_NH3'] = 0.0#10**-4.337
    #PlanetaryDict['MR_SO2'] = 0.0#10**-8.330
  
    PlanetaryDict['PT']=1
else:
    assert 1 == 2, "Target parameters needs to be manually set."


Location = "/media/prajwal/LaCie/CrossSectionFromSuperCloud4JWST/CrossSectionTIERRA"


#Calculate for H2
Total_MR = 0
for keys, values in PlanetaryDict.items():
    if 'MR' in keys and not('MR_H2'==keys):
        Total_MR += values
MR_H2 = (1 - Total_MR)/1.157
PlanetaryDict['MR_H2'] = MR_H2


#MoleculeNames = ["CH4","CO","CO2","H2O","H2S","HCN","NH3","SO2","H2"]
MoleculeNames = ["CO2", "H2O", "H2"]


CurrentTarget = System(MoleculeNames, PlanetaryDict, StellarDict)
CurrentTarget.LoadCrossSection(Location, CIA_Flag=True)
CurrentTarget.InitiateSystem(PlanetaryDict)
CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)

Ref_Nz0 = CurrentTarget.nz0
RefNz = CurrentTarget.nz
print("The molecules are:", CurrentTarget.MoleculeNames)
ColumnIndex = CurrentTarget.MoleculeNames.index("H2O")
print("The column index is:", ColumnIndex)


AllPressureCutOff = [-9, -5, -4, -3, -2, -1]

'''plt.figure(figsize=(12,7))
plt.plot( CurrentTarget.TzAnalytical, CurrentTarget.PzAnalyticalLog, label="H2O")
plt.axhline(y=-4, color='r', linestyle='--', label="Cut off Pressure")
plt.ylabel("Log Pressure [bar]")
plt.xlabel("Temperature [K]")
plt.tight_layout()
plt.show()'''





#Use different colors for different type of models.
colorList = ["#0d0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921"]

fig = plt.figure(figsize=(12,7))
gs = GridSpec(1, 1, figure=fig)

# Plot data on the first axis (top)
ax1 = fig.add_subplot(gs[0, 0])


ax1.errorbar(Wavelength, TransitDepth+750, yerr=TransitDepthErr, marker="x", color="red", capsize=3, linestyle="None", label="HD209458b Data")
ax1.errorbar(WavelengthHST, TransitDepthHST+750, yerr=TransitDepthErrHST, marker="d", color="red", capsize=3, linestyle="None", label="HD209458b Data")



for Counter, PressureCutOff in enumerate(AllPressureCutOff):
    
    print("\n\n", Counter, PressureCutOff)

    #Model = transmission(StellarDict, PlanetaryDict, WaterAmount)
    Label = 'Cutoff Log10 Pressure:'+str(PressureCutOff)

    #Set things to zero
    #SelectIndex = bisect.bisect_left(CurrentTarget.PzAnalyticalLog, PressureCutOff)

    SelectIndex = len(CurrentTarget.PzAnalyticalLog) - bisect.bisect_left(CurrentTarget.PzAnalyticalLog[::-1], PressureCutOff)
    print("The index selected is:", SelectIndex)

    #Set the number density
    TempNz = np.copy(RefNz)
    TempNz0 = np.copy(Ref_Nz0)
    TempNz[ColumnIndex,SelectIndex:] = 0.0
    CurrentTarget.nz = TempNz

    

    #Create the model
    Model = TransmissionSpectroscopy(CurrentTarget, CIA=True)
    Model.CalculateTransmission(CurrentTarget)

    BinnedWavelength, BinnedModel = BinTheModel(CurrentTarget.WavelengthArray*1e4, Model.Spectrum, NBins=1500)
    ax1.plot(BinnedWavelength, BinnedModel, color=colorList[Counter], lw=2, label=Label)

    


ax1.set_xlim(0.5, 5.3)
ax1.set_ylim(14610, 16500)
ax1.set_ylabel('Transit Depth [ppm]', fontsize=30)
ax1.set_ylabel('Wavelength [Microns]', fontsize=30)
ax1.legend(loc=1, ncols=2, fontsize=10)
plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig("WaterStepFunction_P0_10.png")
plt.savefig("WaterStepFunction_P0_10.pdf")
plt.close()

#Describe tierra and the limitations of the model. 
#Show impact of the molecules for each of the molecules.