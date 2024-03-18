import numpy as np
import matplotlib.pyplot as plt
import bisect

import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import matplotlib as mpl
mpl.rc('font',**{'sans-serif':['Helvetica'], 'size':20,'weight':'bold'})
mpl.rc('axes',**{'labelweight':'bold', 'linewidth':1.5})
mpl.rc('ytick',**{'major.pad':22, 'color':'k', 'direction':'in', 'right':'True'})
mpl.rc('xtick',**{'major.pad':10,})
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

if SystemName == "HD209458":
    StellarDict = {}
    StellarDict['Mass']= 1.119 # changed for HD209458
    StellarDict['MassErr']=0.033
    StellarDict['Radius']= 1.155
    StellarDict['RadiusErr']=0.016
    
    PlanetaryDict = {}
    PlanetaryDict['Mass']= 1.0*317.907#0.682*317.907
    PlanetaryDict['MassErr']= 0.015*317.907
    PlanetaryDict['Radius']= 1.0*11.2089#1.359*11.2089
    PlanetaryDict['RadiusErr']= 0.019*11.2089

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


    PlanetaryDict['P0'] = 10.0#10**-0.3871
    PlanetaryDict['T0'] = 500.0
    #PlanetaryDict['MR_CH4'] = 10**-10.25
    #PlanetaryDict['MR_CO'] = 10**-4.9367
    PlanetaryDict['MR_H2O'] = 10**-1.3087
    #PlanetaryDict['MR_H2O'] = 10**-4.0
    PlanetaryDict['MR_CO2'] = 10**-5.059
    #PlanetaryDict['MR_CO2'] = 10**-5.0
    #PlanetaryDict['MR_H2S'] = 10**-3.398
    #PlanetaryDict['MR_HCN'] = 10**-6.918
    #PlanetaryDict['MR_NH3'] = 10**-4.337
    #PlanetaryDict['MR_SO2'] = 10**-8.330
    PlanetaryDict['PT']=1
else:
    assert 1 == 2, "Target parameters needs to be manually set."


Location = "/media/prajwal/LaCie/CrossSectionFromSuperCloud4JWST/CrossSectionTIERRA"




NBins = 1500
#MoleculeNames = ["CH4", "CO", "CO2", "H2O", "H2S", "HCN", "NH3", "SO2", "H2"]
MoleculeNames = ["H2O", "CO2", "H2"]


#colorList = ["#0d0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921"]
colorList = ["#0d0887", "#cc4778", "#f0f921"]




HeH2Ratio1 = 0.157
PlanetaryDict["HeH2Ratio"] = HeH2Ratio1

CurrentTarget = System(MoleculeNames, PlanetaryDict, StellarDict)
CurrentTarget.InitiateSystem(PlanetaryDict)
CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)


print("Before")
print(CurrentTarget.nz0)
print(CurrentTarget.nz[:,:5].T)
print("Hydrogen Ama:", CurrentTarget.nz_H2_ama[:5])
print("We will wait and check...@115  : ")

CurrentTarget.nz = CurrentTarget.nz*1000.0
CurrentTarget.nz0 = CurrentTarget.nz0*1000.0
CurrentTarget.nz_H2_ama = CurrentTarget.nz_H2_ama*1000.0

print(CurrentTarget.H2Present)
input("Wait here...@122")


print("After")
print(CurrentTarget.nz0)
print(CurrentTarget.nz[:,:5].T)
print("Hydrogen Ama:", CurrentTarget.nz_H2_ama[:5])
print("We will wait and check...@126  : ")



print("The mass is:", CurrentTarget.Mp)
print("The radius is:", CurrentTarget.Rp)
print("The gravity is:", CurrentTarget.Gp)
print("The mean molecular weight is:", CurrentTarget.mu)
print("The scale height is:", CurrentTarget.H0)
print("Just before loading the cross-section@135:")
CurrentTarget.LoadCrossSection(Location, CIA_Flag=True)



#Create the model
Model = TransmissionSpectroscopy(CurrentTarget, CIA=True)
Model.CalculateTransmission(CurrentTarget)
TS1 = Model.Spectrum
print("First case Mean Molecular Weight:", CurrentTarget.mu)

#Plot here the total spectrum
Ref_nz = CurrentTarget.nz


plt.figure()
plt.plot(CurrentTarget.PzAnalytical, CurrentTarget.nz.T)
plt.xlabel("Pressure [atm]")
plt.ylabel("Number Density [cm$^{-3}$]")
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')
plt.show()

#Input("Wait here...")

fig = plt.figure(figsize=(12,7))
gs = GridSpec(1, 1, figure=fig)

# Plot data on the first axis (top)
ax1 = fig.add_subplot(gs[0, 0])


NumMolecules = len(CurrentTarget.MoleculeNames)
print("The number of molecules is:", NumMolecules)
#Plot the contribution for each of the molecule
for i in range(NumMolecules+1):
    print("\n\n")
    print("Iteration for generating model:", i)
    TempRefNz = np.copy(Ref_nz)
    NotRow = np.arange(len(CurrentTarget.nz0))!=i
    TempRefNz[NotRow,:]*=0.0

    if i<NumMolecules-1:
        print("Case for molecule:", CurrentTarget.MoleculeNames[i])
        CurrentTarget.nz = TempRefNz
        if not(CurrentTarget.MoleculeNames=="H2"):
            print("Not H2 is it?")
        T1 = TransmissionSpectroscopy(CurrentTarget, CIA=True)
        LabelText = CurrentTarget.MoleculeNames[i]
        CurrentColor = colorList[i]
        AlphaValue = 0.85
        LW = 2
        ZORDER = 5
    else:
        print("Considereing all the molecules together.")
        CurrentTarget.nz = Ref_nz
        T1 = TransmissionSpectroscopy(CurrentTarget)
        LabelText = "Combined Model"
        CurrentColor = "black"
        AlphaValue = 0.3
        LW = 2.5
        ZORDER = 10

    #print(CurrentTarget.nz0)
    #print(CurrentTarget.nz[:,:5].T)
    #print("Hydrogen Ama:", CurrentTarget.nz_H2_ama[:5])
    #input("We will wait here@204...")

    #CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)
    #print("After recalculating the PT Profile")
    #print(CurrentTarget.nz0)
    #print(CurrentTarget.nz[:,:5].T)
    #print("Hydrogen Ama:", CurrentTarget.nz_H2_ama[:5])
    #input("We will wait here@211...")
    T1.CalculateTransmission(CurrentTarget)
    BinnedWavelength, BinnedModel = BinTheModel(CurrentTarget.WavelengthArray*1e4, T1.Spectrum, NBins=NBins)

   
    #if k==0:
        #label only the first case    
    ax1.plot(BinnedWavelength, BinnedModel, label=LabelText, color=CurrentColor, lw=LW, zorder=ZORDER, alpha=1.0)
    #else:
    #No labels
    #ax1.plot(Wavelength, BinnedModel, color=CurrentColor, lw=LineWidth, zorder=ZORDER, alpha=EstAlphaValue) 




ax1.set_xlim(0.5, 5.3)
#ax1.set_ylim(14400, 15800)

ax1.set_ylabel('Transit Depth [ppm]', fontsize=20)
ax1.set_xlabel('Wavelength ($ \\rm \mu$m)', fontsize=20)
plt.legend(loc="upper right", fontsize=15)
plt.show()










print("Now do it for Helium dominated atmosphere - No CIA")
HeH2Ratio2 = 1.0
PlanetaryDict["HeH2Ratio"] = HeH2Ratio2
CurrentTarget.InitiateSystem(PlanetaryDict)
CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)
Model = TransmissionSpectroscopy(CurrentTarget)
Model.CalculateTransmission(CurrentTarget)
TS2 = Model.Spectrum
print("Second case Mean Molecular Weight:", CurrentTarget.mu)


print("Now do it for Helium dominated atmosphere - With CIA")
CurrentTarget = System(MoleculeNames, PlanetaryDict, StellarDict)
CurrentTarget.LoadCrossSection(Location, CIA_Flag=True)
CurrentTarget.InitiateSystem(PlanetaryDict)
CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)
Model = TransmissionSpectroscopy(CurrentTarget)
Model.CalculateTransmission(CurrentTarget)
TS3 = Model.Spectrum


BinnedWavelength1, BinnedModel1 = BinTheModel(CurrentTarget.WavelengthArray*1e4, TS1, NBins=NBins)
ax1.plot(BinnedWavelength1, BinnedModel1, color=colorList[0], lw=2, label="He/H2 = %s" %HeH2Ratio1)


BinnedWavelength2, BinnedModel2 = BinTheModel(CurrentTarget.WavelengthArray*1e4, TS2, NBins=NBins)
ax1.plot(BinnedWavelength2, BinnedModel2, color=colorList[1], lw=2, label="He/H2 = %s, No CIA" %HeH2Ratio2)



BinnedWavelength3, BinnedModel3 = BinTheModel(CurrentTarget.WavelengthArray*1e4, TS3, NBins=NBins)
ax1.plot(BinnedWavelength2, BinnedModel2, color=colorList[2], lw=2, label="He/H2 = %s, With CIA" %HeH2Ratio2)


ax1.set_xlim(0.5, 5.3)
ax1.set_ylim(14400, 15800)

ax1.set_ylabel('Transit Depth [ppm]', fontsize=20)
ax1.set_xlabel('Wavelength ($ \\rm \mu$m)', fontsize=20)

TransitDepth = (CurrentTarget.Rp/CurrentTarget.Rs)**2*1e6
ax1.legend(loc="upper right", fontsize=15)
plt.tight_layout()



plt.savefig("RayleighScattering.png")
plt.savefig("RayleighScattering.pdf")
plt.close()

#Describe tierra and the limitations of the model. 
#Show impact of the molecules for each of the molecules.