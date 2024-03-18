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


    PlanetaryDict['P0'] = 1.0#10**-0.3871
    PlanetaryDict['T0'] = 799.654
    PlanetaryDict['MR_CH4'] = 0.0#10**-10.25
    PlanetaryDict['MR_CO'] = 0.0#10**-4.9367
    PlanetaryDict['MR_CO2'] = 10**-5.059
    #PlanetaryDict['MR_H2O'] = 10**-1.3087
    PlanetaryDict['MR_H2O'] = 10**-1.3087
    PlanetaryDict['MR_H2S'] = 0.0#10**-3.398
    PlanetaryDict['MR_HCN'] = 0.0#10**-6.918
    PlanetaryDict['MR_NH3'] = 0.0#10**-4.337
    PlanetaryDict['MR_SO2'] = 0.0#10**-8.330
    PlanetaryDict['MR_H2'] = 0.80
    PlanetaryDict['PT']=1
else:
    assert 1 == 2, "Target parameters needs to be manually set."


Location = "/media/prajwal/LaCie/CrossSectionFromSuperCloud4JWST/CrossSectionTIERRA"



#MoleculeNames = ["CH4", "CO", "CO2", "H2O", "H2S", "HCN", "NH3", "SO2", "H2"]
MoleculeNames = ["CO2", "H2O", "H2"]


CurrentTarget = System(MoleculeNames, PlanetaryDict, StellarDict)
CurrentTarget.LoadCrossSection(Location, CIA_Flag=True)
CurrentTarget.InitiateSystem(PlanetaryDict)
CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)

DifferentWaterAmount = [-2.00,  -1.50]


#Use different colors for different type of models.
colorList = ["#0d0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921"]

fig = plt.figure(figsize=(12,14))
gs = GridSpec(2, 1, figure=fig)

# Plot data on the first axis (top)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])


for Counter, WaterAmount in enumerate(DifferentWaterAmount):
    
    print("\n\n", Counter, WaterAmount)

    #Model = transmission(StellarDict, PlanetaryDict, WaterAmount)
    Label = 'Log MR$_{\\rm H_2O}$ = '+str(WaterAmount)

    #PlanetaryDict['MR_H2O'] = WaterAmount
    PlanetaryDict['MR_H2O'] = 10**WaterAmount

    Total_MR = 0
    for keys, values in PlanetaryDict.items():
        if 'MR' in keys and not('MR_H2'==keys):
            Total_MR += values
    
    print("Total_MR:", Total_MR)
    MR_H2 = (1 - Total_MR)/1.157
    print("MR_H2:", MR_H2)  

    #Calculate hydrogen to helium ratio
    CurrentTarget.InitiateSystem(PlanetaryDict)
    CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)

    

    #Create the model
    Model = TransmissionSpectroscopy(CurrentTarget)
    Model.CalculateTransmission(CurrentTarget, ShowPlot=False)

    #How do you create the figure for water vs other molecules.
    #Create darker feature around water.
    #BinnedWavelength, BinnedModel = BinTheModel(CurrentTarget.WavelengthArray*1e4, Model.Spectrum, NBins=1500)

    if WaterAmount<=-1.75:
        ax1.plot(CurrentTarget.WavelengthArray*1e4, Model.Spectrum, color=colorList[Counter], lw=2, label=Label)

    if WaterAmount>=-1.75:
        ax2.plot(CurrentTarget.WavelengthArray*1e4, Model.Spectrum, color=colorList[Counter], lw=2, label=Label)    
    #plt.plot(CurrentTarget.WavelengthArray*1e4, Model.Spectrum, color=colorList[Counter], label=Label)



ax1.set_xlim(0.5, 5.3)
#ax1.set_ylim(14400, 15800)
ax1.set_xticks([])
ax2.set_xlim(0.5, 5.3)
#ax2.set_ylim(14400, 15800)
ax1.set_ylabel('Transit Depth [ppm]', fontsize=30)

ax1.text(1.0, 15650, "Low Metallicity")
ax1.text(1.0, 15285, "\\rm{H$_2$O}")
ax1.text(1.30, 15480, "\\rm{H$_2$O}")
ax1.text(1.8, 15430, "\\rm{H$_2$O}")
ax1.text(2.65, 15580, "\\rm{H$_2$O}")
ax1.text(4.200, 15580, "\\rm{CO$_2$}")
ax1.text(4.200, 15580, "\\rm{CO$_2$}")

ax2.text(1.0, 15285, "\\rm{H$_2$O}")
ax2.text(1.30, 15480, "\\rm{H$_2$O}")
ax2.text(1.8, 15430, "\\rm{H$_2$O}")
ax2.text(2.65, 15580, "\\rm{H$_2$O}")
ax2.text(4.200, 15500, "\\rm{CO$_2$}")
ax2.text(4.200, 15500, "\\rm{CO$_2$}")

ax2.set_ylabel('Transit Depth [ppm]', fontsize=30)
ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=30)
ax2.text(1.0, 15650, "High Metallicity")


ax1.legend(loc=4, ncols=2, fontsize=20)
ax2.legend(loc=4, ncols=2, fontsize=20)
plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig("DifferentWaterAbundance_Square_LongerRange_HighResolution.png")
plt.savefig("DifferentWaterAbundance_Square_LongerRange_HighResolution.pdf")
plt.close()

#Describe tierra and the limitations of the model. 
#Show impact of the molecules for each of the molecules.