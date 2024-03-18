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
    #PlanetaryDict['Radius']=1.359*11.2089
    PlanetaryDict['Radius']=1.34*11.2089
    PlanetaryDict['RadiusErr']=0.019*11.2089

    '''
    Residual:33.32471021581157
    Mass:217.83986888308795
    P0:-0.49399537848543484
    T0:870.3398745821831
    Vplanet:74.02975368417204   
    LOG10HeH2Ratio:0.30854102501751185
    LogMR_CH4:-10.121447637389048
    LogMR_CO:-4.3215869469076065
    LogMR_CO2:-5.211554246258529
    LogMR_H2O:-1.4846811186940945
    LogMR_H2S:-4.173691820357737
    LogMR_HCN:-9.77224759424785
    LogMR_NH3:-4.4701921644558045
    LogMR_SO2:-6.222072894731108

    Residual:33.40143882991212
    Mass:218.32024615154458
    P0:-0.716365602923927
    T0:854.4727794128696
    Vplanet:-30.40897675253926
    LOG10HeH2Ratio:-1.685985567012477
    LogMR_CH4:-8.777285714948107
    LogMR_CO:-4.930238502421385
    LogMR_CO2:-4.757890420010508
    LogMR_H2O:-0.9214792564741192
    LogMR_H2S:-2.9483310576015196
    LogMR_HCN:-6.345385195298593
    LogMR_NH3:-11.47516285623845
    LogMR_SO2:-6.762903625662893
    '''


    PlanetaryDict['P0'] = 10**-0.716365602923927
    PlanetaryDict['T0'] = 854.4727794128696
    #PlanetaryDict['MR_CH4'] = 0.0#10**-10.25
    #PlanetaryDict['MR_CO'] = 0.0#10**-4.9367
    PlanetaryDict['MR_CO2'] = 10**-4.757890420010508
    #PlanetaryDict['MR_H2O'] = 10**-1.3087
    PlanetaryDict['MR_H2O'] = 10**-0.9214792564741192
    PlanetaryDict['MR_H2S'] = 10**-2.9483310576015196
    #PlanetaryDict['MR_HCN'] = 0.0#10**-6.918
    #PlanetaryDict['MR_NH3'] = 0.0#10**-4.337
    #PlanetaryDict['MR_SO2'] = 0.0#10**-8.330
    #PlanetaryDict['MR_H2'] = 0.0
    PlanetaryDict['HeH2Ratio']=10**-1.685985567012477
    PlanetaryDict['PT']=1
else:
    assert 1 == 2, "Target parameters needs to be manually set."


Location = "/media/prajwal/LaCie/CrossSectionFromSuperCloud4JWST/CrossSectionTIERRA"



#MoleculeNames = ["CH4", "CO", "CO2", "H2O", "H2S", "HCN", "NH3", "SO2", "H2"]
MoleculeNames = ["CO2", "H2O", "H2S", "H2"]


DifferentWaterAmount = [round(np.log10(PlanetaryDict['MR_H2O']),6)]


#Use different colors for different type of models.
colorList = ["#0d0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921"]

fig = plt.figure(figsize=(12,7))
gs = GridSpec(1, 1, figure=fig)

# Plot data on the first axis (top)
ax1 = fig.add_subplot(gs[0, 0])



for Counter, WaterAmount in enumerate(DifferentWaterAmount):
    
    print("\n\n", Counter, WaterAmount)

    #Model = transmission(StellarDict, PlanetaryDict, WaterAmount)
    Label = 'Log MR$_{\\rm H_2O}$ = '+str(WaterAmount)

    #PlanetaryDict['MR_H2O'] = WaterAmount
    PlanetaryDict['MR_H2O'] = 10**WaterAmount

    Total_MR = 0
    for keys, values in PlanetaryDict.items():
        if 'MR' in keys and not('MR_H2'==keys):
            print("keys:",keys, " = ", values)
            Total_MR += values
    
    print("Total_MR:", Total_MR)
    MR_H2 = (1 - Total_MR)/(1+PlanetaryDict['HeH2Ratio'])
    print("MR_H2:", MR_H2)  
    PlanetaryDict['MR_H2'] = MR_H2

    #Calculate hydrogen to helium ratio
    print(PlanetaryDict)
    input("Press Enter to continue...")
    CurrentTarget = System(MoleculeNames, PlanetaryDict, StellarDict)
    CurrentTarget.LoadCrossSection(Location, CIA_Flag=True)
    CurrentTarget.InitiateSystem(PlanetaryDict)

    print("The mean molecular weight:", CurrentTarget.mu)

    CurrentTarget.PT_Profile(zStep=0.25, ShowPlot=False)
    Model = TransmissionSpectroscopy(CurrentTarget, CIA=True)
    Model.CalculateTransmission(CurrentTarget, ShowPlot=False)

    #How do you create the figure for water vs other molecules.
    #Create darker feature around water.
    BinnedWavelength, BinnedModel = BinTheModel(CurrentTarget.WavelengthArray*1e4, Model.Spectrum, NBins=1500)

    ax1.plot(BinnedWavelength, BinnedModel, color="red", lw=2, label=Label)


ax1.errorbar(Wavelength, TransitDepth-50, yerr=TransitDepthErr, marker="x", color="black", capsize=3, linestyle="None", label="HD209458b Data")
ax1.errorbar(WavelengthHST, TransitDepthHST-500, yerr=TransitDepthErrHST, marker="d", color="black", capsize=3, linestyle="None", label="HD209458b Data")

ax1.set_xlim(0.3, 5.3)
#ax1.set_ylim(14400, 15800)
ax1.set_xticks([])
ax1.set_ylabel('Transit Depth [ppm]', fontsize=30)
ax1.legend(loc=0, ncols=2, fontsize=20)
plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig("DifferentWaterAbundance_WithData_WithOffset.png")
plt.savefig("DifferentWaterAbundance_WithData_WithOffset.pdf")
plt.close()

#Describe tierra and the limitations of the model. 
#Show impact of the molecules for each of the molecules.