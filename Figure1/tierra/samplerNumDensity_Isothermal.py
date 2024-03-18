from socket import inet_pton
import numpy as np
import matplotlib.pyplot as plt
import emcee


from tierra import Target
from tierra.transmission import TransmissionSpectroscopy

from bisect import bisect
import multiprocessing as mp
import os
from scipy.interpolate import interp1d

import glob
import datetime
import h5py
import time


def logLikelihood(theta, Wavelength, Depth, DepthErr):
    '''
    The log likelihood for calculation
    '''
   
    global LeastResidual, ParameterNames, CurrentSaveName, PlanetParamName, \
           nWalkers, StellarParamsDictionary, CurrentSystem

    CurrentM, P0, T0 = theta[:3]
    LogMR = theta[3:]
    
    PrintFlag = False

    ### Delete the following variables
    if P0<0 or P0>100.:
        print("The case of P0")
        return -np.inf       

    if not(100.<T0<3000.):
        print("The case of T0")
        return -np.inf            
    
    if np.log10(P0)<-10 or np.log10(P0)>2.0:
        print("The case for the base pressure")
        return -np.inf

    if sum(LogMR)>0.50:
        print("The maximum case for the mixing ratio")
        return -np.inf

    if min(LogMR)<-25.0:
        print("The minimum case of the mixing ratio")
        return -np.inf

    #Calculating the mixing ratio
    MR = 10**LogMR
    MR_H2 = (1.0-np.sum(MR))/1.157

    if MR_H2<1e-24 or MR_H2>1.0:
        print("Prior snag on the mixing ratio for hydrogen")
        return -np.inf

    #print("The mixing ratio of hydrogen is given by:", MR_H2)
    #Calculate the PT Profile
    NewPlanetaryDict ={}    
    NewPlanetaryDict['P0'] = P0

    #Arbitrarily assigned
    NewPlanetaryDict['T0'] = T0   


    for counter, MoleculeItem in enumerate(PlanetParamName):
        NewPlanetaryDict[MoleculeItem]=MR[counter]
    NewPlanetaryDict['MR_H2'] = MR_H2

    global PlanetParamsDatabase, StellarParamsDatabase
    #remove this later to make things more automatic
    NewPlanetaryDict['Mass'] = CurrentM
    NewPlanetaryDict['Radius'] = 1.22*11.2089


    CurrentSystem.InitiateSystem(NewPlanetaryDict)
    CurrentSystem.PT_Profile(zStep=0.25, ShowPlot=False)
    T1 = TransmissionSpectroscopy(CurrentSystem, CIA=True)
    T1.CalculateTransmission(CurrentSystem)
    CurrentModel = T1.Spectrum

    if PrintFlag:
        print("The current planetary parameter dictionary is given by: ", NewPlanetaryDict)
        print("The current stellar parameter dictionary is given by: ", StellarParamsDictionary)
        print("The mean molecular mass of the system is: ", CurrentSystem.mu)
        print("The radius of the system is: ", CurrentSystem.Rp)
        print("The mass of the system is: ", CurrentSystem.Mp)
        print("The gravity of the system is:", CurrentSystem.Gp)
        print("Is hydrogen present:", CurrentSystem.H2Present)
        print("The name of the molecules are given by: ", CurrentSystem.MoleculeNames)
        print("And their molecular mass is given by: ", CurrentSystem.MolecularMass)
        print("The number density at the base of the atmosphere is: ", CurrentSystem.nz0)
        print("The mixing ratios for the current system is given by: ", CurrentSystem.MixingRatios)
        input("We will wait here... 100")


    BinnedModel = np.zeros(len(Wavelength))

    if np.sum(np.isnan(BinnedModel)):
        print("There is nan in the binned model.")
        print("The value of theta is given by:", theta)
        return -np.inf

    global StartIndexAll, StopIndexAll
    counter = 0

    for StartIndex, StopIndex in zip(StartIndexAll, StopIndexAll):
        BinnedModel[counter] = np.mean(CurrentModel[StartIndex:StopIndex])
        counter+=1  


    #Add Gaussian prior in Mass
    GaussianPriorMass = (CurrentM-PlanetParamsDatabase['Mass'])**2/(PlanetParamsDatabase['MassErr'])**2
    Residual = np.sum(np.power(Depth-BinnedModel,2)/(DepthErr*DepthErr))+GaussianPriorMass
    #np.random.normal(, PlanetParamsDatabase['MassErr'], 1)[0]

    ChiSqr = -0.5*Residual

    if Residual<LeastResidual:
       print("The value of Residual is::", Residual)
       print("Saving the best model.")
       LeastResidual = Residual
       with open("MCMCParams/BestParam"+CurrentSaveName+".txt", 'w+') as f:
         f.write("Residual:"+str(Residual)+"\n")
         for key,value in zip(ParameterNames, theta):
              f.write(key+":"+str(value)+"\n")
      
       if 1==1:

          np.savetxt("BestModel.txt", np.transpose((Wavelength, BinnedModel)))
          
            
          plt.figure(figsize=(12,8))
          plt.subplot(211)
          plt.errorbar(Wavelength, Depth, yerr=DepthErr, marker=".", markersize=5, color="black", linestyle='None', capsize=2, label="Data")
          plt.plot(Wavelength, BinnedModel, "r+-", lw=3, alpha=0.4, label="Model")
          plt.xlim([min(Wavelength),max(Wavelength)])
          plt.ylabel("Transit Depth [ppm]", fontsize=20)
          plt.legend(loc=1)
          plt.subplot(212)
          plt.plot(Wavelength, (Depth-BinnedModel)/DepthErr, "ko")
          plt.xlim([min(Wavelength),max(Wavelength)])
          plt.axhline(y=0, color="red", lw=2)
          plt.xlabel("Wavelength (Microns)", fontsize=20)
          plt.ylabel("Deviation [$\sigma$]", fontsize=20)
          plt.tight_layout()
          plt.savefig("Figures/CurrentBestModel_%s.png" %CurrentSaveName)
          plt.close('all')

          if PrintFlag:
            input("Crash here...")
          print("Best Model Updated. Figure saved.")
    return ChiSqr


def RunMCMC(Molecule2Look, PlanetParamsDict, StellarParamsDict, LCFile=None, CSLocation=None, AssignedzStep=0.25,  SaveName="Default", NSteps=20000, NCORES=4, CS_Case="1", NewStart=False, SLICE=True):
    '''
    Run MCMC value.

    Parameters
    ##########

    PlanetParamDict: dictionary
                     Dictionary containing planetary parameter value

    CSLocation: string
                Base location of the cross-section

    AssignedzStep: float
                    Assigned value for the zStep size. Should be smaller than 0.15

    StellarParamDict: dictionary
                      Dictionary containing stellar parameter value

    NumberPTLayers: integer
                    Number of PT layers for the calculation

    NSteps: integer
            Number of steps
    '''


    if "gridsan" in CSLocation:
        print("Using half of the ")
        os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count()//3)
    else:    
        #By default use four cores in CS
        print("Currently overriding this by 4")
        os.environ["OMP_NUM_THREADS"] = "4"#str("NCORES")

    global LeastResidual, CurrentSaveName, CurrentzStep, StartTime, AllMolecules, PlanetParamsDatabase, StellarParamsDatabase

    PlanetParamsDatabase = PlanetParamsDict
    StellarParamsDatabase = StellarParamsDict

    print("Molecules 2 look at line 195", Molecule2Look)
    # This is temporary assignment
    Molecule2Look.append("H2")   
    PlanetParamsDict["MR_H2"]=0.90     
    print("Molecules 2 look at line 199", Molecule2Look)

    AllMolecules = Molecule2Look

    StartTime = str(datetime.datetime.now()).replace(" ", "-").replace(":","-").split(".")[0]
    CurrentSaveName = SaveName
    LeastResidual = np.inf
    CurrentzStep = AssignedzStep
    BaseLocation = CSLocation

    global CurrentSystem, StellarParamsDictionary

    

    StellarParamsDictionary = StellarParamsDict
    PlanetParamsDict['PT']=1
    CurrentSystem = Target.System(Molecule2Look, PlanetParamsDict, StellarParamsDict)

    print("Now loading CS-", CS_Case)
    CurrentSystem.LoadCrossSection(BaseLocation, SubFolder="CS_"+CS_Case, CIA_Flag=True)

    try:
        Wavelength, WavelengthBin, Depth, DepthErr, Order = np.loadtxt(LCFile, unpack=True, skiprows=1, delimiter=",")
    except:
        #This is fire firefly reduction for WASP-39b
        print("Now loading for WASP-39")
        Wavelength, WavelengthBin, Depth, DepthErr = np.loadtxt(LCFile, unpack=True)
        Depth*=1e6
        DepthErr*=1e6
    
    WavelengthLower = Wavelength-WavelengthBin/2
    WavelengthUpper = Wavelength+WavelengthBin/2
   
    #Slice
    if SLICE:
        SmallestWavelength = np.min(WavelengthLower)
        HighestWavelength = np.max(WavelengthUpper)
        StartIndex =  bisect(CurrentSystem.WavelengthArray*1e4, SmallestWavelength)-3
        StopIndex =  bisect(CurrentSystem.WavelengthArray*1e4, HighestWavelength)+3

        CurrentSystem.CrossSectionData = CurrentSystem.CrossSectionData[:,:,StartIndex:StopIndex,:]
        CurrentSystem.WavelengthArray = CurrentSystem.WavelengthArray[StartIndex:StopIndex]

        if hasattr(CurrentSystem, 'CIA_CS'):
            CurrentSystem.CIA_CS = CurrentSystem.CIA_CS[:,:,StartIndex:StopIndex]

    global StartIndexAll, StopIndexAll, nWalkers

    StartIndexAll = []
    StopIndexAll = []
    CurrentWavelength = CurrentSystem.WavelengthArray*1e4
    for Wl, Wp in zip(WavelengthLower, WavelengthUpper):
        StartIndex = bisect(CurrentWavelength, Wl)
        StopIndex = bisect(CurrentWavelength, Wp)
        StartIndexAll.append(StartIndex)
        StopIndexAll.append(StopIndex)

    #Check for the file name
    FileName = "ProgressData/{}.h5".format(SaveName)
    FileExist = os.path.exists(FileName) #len(FileName)>0 check for the files later 
     
    global backend, sampler
    backend = emcee.backends.HDFBackend(FileName)

    global ParameterNames, PlanetParamName

    #Construct the parta
    ParameterNames = ["Mass", "P0", "T0"]
    PlanetParamName = []

    for MoleculeItem in Molecule2Look[:-1]:
        ParameterNames.append("LogMR_"+MoleculeItem)
        PlanetParamName.append("MR_"+MoleculeItem)

    nDim = len(ParameterNames)
    nWalkers = nDim*4

    #Making sure there are at least some saved samples
    if FileExist:
        with h5py.File(FileName, "r") as f:
            LogProb = np.array(f['mcmc']['log_prob'])
            MeanLogProb = np.abs(np.mean(LogProb, axis=1))
            SelectIndex = MeanLogProb>1e-5
            NCases = len(MeanLogProb[SelectIndex])
            print("The number of run steps is:", NCases)
            if NCases<5:
                FileExist=False



    if not(FileExist):
        backend.reset(nWalkers, nDim)



    #Convert the mixing ratio into log of the mixing ratio
    if NewStart or not(FileExist):
        MassInit = np.random.normal(PlanetParamsDict['Mass'],PlanetParamsDict['MassErr'], nWalkers)         #The mass of the exoplanet.
        P0Init = np.random.normal(1.1, 0.02, nWalkers)
        T0Init = np.random.normal(700., 10., nWalkers)
        
        #Now for molecular number density
        LogMR = np.random.uniform(-3, -7, (nWalkers, nDim-3))     #Mixing ratio for the molecules
        
        StartingGuess = np.column_stack((MassInit, P0Init, T0Init, LogMR))

    sampler = emcee.EnsembleSampler(nWalkers, nDim, logLikelihood, args=[Wavelength, Depth, DepthErr], backend=backend)    
    
    
    if not(FileExist):
        print("Starting ANEW")
        sampler.run_mcmc(StartingGuess, NSteps, store=True)

    else:
        print("Starting NOT fresh")
        sampler.run_mcmc(None, NSteps, store=True)

    '''with mp.Pool() as pool: 
        sampler = emcee.EnsembleSampler(nWalkers, nDim, logLikelihood, args=[Wavelength, Depth, DepthErr], backend=backend, pool=pool)

        if not(FileExist):
            print("Starting ANEW")
            sampler.run_mcmc(StartingGuess, NSteps, store=True)

        else:
            print("Starting NOT fresh")
            #input("Now starting MCMC")
            sampler.run_mcmc(None, NSteps, store=True)
    '''



