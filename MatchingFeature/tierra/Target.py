import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d


class System:
    def __init__(self, MoleculeNames, PlanetParamsDict, StellarParamsDict):
        '''
        LoadFromFile: bool
                      True if to be loaded from the data

        '''

        self.InitiateConstants()


        #Store the dictionaries and parameters into class variables    
        self.PlanetParams = PlanetParamsDict
        self.StellarParams = StellarParamsDict
        self.MoleculeNames = MoleculeNames[:]  

        if 'HeH2Ratio' in PlanetParamsDict.keys():
            self.RatioHeH2 = PlanetParamsDict['HeH2Ratio']
            Total_MR = 0
            for keys, values in PlanetParamsDict.items():
                 if 'MR' in keys and not('MR_H2' == keys) and not('MR_He' == keys):
                    Total_MR += values
            print("Total_MR:", Total_MR)
            self.PlanetParams['MR_H2'] = (1-Total_MR)/(1+self.RatioHeH2)
            print("The mixing ratio of hydrogen is:", self.PlanetParams['MR_H2'])


        LocationN2Index = (np.array(self.MoleculeNames)=="MR_N2")+(np.array(self.MoleculeNames)=="N2")
        self.N2Present =  np.sum(LocationN2Index)==1
        if self.N2Present: 
            self.N2Index = np.where(LocationN2Index)[0][0]

        self.LocationH2Index = (np.array(self.MoleculeNames)=="MR_H2")+(np.array(self.MoleculeNames)=="H2")
        self.H2Present =  np.sum(self.LocationH2Index)==1


        if self.H2Present: 
            self.H2Index = np.where(self.LocationH2Index)[0][0]   
            #Add Helium according to the ratio of 1:self.RatioHeH2
            self.MoleculeNames.append("He")  
        
        self.PT = self.PlanetParams['PT']

        self.MolDict = {"H2O":18.010565, "CO2":43.989830, "O3":47.984745,  "N2O":44.001062, \
        "HCl":35.976678, "CO":27.994915,  "CH4":16.031300, "NH3":17.026549, \
        "O2":31.98983,   "H2":2.015650,   "He":4.002602,   "N2":28.006148, \
        "C2H2":26.015650, "C4H2":50.015650, "C2H4":28.031300, "C2H6":30.046950,\
        "13CH4":17.034655, "CH3D":17.037475, "HC3N":51.010899, "HCN":27.010899,\
        "C3H8":44.09562, "SO2":63.961901, "Na":22.989769, "K":39.0983, "H2S":33.987721}


        self.MolecularMass = []
        for Molecule in self.MoleculeNames:
            self.MolecularMass.append(self.MolDict[Molecule])
        self.MolecularMass = np.array(self.MolecularMass)

       
        if 'Mass' in self.StellarParams.keys():
            self.Ms = self.StellarParams['Mass']*self.M_sun
        if 'Radius' in self.StellarParams.keys():
            self.Rs = self.StellarParams['Radius']*self.R_sun
            self.RsKm = self.Rs/1.e5

        
       
        
        #Now calculate the mean molecular mass of the atmosphere
        self.InitiateSystem(PlanetParamsDict)

        #To check in the cross-section has been loaded
        self.CSDataLoaded = False



    def InitiateSystem(self, PlanetParamsDict):
        '''
        Initiate the calculation for mean molecular mass
        and assign the value for pressure and temperature
        '''
        
        #Now setting up for PT profile
        self.Rp = PlanetParamsDict['Radius']*self.R_ear          #Radius in cm
        self.Mp = PlanetParamsDict['Mass']*self.M_ear            #Mass in grams
        self.RpKm = self.Rp/1.e5
        self.Gp = self.G_gr*self.Mp/(self.Rp*self.Rp)
        self.P0 = PlanetParamsDict['P0']*self.P_atm              #Pressure in atmospheric pressure converted to Pascal
        self.T0 = PlanetParamsDict['T0']                         #Temperature at Rp

        if 'HeH2Ratio' in PlanetParamsDict.keys():
            self.HeH2Ratio = PlanetParamsDict['HeH2Ratio']
            Total_MR = 0
            for keys, values in PlanetParamsDict.items():
                if 'MR' in keys and not('MR_H2' == keys) and not('MR_He' == keys):
                    Total_MR += values
            self.PlanetParams['MR_H2'] = (1-Total_MR)/(1+self.HeH2Ratio)
        
        if int(self.PT)==2:
            self.Tinf = PlanetParamsDict['Tinf']
            self.Alpha = PlanetParamsDict['Alpha']           


        self.MixingRatios = []

        for MolItem in self.MoleculeNames:
            if not("He" in MolItem):
                NewKey = "MR_"+MolItem
                self.MixingRatios.append(PlanetParamsDict[NewKey])

        if self.H2Present: 
            self.MixingRatios.append(self.MixingRatios[self.H2Index ]*self.HeH2Ratio)  
            PlanetParamsDict["MR_He"] = self.MixingRatios[-1]

        self.MixingRatios = np.array(self.MixingRatios)
        self.mu = np.sum(self.MixingRatios*self.MolecularMass)
        #print("The self.MixingRatios:", self.MixingRatios)
        #print("The self.MolecularMass:", self.MolecularMass)
        #input("Wait here@126...")
        

    def ParsePlanetFile(self):
        '''
        This function parses the planetary file
        '''
        self.PlanetParams = {}

        if os.path.exists("PlanetParam.ini"):
            FileContent = open("PlanetParam.ini", "r").readlines()
            for Line in FileContent:
                Item = Line.split("#")[0].replace(" ","")
                key, Value = Item.split(":")
                self.PlanetParams[key]=float(Value)
        else:
            print("PlanetParam.ini does not exist in the local dictionary")


    def ParseStarFile(self):
        '''
        This function parses the star file i.e StelarParam.ini
        '''

        self.StellarParams = {}

        if os.path.exists("StellarParam.ini"):
            FileContent = open("StellarParam.ini", "r").readlines()
            for Line in FileContent:
                Item = Line.split("#")[0].replace(" ","")
                key, Value = Item.split(":")
                self.StellarParams[key]=float(Value)
        else:
            print("StellarParam.ini does nopt exist in the local dictionary")


    def InitiateConstants(self):
        #Astronomical Constants
        self.R_sun=6.957E10                     #Sun radius in cm
        self.T_sun=5770                         #Effective Temperature of the Sun
        self.M_sun=1.98911e33                   #Mass of the sun in kilograms
        self.P_terre=86400*365.25696            #Days in seconds
        self.R_ear=6.371e8                      #earth's radius in centimeters
        self.r_t=1.496e13                       #1 AU in centimeters
        self.parsec = 3.085677e18               #parsec in centimeters
        self.M_jup = 1.8986e30                  #Mass of Jupiter in grams
        self.M_ear = 5.9736e27                  #Mass of the earth in grams
        self.R_jup = 6.995e9                    #Radius of Jupiter in centi meters

        #Physical constants
        self.G_gr = 6.67384E-8                  #Gravitational constant in CGS
        self.c = 2.998E10                       #Speed of Light in CGS
        self.h_pl = 6.626069E-27                #Planck's Constant in CGS
        self.k_bo = 1.38065E-16                 #Boltzmann Constant in CGS
        self.P_atm= 1.013e6                     #1 atmospheric pressure of earth in CGS
        self.N_av = 6.02214E23                  #Avogadro's Number
        self.sigma_bo = 5.670E-5                #stefan Boltzmann's Constant
        self.loschmidt = 2.6867811E19           #Lochsmidt number


    def PT_Profile(self, zStep=0.25, ShowPlot=False, verbose=False):
        '''
        This method calculates the Pressure Temperature profile
        for the planet as well as the number density as the function of 
        altitude(z).

        Parameters:
        -----------------

        zStep: Float
                   Stepsize in atmospheric scale.

        ShowPlot: Boolean
                  Default value is False. Plot the data if True.

        '''

        #assert np.sign(self.Tinf-self.T0) == -np.sign(self.Gam)
       
        self.ScaleHeight = np.arange(0,100,zStep)    
        self.H0 = self.k_bo*self.T0/(self.mu/self.N_av*self.Gp)/1e5 #H0 is in kilometers
        self.H0cm = self.H0*1e5
        self.MuNumDensity = (1.e-6/self.k_bo)*self.MixingRatios*(self.P0/self.T0)           #in cm-3

       

        if self.PT==1:
            if verbose:
                print("Using the isothermal profile.")
            self.zValues = self.ScaleHeight*self.H0 #In kilometers
            self.dz = np.diff(self.zValues) #In kilometers
            self.zValuesCm=self.zValues*1e5
            self.Gz = self.G_gr*self.Mp/((self.Rp+self.zValuesCm)**2)

            self.TzAnalytical = np.ones(len(self.zValues))*self.T0
            self.Hz = self.k_bo*self.TzAnalytical/(self.mu/self.N_av*self.Gz)/1e5

            self.PzAnalytical = [self.P0/self.P_atm]
            for i in range(len(self.TzAnalytical)-1):
                self.PzAnalytical.append(self.PzAnalytical[-1]*np.exp(-self.dz[i]/self.Hz[i]))
            self.PzAnalytical = np.array(self.PzAnalytical)    #In atmosphere 
            self.PzAnalyticalLog = np.log10(self.PzAnalytical) #In atm
            self.dz_cm = self.dz*1e5

            SelectIndex = self.PzAnalyticalLog>-10.0
            self.PzAnalytical = self.PzAnalytical[SelectIndex]
            self.TzAnalytical = self.TzAnalytical[SelectIndex]
            self.zValues = self.zValues[SelectIndex]
            self.dz = np.diff(self.zValues)
            self.zValuesCm = self.zValuesCm[SelectIndex]
            #self.Gz = self.Gz[SelectIndex]
            #self.Hz = self.Hz[SelectIndex]

        elif self.PT==2: 
            if verbose:
                print("Using PT profile based on the transmission model.")    
            CurrentGravity0= self.G_gr*self.Mp/((self.Rp)**2)    
            self.H0 = self.k_bo*self.T0/(self.mu/self.N_av*CurrentGravity0)/1e5   
            self.zValuesOld = self.ScaleHeight*self.H0 #In kilometers
            self.zValuesCmOld = self.zValuesOld*1e5    #In centimeter
            
        
            #Now use the second parametric case...
            self.PzAnalytical = self.P0/self.P_atm*np.exp(-self.ScaleHeight)     #In units of atmosphere
            self.PzAnalyticalLog = np.log10(self.PzAnalytical)
            self.TzAnalytical = self.Tinf+(self.T0-self.Tinf)*np.exp((-self.Alpha*(self.PzAnalyticalLog+1.0)**2))

            self.Gz = self.G_gr*self.Mp/((self.Rp+self.zValuesCmOld)**2)
            self.Hz = self.H0*self.TzAnalytical/self.T0*CurrentGravity0/self.Gz
           
            self.zValues = [0]
            for HzValues in self.Hz[:-1]:
                self.zValues.append(self.zValues[-1]+HzValues*zStep)
            self.zValues = np.array(self.zValues)

            self.zValuesCm = self.zValues*1e5

            self.dz = np.diff(self.zValues)
            self.dz_cm = self.dz*1e5
            SelectIndex = self.PzAnalyticalLog>-10.0
            self.PzAnalytical = self.PzAnalytical[SelectIndex]

            #Truncating the temperature at 10^-10 atmosphere
            self.TzAnalytical = self.TzAnalytical[SelectIndex]
            self.zValues = self.zValues[SelectIndex]
            self.dz = np.diff(self.zValues)
            self.zValuesCm = self.zValuesCm[SelectIndex]
            self.Gz = self.Gz[SelectIndex]
            self.Hz = self.Hz[SelectIndex]

        
    

        #Number density in per cm^3
        self.nz0 = self.N_av/22400.0*self.P0/self.P_atm*273.15/self.T0*self.MixingRatios
        self.NumLayers = len(self.zValues)

        self.nz = np.zeros((len(self.nz0), len(self.PzAnalytical)))

        for i in range(len(self.nz0)):
           self.nz[i,:] = self.nz0[i]*self.PzAnalytical/self.PzAnalytical[0]

        if self.N2Present:
            self.nz_N2_ama = self.nz[self.N2Index, :]

        if self.H2Present:
            self.nz_H2_ama = self.nz[self.H2Index, :]    


        if ShowPlot:
            #Generating the figure
          
            fig, ax = plt.subplots(figsize=(14,6),nrows=1, ncols=2)
            ax[0].plot(self.PzAnalytical, self.zValues, "r-", linewidth=2.5)
            ax[0].set_xlabel('Pressure (atm)', color='red', fontsize=20)
            ax[0].set_ylabel('Atmosphere (km)', color='blue', fontsize=20)
            ax[0].grid(True)
            ax[0].tick_params(axis='x', labelcolor='red')
            ax[0].set_xscale('log')

            ax_0 = ax[0].twiny()
            ax_0.plot(self.TzAnalytical, self.zValues, "g-", linewidth=2.5)
            ax_0.set_xlabel('Temperature (K)', color='green', fontsize=20)
            ax_0.tick_params(axis='x', labelcolor='green')
            ax[0].set_ylim([min(self.zValues), max(self.zValues)])

            #Name of the molecules
            LineStyles = [':', '-.', '--', '-']
            for i in range(len(self.nz0)):
                ax[1].plot(self.nz[i,:], self.zValues,linewidth=1, \
                linestyle=LineStyles[i%len(LineStyles)], label=self.MoleculeNames[i])
            ax[1].set_ylim([min(self.zValues), max(self.zValues)])
            ax[1].grid(True)
            ax[1].set_xscale('log')
            ax[1].set_ylabel('Atmosphere (km)', color='blue', fontsize=20)
            ax[1].legend(loc=1)
            plt.tight_layout()
            plt.show()



    def LoadCrossSection(self, Location, SubFolder="CS_1", CIA_Flag=False):
        '''
        The location is expected to have
        Pressure.txt where the pressure values are provided
        Temperature.txt where the temperature values are provided
        Wavelength.npy where the wavelength values are provided.

        AllLocations: 
            The list of molecules

        This method is supposed to load the cross-section

        The expected location is
        '''

        #Now load the cross-section based on the MR
        #Now assign the name of the molecule we have here...
       
        assert os.path.exists(Location+"/Wavelength.npy"), "Wavelength.npy is needed "
        
        self.WavelengthArray = np.load(Location+"/Wavelength.npy")
        self.TemperatureArray = np.loadtxt(Location+"/Temperature.txt")
        self.PressureArray = np.loadtxt(Location+"/Pressure.txt")

        if CIA_Flag:

            self.CIA_CS = np.load(Location+"/CIA/CIA.npy")
            #print("The shape of CIA is given by:", np.shape(self.CIA_CS))

            #plt.figure()
            #plt.plot(self.WavelengthArray, self.CIA_CS[0,41,:])
            #plt.xlabel("Wavelength [microns]")
            #plt.ylabel("CIA Cross Section [cm$^2$]")
            #plt.tight_layout()
            #plt.show()

        NumWavelength = len(self.WavelengthArray)
        NumTemp = len(self.TemperatureArray)
        NumPressure = len(self.PressureArray)
        
        #Copy the code to load the cross section here
        All_CS_Available = np.array(glob.glob(Location+"/"+SubFolder+"/*.npy"))
        MoleculesAvailable = [Item.split("/")[-1].replace(".npy","") for Item in All_CS_Available]
        self.SelectedPath = []
        for CurrentMolecule in self.MoleculeNames:
            if not("He" in CurrentMolecule):
                SelectIndex = np.array([(Path==CurrentMolecule) for Path in MoleculesAvailable])
                assert np.sum(SelectIndex)==1,  "%s Molecule Not found" %CurrentMolecule
                self.SelectedPath.append(str(All_CS_Available[SelectIndex][0]))


        #Make the CS matrix based on the number of molecules assigned        
        self.CrossSectionData = np.zeros((NumTemp, NumPressure, NumWavelength, len(self.MoleculeNames)))
        
        for counter, CurrentPath in enumerate(self.SelectedPath):
            print("Now loading:", CurrentPath)
            CurrentCS = np.load(CurrentPath)
            self.CrossSectionData[:,:,:,counter] = CurrentCS[:,:,:]

        self.CSDataLoaded = True
       
        

