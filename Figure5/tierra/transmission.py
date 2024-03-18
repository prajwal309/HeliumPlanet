import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left
import sys
from numba import prange


class TransmissionSpectroscopy:

    def __init__(self, Target, beta=-4.0, tau0=0.0001, CIA=False):
        '''
        Initiate the transmission
        '''


        #Flag to collision induced absportion flag
        self.CIA = CIA
        
        self.beta = beta                          #Haze scattering factor
        self.tau0 = tau0                          #Reference tau0 for the haze scattering
        if self.CIA:
            assert hasattr(Target, 'CIA_CS')

        sz = Target.NumLayers
        self.dz_cm= np.concatenate(([Target.zValuesCm[0]], np.diff(Target.zValuesCm)))


        Z_ii, Z_jj = np.meshgrid(Target.zValuesCm[1:], Target.zValuesCm[:-1])
        self.x__ = np.sqrt((Target.Rp+Z_ii)*(Target.Rp+Z_ii)
                   -(Target.Rp+Z_jj)*(Target.Rp+Z_jj))
        self.x__[np.isnan(self.x__)]=0.0


        x__Copy = np.copy(self.x__)
        self.xNew_ = np.pad(x__Copy,(1,0), mode='constant')[1:,:-1]
        self.ds_= self.x__- self.xNew_

        #Averaging the distance
        dsCopy = np.copy(self.ds_)
        self.dsNew_ = np.pad(dsCopy,(1,0),mode='constant')[1:,:-1]
        self.ds_ = 0.5*(self.ds_ + self.dsNew_)

        ###########################################################################
        ###########################################################################




    def CalculateTransmission(self, Target):
        '''
        This method calculates the spectrum given the planetary parameters.

        Parameters:
        -----------
        Target: Tierra Target object

        ShowPlot: Boolean

        interpolation: string
            Either use the bilinear or hill method

        Returns
        --------
        Array

        Spectrum of the planet is returned.
        '''

        #Initiating the alpha function
        self.alpha = np.zeros((len(Target.WavelengthArray),Target.NumLayers),dtype=np.float64)

        #for i in range(Target.NumLayers):
        #    self.alpha[:,i] += self.tau0*(Target.WavelengthArray/5e-5)**self.beta*np.exp(-Target.zValuesCm[i]/Target.H0cm)

        #Now solve for the atmosphere of the planet
        self.Spectrum = np.zeros(len(Target.WavelengthArray), dtype=np.float64)


        #for self.CurrentLayer in prange(Target.NumLayers):
        for self.CurrentLayer in range(Target.NumLayers):
            CurrentT = Target.TzAnalytical[self.CurrentLayer]
            CurrentP = np.log10(Target.PzAnalytical[self.CurrentLayer])

            TIndex = bisect_left(Target.TemperatureArray, CurrentT)
            PIndex = bisect_left(Target.PressureArray, CurrentP)
           
            co_t = (CurrentT-Target.TemperatureArray[TIndex-1])/(Target.TemperatureArray[TIndex]-Target.TemperatureArray[TIndex-1])
            if CurrentP>-5:
                co_p = (CurrentP-Target.PressureArray[PIndex-1])/(Target.PressureArray[PIndex]-Target.PressureArray[PIndex-1])
            else:
                co_p = 0.0

            assert -1e-16<co_t<1.000000001
            assert -1e-16<co_p<1.000000001

            if co_p>0:
                FirstTerm = Target.CrossSectionData[TIndex-1, PIndex-1,:,:]@Target.nz[:, self.CurrentLayer]
                SecondTerm = Target.CrossSectionData[TIndex-1, PIndex,:,:]@Target.nz[:, self.CurrentLayer]
                ThirdTerm = Target.CrossSectionData[TIndex, PIndex-1,:,:]@Target.nz[:, self.CurrentLayer]
                FourthTerm = Target.CrossSectionData[TIndex, PIndex,:,:]@Target.nz[:, self.CurrentLayer]

                self.alpha[:,self.CurrentLayer] += np.abs(((1-co_t)*(1-co_p))*FirstTerm + \
                                                            ((1-co_t)*co_p)*SecondTerm +  \
                                                            (co_t*(1-co_p))*ThirdTerm + \
                                                            (co_t*co_p)*FourthTerm)

            elif co_p == 0:
                FirstTerm = Target.CrossSectionData[TIndex-1, PIndex,:,:]@Target.nz[:, self.CurrentLayer]
                ThirdTerm = Target.CrossSectionData[TIndex, PIndex,:,:]@Target.nz[:, self.CurrentLayer]

                self.alpha[:,self.CurrentLayer] += np.abs((1-co_t)*FirstTerm + \
                                                    co_t*ThirdTerm)

            if self.CIA:
                if Target.N2Present:
                    self.alpha[:,self.CurrentLayer] += np.abs((Target.nz_N2_ama[self.CurrentLayer]*Target.nz_N2_ama[self.CurrentLayer])*((1-co_t)*Target.CIA_CS[2,TIndex-1,:]+(co_t)*Target.CIA_CS[2,TIndex,:]))
                if Target.H2Present:
                    self.alpha[:,self.CurrentLayer] +=  (Target.nz_H2_ama[self.CurrentLayer]*Target.nz_H2_ama[self.CurrentLayer])*((1-co_t)*Target.CIA_CS[0,TIndex-1,:]+(co_t)*Target.CIA_CS[0,TIndex,:]) + \
                    np.abs((Target.nz_H2_ama[self.CurrentLayer]*Target.nz_H2_ama[self.CurrentLayer]*Target.HeH2Ratio)*((1-co_t)*Target.CIA_CS[1,TIndex-1,:]+(co_t)*Target.CIA_CS[1,TIndex,:]))   

                    

        sz = Target.NumLayers

      
        self.Spectrum = ((Target.Rp)**2+ \
                        2.0*np.matmul(1.0-(np.exp(-(2.0*(np.matmul(self.alpha[:,0:sz-1],np.transpose(self.ds_[:,:sz-1])))))), \
                        (Target.Rp+Target.zValuesCm[:sz-1])*np.transpose(self.dz_cm[:sz-1])))/Target.Rs**2
                        
        self.Spectrum = self.Spectrum.flatten()*1e6 #Converting into ppm

        ##Following two are equivaluent
        self.SpectrumHeight = 0.5*(self.Spectrum/Target.Rp-Target.Rp)*1e-5


        return self.SpectrumHeight/1e5 #This is in the km



