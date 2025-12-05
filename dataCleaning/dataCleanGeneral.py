## THIS CODE TAKES A DATASET WITH THE FOLLOWING COLUMNS AND RETURNS A CSV FILE THAT 
    ## IS USED TO PLOT A COLOR COLOR DIAGRAM AND/OR A COLOR-MAG DIAGRAM

## It requires the dataset to be matched with Gaia DR2/3 and with the following columns...

# ['Name', some 'DR3/2Name', 'RA', 'DEC', 'Gmag', 'BPmag', 'RPmag', 'BP-RP', 'nuv_mag', 'nuv_magerr', # 'e_bv', 'FeH', 'parallax'] 

import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import sys
#import itertools
import os
from gaiaReddening import aLambdaGaiaCalc
#import mpl_scatter_density # adds projection='scatter_density'
#from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import time
from astroquery.simbad import Simbad
from astroquery.irsa_dust import IrsaDust
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from datetime import datetime

from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import Angle
from astropy.table import Table

### used for the individual literature data --------------------------

def dfFromTxtFile(f): 

    #TAKES A TXT FILE PATH FROM UPLOADTOYSE, GETS PANDAS READ_CSV OF IT, AND TURNS IT INTO A PANDAS DATAFRAME
    fExtract = pd.read_csv(f, sep="\t")

    colString = list(fExtract.columns)[0]
    columnNames = colString.split(',')
    columnNames = [i.strip() for i in columnNames]
    
    #print(len(columnNames))

    lstOfValues = []

    for val in tqdm(fExtract[colString].to_numpy(), desc="Turning CSV file to DF...", unit="iteration"):
        lstOfValues.append(val.split(','))
    
    finalDF = pd.DataFrame(lstOfValues, columns = columnNames)
    return finalDF
    #print("---------COLUMN NAMES---------", columnNames) #SPLIT IS the column names

def convertParallaxtoArcsec(df):
    parallax = [i/1000 for i in df['parallax'].to_list()]  #gaia reports in milliarcsec, so convert to arcsec
    df['parallax'] = parallax

    return df

def insertReddeningCol(df):
    # returns reddening magnitudes that must be subtracted from photo_gmeanmag/bpmeanmag/rp... to get dereddened
    print("Starting reddening calculations")
    bprpList = [float(i) for i in df["BP-RP"].tolist()]
    ebvList = [float(i) for i in df["e_bv"].tolist()]
    lenBPRP = len(bprpList)

    redGList = np.empty(0)
    redBPList = np.empty(0)
    redRPList = np.empty(0)
    n = 10
    j = 1
    iterations = len(bprpList)
    
    for i in tqdm(range(iterations), desc="Calculating reddening...", unit="iteration"):
        bprp = bprpList[i]
        ebv = ebvList[i]
        redG, redBP, redRP = aLambdaGaiaCalc(bprp, ebv, n)
        redGList = np.append(redGList, np.mean(redG))
        redBPList = np.append(redBPList, np.mean(redBP))
        redRPList = np.append(redRPList, np.mean(redRP))
        j += 1

    df["redG"] = redGList
    df["redBP"] = redBPList
    df["redRP"] = redRPList

def deredden(df):
    bpList = [float(i) for i in df["BPmag"].tolist()]
    rpList = [float(i) for i in df["RPmag"].tolist()]
    gList = [float(i) for i in df["Gmag"].tolist()]
    nuvList = [float(i) for i in df["nuv_mag"].tolist()]
    
    ebvList = [float(i) for i in df["e_bv"].tolist()]

    bpRedFactorList = [float(i) for i in df["redBP"].tolist()]
    rpRedFactorList = [float(i) for i in df["redRP"].tolist()]
    gRedFactorList = [float(i) for i in df["redG"].tolist()]

    bpDered = np.empty(0)
    rpDered = np.empty(0)
    gDered = np.empty(0)
    NUVDered = np.empty(0)

    iterations = len(bpList)
    
    for i in tqdm(range(iterations), desc="Dereddening...", unit="iteration"):
        bp = bpList[i]
        bpDered = np.append(bpDered, bpList[i] - bpRedFactorList[i])
        rpDered = np.append(rpDered, rpList[i] - rpRedFactorList[i])
        gDered = np.append(gDered, gList[i] - gRedFactorList[i])
        NUVDered = np.append(NUVDered, deredden_nuv(nuvList[i], ebvList[i]))

    df["bpDered"] = bpDered
    df["rpDered"] = rpDered
    df["gDered"] = gDered
    df["NUVDered"] = NUVDered
    df["bp_rp_dered"] = [bpDered[i] - rpDered[i] for i in range(len(bpDered))]

    print("DONE dereddening bp, rp, and g and inserting columns in DF")

def deredden_nuv(nuv_magnitude, ebv):
    """
    Deredden NUV magnitudes using the provided E(B-V) and extinction coefficient.

    Parameters:
    nuv_magnitude value of observed NUV magnitude
    ebv (float): Color excess E(B-V).
    k_nuv (float): Extinction coefficient for NUV band (default is 8.2).

    Returns:
    Dereddened NUV magnitude
    """

    ## https://academic.oup.com/mnras/article/489/4/5046/5565069
            ## second col of table 1 uses ebv. table 4 says, using 2nd col, k_nuv is 7.24. 
            ## this value is the one I used.
            ## unc is +/- 0.08 (use later if needed)

    k_nuv = 7.24 
    a_nuv = k_nuv * ebv
    nuv_dereddened = nuv_magnitude - a_nuv

    return nuv_dereddened

def insertAbsMagCol(df):
        absMag = np.empty(0)
        absMagDered = np.empty(0)
        gMag = [float(i) for i in df['Gmag'].to_list()]
        gMagDered = [float(i) for i in df['gDered'].to_list()]
        gaiaDR3IDLst = [int(i.split()[0]) for i in df['DR3Name'].to_list()]
        
        parallax = np.empty(0)

        for gaiaObjID in gaiaDR3IDLst:
            customSimbad = Simbad()
            customSimbad.add_votable_fields('parallax')
            temp = customSimbad.query_object(f"Gaia DR3 {gaiaObjID}")
            if temp is None:
                parallax = np.append(parallax, "")
            else:
                parallax = np.append(parallax, temp["PLX_VALUE"][0])
        
        parallax = [i/1000 for i in parallax]  #SIMBAD reports in milliarcsec, so convert to arcsec (CHECK IF IT IS)
        df['parallax'] = parallax

        iterations = len(gMag)
        
        for i in tqdm(range(iterations), desc="Calculating Absolute G_Magnitude for stars...", unit="iteration"):
            dist = 1/parallax[i]
            mag = gMag[i] - (5*np.log10(dist)) + 5
            magDered = gMagDered[i] - (5*np.log10(dist)) + 5
            absMag = np.append(absMag, mag)
            absMagDered = np.append(absMagDered, magDered)
        
        df['absMag'] = absMag
        df['absMagDered'] = absMagDered

def colorColorShiftYAxisDown(df):
    #takes columns and adds a new column with the y axis for colorcolor plot shifted to be horiz
    bprpLst = [float(i) for i in df['bp_rp_dered'].to_list()]
    nuvList = [float(i) for i in df['NUVDered'].tolist()]
    gList = [float(i) for i in df['Gmag'].tolist()]
    nuvErrList = [float(i) for i in df['nuv_magerr'].tolist()]
    #gErr = [float(i) for i in df['e_Gmag'].tolist()]
    #print(gErr[:5])
    metalConst = 6.5 #2.731 #4.096 #2.731 #was 4.731
    shiftVal = 0 #1.366
    yaxisList = [(nuvList[i] - gList[i]) - ((bprpLst[i] * metalConst) + shiftVal) for i in range(len(gList))]
    #yErrList = [nuvErrList[i]+gErr[i]+(metalConst*gErr[i]) for i in range(len(gErr))]
    df['colorColorY'] = yaxisList
    #df['colorColorYErr'] = yErrList
    return df

def saveDFasCSV(df, csvFileName):
    df.to_csv(csvFileName, header = True)
    print("Saved DF to CSV file")

if __name__ == '__main__':
    fName = 'gaiaGalexCayrelDatForDataClean.csv'
    paperGaiaDF = dfFromTxtFile(fName)

    insertReddeningCol(paperGaiaDF)
    deredden(paperGaiaDF)
    #insertAbsMagCol(cutSanilMastDF)
    finalDF = colorColorShiftYAxisDown(paperGaiaDF)
    print(finalDF.head(10))
    print(finalDF.columns.tolist())
    saveDFasCSV(finalDF, 'gaiaGalexCayrelDatForPlotting.csv')
