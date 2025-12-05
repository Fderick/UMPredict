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

### used for the gaia GALEX crossmatched data --------------------------

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

def fitstoDF(f): 
    #opens a .fits file and turns it into a pandas dataframe
    dat = Table.read(f, format='fits')
    df = dat.to_pandas()
    return df

def downsizeDF(df, colNameList):
    # taking full DF and extracting columns written in colNameList and rewriting to new df
    newDF = df[colNameList].copy()
    return newDF

def insertReddeningCol(df):
    # returns reddening magnitudes that must be subtracted from photo_gmeanmag/bpmeanmag/rp... to get dereddened
    print("Starting reddening calculations")
    bprpList = df["bp_rp"].tolist()
    ebvList = df["e_bv"].tolist()
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

    #print(df[["GALEX_ID", "Gaia_ID", "e_bv", "bp_rp", "redG", "redBP", "redRP"]].head(10))
    #     

def insertAbsMagCol(df):
    absMag = np.empty(0)
    absMagDered = np.empty(0)
    gMag = df['phot_g_mean_mag'].to_list()
    gMagDered = df['gDered'].to_list()
    parallax = [i/1000 for i in df['parallax'].to_list()]  #gaia reports in milliarcsec, so convert to arcsec
    
    iterations = len(gMag)
    
    for i in tqdm(range(iterations), desc="Calculating Absolute G_Magnitude...", unit="iteration"):
        dist = 1/parallax[i]
        mag = gMag[i] - (5*np.log10(dist)) + 5
        magDered = gMagDered[i] - (5*np.log10(dist)) + 5
        absMag = np.append(absMag, mag)
        absMagDered = np.append(absMagDered, magDered)
    
    df['absMag'] = absMag
    df['absMagDered'] = absMagDered

def getRidOfObjectsWithBlanks(df):
    print("Getting rid of objects with blanks")
    ## takes df and gets rid of all objects with empty string '' for the bp-rp value
    df = df.drop(df[df['bp_rp'] == ''].index)
    df = df.drop(df[df['parallax_over_error'] == ''].index)

    return df

def getRidOfObjectsWithBlanksPost(df):
    df = df.drop(df[df['mh_gspphot'] == ''].index)
    df = df.drop(df[df['teff_gspphot'] == ''].index)
    df = df.drop(df[df['logg_gspphot'] == ''].index)
    df = df.drop(df[df['mh_gspphot'].isna()].index)
    df = df.drop(df[df['teff_gspphot'].isna()].index)
    df = df.drop(df[df['logg_gspphot'].isna()].index)

    df = df.drop(df[df['mh_gspspec'] == ''].index)
    df = df.drop(df[df['teff_gspspec'] == ''].index)
    df = df.drop(df[df['logg_gspspec'] == ''].index)
    df = df.drop(df[df['mh_gspspec'].isna()].index)
    df = df.drop(df[df['teff_gspspec'].isna()].index)
    df = df.drop(df[df['logg_gspspec'].isna()].index)
    return df

def stringToFloat(df):
    print("Turning all 'string type' data values to floats")
    #takes all columns that we need as floats but they are strings and turns them into floats
    df['bp_rp'] = df["bp_rp"].apply(lambda x: float(x))
    df['e_bv'] = df["e_bv"].apply(lambda x: float(x))
    df['phot_g_mean_mag'] = df["phot_g_mean_mag"].apply(lambda x: float(x))
    df['phot_bp_mean_mag'] = df["phot_bp_mean_mag"].apply(lambda x: float(x))
    df['phot_rp_mean_mag'] = df["phot_rp_mean_mag"].apply(lambda x: float(x))
    df['nuv_mag'] = df["nuv_mag"].apply(lambda x: float(x))
    df['parallax_over_error'] = df["parallax_over_error"].apply(lambda x: float(x))
    df['phot_g_mean_flux_over_error'] = df["phot_g_mean_flux_over_error"].apply(lambda x: float(x))
    df['phot_rp_mean_flux_over_error'] = df["phot_rp_mean_flux_over_error"].apply(lambda x: float(x))
    df['phot_bp_mean_flux_over_error'] = df["phot_bp_mean_flux_over_error"].apply(lambda x: float(x))
    df['phot_bp_rp_excess_factor'] = df["phot_bp_rp_excess_factor"].apply(lambda x: float(x))
    df['parallax'] = df["parallax"].apply(lambda x: float(x))

def deredden(df):
    bpList = df["phot_bp_mean_mag"].tolist()
    rpList = df["phot_rp_mean_mag"].tolist()
    gList = df["phot_g_mean_mag"].tolist()
    nuvList = df["nuv_mag"].tolist()
    ebvList = df["e_bv"].tolist()

    bpRedFactorList = df["redBP"].tolist()
    rpRedFactorList = df["redRP"].tolist()
    gRedFactorList = df["redG"].tolist()

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
    #print(df[["GALEX_ID", "redG", "redBP", "redRP", "bpDered", "rpDered", "gDered", "bp_rp", "bp_rp_dered"]].head(10))

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

def gaiaStarCuts(df):
    print("Running initial cuts to result in GAIA star objects only")
    # from https://www.aanda.org/articles/aa/pdf/2018/08/aa32843-18.pdf section 2.1 and appendix B
    numLeft = len(df['bp_rp'].to_list())
    #print("--------------------------------- NUM Points to Start: " + str(numLeft) + " ---------------------------------")

    df = df.drop(df[df['parallax_over_error'] < 10].index) #select for parallax_over_error > 10
    numLeft = len(df['bp_rp'].to_list())
    #print("--------------------------------- NUM Points left after parallax_over_error cuts: " + str(numLeft) + " ---------------------------------")
    # By requiring high precision in the photometric data, this filter excludes objects with poor or 
    # unreliable photometric measurements, which are often non-stellar objects or stars in crowded fields 
    # where measurements are difficult.

    df = df.drop(df[df['phot_g_mean_flux_over_error'] < 50].index) #select for phot_g_mean_flux_over_error > 50 to remove variable stars (Gaia 2018)
    df = df.drop(df[df['phot_rp_mean_flux_over_error'] < 20].index) #select for phot_X_mean_flux_over_error > 20 to remove variable stars (Gaia 2018)
    df = df.drop(df[df['phot_bp_mean_flux_over_error'] < 20].index) #select for phot_X_mean_flux_over_error > 20 to remove variable stars (Gaia 2018)
    numLeft = len(df['bp_rp'].to_list())
    #print("--------------------------------- NUM Points left after phot_X_mean_flux_over_error cuts: " + str(numLeft) + " ---------------------------------")
    # This filter helps exclude objects with problematic or contaminated photometric measurements, which are often not single stars.
    
    photExcessLst = np.empty(0)

    photExcess = df['phot_bp_rp_excess_factor'].to_list()
    bpMeanMag = df['phot_bp_mean_mag'].to_list()
    rpMeanMag = df['phot_rp_mean_mag'].to_list()
    
    # want to keep phot_bp_rp_excess_factor < 1.3 + 0.06 * power(phot_bp_mean_mag - phot_rp_mean_mag, 2) and
    # phot_bp_rp_excess_factor > 1.0 + 0.015 * power(phot_bp_mean_mag - phot_rp_mean_mag, 2)
    # These criteria help to filter out objects with problematic BP and RP fluxes, which might otherwise introduce errors in color-magnitude diagrams.
    for i in range(len(photExcess)):
        if (1.3 + 0.06 * np.power(bpMeanMag[i] - rpMeanMag[i], 2) < photExcess[i]) and (1.0 + 0.015 * np.power(bpMeanMag[i] - rpMeanMag[i], 2) > photExcess[i]):
            photExcessLst = np.append(photExcessLst, "Keep")
        else:
            photExcessLst = np.append(photExcessLst, "Discard")

    df["photExcessCutBool"] = photExcessLst
   
    df = df.drop(df[df['photExcessCutBool'] == 'Discard'].index)

    numLeft = len(df['bp_rp'].to_list())
    print("--------------------------------- NUM Points left after phot_bp_rp_excess_factor cuts: " + str(numLeft) + " ---------------------------------")
    #print("--------------- NUM Points left after ALL cuts: " + str(numLeft) + " ---------------")
    # AND visibility_periods_used > 8 #Ensures that the star has been observed sufficiently frequently to provide reliable data (??? should I use)
    # AND astrometric_chi2_al / (astrometric_n_good_obs_al - 5) < 1.44 * greatest(1, exp(-0.4 * (phot_g_mean_mag - 19.5))) # Helps to filter out objects with poor astrometric fits. (???)
    return df

def lowExtinctionCuts(df):
    df = df.drop(df[df['e_bv'] > 0.015].index) #select for low extinction objects
    numLeft = len(df['e_bv'].to_list())
    print("--------------------------------- NUM Points left after low extinction (EBV) cuts: " + str(numLeft) + " ---------------------------------")
    return df

def saveDFasCSV(df, csvFileName):
    df.to_csv(csvFileName, header = True)
    print("Saved DF to CSV file")

def stringToFloatRed(df):
    print("Turning all 'string type' data values to floats after reddenning calculations")

    #takes all columns that we need as floats but they are strings and turns them into floats
    df['redG'] = df["redG"].apply(lambda x: float(x))
    df['redBP'] = df["redBP"].apply(lambda x: float(x))
    df['redRP'] = df["redRP"].apply(lambda x: float(x))
    df['bpDered'] = df["bpDered"].apply(lambda x: float(x))
    df['rpDered'] = df["rpDered"].apply(lambda x: float(x))
    df['gDered'] = df["gDered"].apply(lambda x: float(x))
    df['NUVDered'] = df["NUVDered"].apply(lambda x: float(x))
    df['bp_rp_dered'] = df["bp_rp_dered"].apply(lambda x: float(x))

def getKnownUMPs():
    
    def getEBV(df):
        
        gaiaDR2IDLst = df['GaiaDR2_ID'].to_list()
        
        ebvLst = np.empty(0)
        for gaiaObjID in gaiaDR2IDLst:
            
            customSimbad = Simbad()
            customSimbad.add_votable_fields('ra','dec')
            temp = customSimbad.query_object(f"Gaia DR2 {gaiaObjID}")
            if temp is None:
                ebvLst = np.append(ebvLst, "")
            else:
                ra = temp['RA'][0]
                dec = temp['DEC'][0]

            ebvLst = np.append(ebvLst, get_ebv_irsa(ra, dec))

        ebvLst = [float(i) for i in ebvLst]
        df['EBV'] = ebvLst

    def reddeningCol(df):
        bprpList = [float(i) for i in df["BP_RP"].tolist()]
        ebvList = df["EBV"].tolist()


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

    def insertAbsMagCol(df):
        absMag = np.empty(0)
        absMagDered = np.empty(0)
        gMag = [float(i) for i in df['G'].to_list()]
        gMagDered = [float(i) for i in df['gDered'].to_list()]
        gaiaDR2IDLst = df['GaiaDR2_ID'].to_list()
        
        parallax = np.empty(0)

        for gaiaObjID in gaiaDR2IDLst:
            customSimbad = Simbad()
            customSimbad.add_votable_fields('parallax')
            temp = customSimbad.query_object(f"Gaia DR2 {gaiaObjID}")
            if temp is None:
                parallax = np.append(parallax, "")
            else:
                parallax = np.append(parallax, temp["PLX_VALUE"][0])
        
        parallax = [i/1000 for i in parallax]  #SIMBAD reports in milliarcsec, so convert to arcsec (CHECK IF IT IS)
        df['parallax'] = parallax

        iterations = len(gMag)
        
        for i in tqdm(range(iterations), desc="Calculating Absolute G_Magnitude for Known UMP stars...", unit="iteration"):
            dist = 1/parallax[i]
            mag = gMag[i] - (5*np.log10(dist)) + 5
            magDered = gMagDered[i] - (5*np.log10(dist)) + 5
            absMag = np.append(absMag, mag)
            absMagDered = np.append(absMagDered, magDered)
        
        df['absMag'] = absMag
        df['absMagDered'] = absMagDered

    def deredden(df):
        bpList = [float(i) for i in df["BP"].tolist()]
        rpList = [float(i) for i in df["RP"].tolist()]
        gList = [float(i) for i in df["G"].tolist()]
        nuvList = [float(i) for i in df["NUV"].tolist()]
        ebvList = [float(i) for i in df["EBV"].tolist()]

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

    def get_ebv_irsa(ra, dec):
        coords = SkyCoord(ra + dec, unit=(u.hourangle, u.deg), frame='icrs')
        table = IrsaDust.get_query_table(coords, section='ebv')
        ebv = table['ext SFD mean'][0]  # Extract the E(B-V) value from the table
        return ebv

    knownUMPdf = dfFromTxtFile('knownUMPs.csv')
    knownUMPdf = knownUMPdf.drop(knownUMPdf[knownUMPdf['NUV'] == ''].index)

    getEBV(knownUMPdf)
    reddeningCol(knownUMPdf)
    deredden(knownUMPdf)
    insertAbsMagCol(knownUMPdf)

    knownUMPdf = knownUMPdf.drop(knownUMPdf[knownUMPdf['parallax'] == 0].index)

    saveDFasCSV(knownUMPdf, 'knownUMPUpdatedDF.csv')
    
def grabGaiaDR2FeHFromVizier(df):
    ## gets metallicities from vizier using gaia_id and inserts the list into the df

    ### VIZIER DOES NOT HAVE METALLICITIES SO WHAT SHOULD I DO?
    ## ['RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS', 'Source', 'Plx', 'e_Plx', 'pmRA', 
        # 'e_pmRA', 'pmDE', 'e_pmDE', 'Dup', 'FG', 'e_FG', 'Gmag', 'e_Gmag', 'FBP', 'e_FBP', 
        # 'BPmag', 'e_BPmag', 'FRP', 'e_FRP', 'RPmag', 'e_RPmag', 'BP-RP', 'RV', 'e_RV', 'Teff', 
        # 'AG', 'E_BP-RP_', 'Rad', 'Lum']

    Vizier.ROW_LIMIT = -1  # To fetch all rows without limit

    # Gaia DR2 catalog ID
    catalog = "I/345/gaia2"

    # List of Gaia DR2 IDs you want to query
    gaia_ids = df["Gaia_ID"].to_list()

    # Create a DataFrame to store results
    metallicities = np.empty(0)
    
    # Query Vizier for each Gaia ID
    iterations = len(gaia_ids)
    
    for i in tqdm(range(iterations), desc="Grabbing Gaia DR2 Metallicities from Vizier...", unit="iteration"):
        gaia_id = gaia_ids[i]
        result = Vizier.query_constraints(catalog=catalog, Source=gaia_id)
        
        if result:
            print("RESULT", list(result[0].columns)) ### VIZIER DOESNT HAVE METALLICITIES...
            #metallicities = np.append(metallicities, result)

    df["Fe_H_GaiaDR2"] = metallicities
    print(df[["Gaia_ID", "Fe_H_GaiaDR2"]].head(10))

def crossmatchGaiaDr2n3Ids(df):
    ## maybe try third party surveys like below:
        ## GALAH survey
        ## LAMOST DR6 Catalog

    ## double check if photometry is the same for both
    gaia_dr2_ids = df["Gaia_ID"].to_list()
    
    crossmatch_query = f"""
    SELECT dr2.source_id, dr3.*
    FROM gaiadr2.gaia_source AS dr2
    JOIN gaiaedr3.dr2_neighbourhood AS dr3
        ON dr2.source_id = dr3.dr2_source_id
    WHERE dr2.source_id IN ({','.join(map(str, gaia_dr2_ids))})
    ORDER BY dr2.source_id ASC
    """

    crossmatch_job = Gaia.launch_job_async(crossmatch_query)
    crossmatch_table = crossmatch_job.get_results().to_pandas()
    return crossmatch_table

def getGaiaDR3MetallicitiesGSPPhot(df):
    
    crossmatchDF = crossmatchGaiaDr2n3Ids(df)
    dr3IDs = list(crossmatchDF["dr3_source_id"])
    dr3_ids_string = ', '.join([f"'{dr3_id}'" for dr3_id in dr3IDs])

    # Query to fetch metallicity data from the Gaia DR3 RVS table
    query = f"""
            SELECT source_id AS dr3_source_id, teff_gspphot, logg_gspphot, mh_gspphot
            FROM gaiadr3.gaia_source
            WHERE source_id IN ({dr3_ids_string})
            """
    try:
        # Launch the query asynchronously
        job = Gaia.launch_job_async(query)
        result = job.get_results()
        # Convert the result to a Pandas DataFrame
        metallicity_df = result.to_pandas()
        #print(metallicity_df)
        #plt.hist(metallicity_df["mh_gspphot"], 25)
        #plt.xticks(np.arange(-4,1, 5/25))
        #plt.show()
    except Exception as e:
        print(f"Error occurred: {e}")
        return df  # Return original df if error happens

    fullCrossmatchMhDF = pd.merge(crossmatchDF, metallicity_df, on='dr3_source_id', how='left')
    fullCrossmatchMhDF['dr2_source_id'] = fullCrossmatchMhDF['dr2_source_id'].astype(str)
    df['Gaia_ID'] = df['Gaia_ID'].astype(str)

    df = pd.merge(df, fullCrossmatchMhDF, how = "left", left_on='Gaia_ID', right_on='dr2_source_id')
    return df    

def getGaiaDR3MetallicitiesGSPSpec(df):
    ## got col names from https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html
    crossmatchDF = crossmatchGaiaDr2n3Ids(df)
    dr3IDs = list(crossmatchDF["dr3_source_id"])
    dr3_ids_string = ', '.join([f"'{dr3_id}'" for dr3_id in dr3IDs])

    # Query to fetch metallicity data from the Gaia DR3 RVS table
    query = f"""
            SELECT source_id AS dr3_source_id, teff_gspspec, logg_gspspec, mh_gspspec
            FROM gaiadr3.astrophysical_parameters
            WHERE source_id IN ({dr3_ids_string})
            """
    try:
        # Launch the query asynchronously
        job = Gaia.launch_job_async(query)
        result = job.get_results()
        # Convert the result to a Pandas DataFrame
        metallicity_df = result.to_pandas()
        #plt.hist(metallicity_df["mh_gspphot"], 25)
        #plt.xticks(np.arange(-4,1, 5/25))
        #plt.show()
    except Exception as e:
        print(f"Error occurred: {e}")
        return df  # Return original df if error happens

    fullCrossmatchMhDF = pd.merge(crossmatchDF, metallicity_df, on='dr3_source_id', how='left')
    fullCrossmatchMhDF['dr2_source_id'] = fullCrossmatchMhDF['dr2_source_id'].astype(str)
    df['Gaia_ID'] = df['Gaia_ID'].astype(str)

    df = pd.merge(df, fullCrossmatchMhDF, how = "left", left_on='Gaia_ID', right_on='dr2_source_id')
    return df    

def turnStr2Float(df):  
    df['mh_gspphot'] = [float(i) for i in df['mh_gspphot'].to_list()]
    df['teff_gspspec'] = [float(i) for i in df['teff_gspspec'].to_list()]
    df['colorColorY'] = [float(i) for i in df['colorColorY'].to_list()]

    return df

def colorColorShiftYAxisDown(df):
    #takes columns and adds a new column with the y axis for colorcolor plot shifted to be horiz
    bprpLst = df['bp_rp_dered'].to_list()
    nuvList = df['NUVDered'].tolist()
    gList = df['gDered'].tolist()
    metalConst = 2.731 #was 4.731
    shiftVal = 1.366
    yaxisList = [(nuvList[i] - gList[i]) - ((bprpLst[i] * metalConst) + shiftVal) for i in range(len(gList))]
    df['colorColorY'] = yaxisList
    return df

def teffCut(df):

    # teff cut so keep anything in the range of 3800-7500
    df = df.drop(df[df['teff_gspspec'] > 7500].index) #select for anything below teff 7500
    df = df.drop(df[df['teff_gspspec'] < 3900].index) #select for anything above teff 3800

    return df

def printTime():
    dt = datetime.now()
    # convert timestamp to string in dd-mm-yyyy HH:MM:SS
    str_date_time = dt.strftime("%d-%m-%Y, %H:%M:%S")
    print("It is:", str_date_time)

if __name__ == '__main__':

    path = 'C:/Users/pchar/Desktop/ncsuRoedererResearch24-25/fullDatafiles/'
    files = os.listdir(path)
    sfileNameList = [file for file in files if file[-5] == 'S']
    sfileNameList = sorted(sfileNameList)
    nfileNameList = [file for file in files if file[-5] == 'N']
    nfileNameList = sorted(nfileNameList)
    fileNameList = np.append(sfileNameList, nfileNameList)
    
    fnameNum = len(fileNameList)

    southFNameLst1 = fileNameList[0:int(0.25*fnameNum)]
    southFNameLst2 = fileNameList[int(0.25*fnameNum):int(0.5*fnameNum)]
    northFNameLst1 = fileNameList[int(0.5*fnameNum):int(0.75*fnameNum)]
    northFNameLst2 = fileNameList[int(0.75*fnameNum):fnameNum]

    ## shown here below is just a few files at certain galactic latitudes (in north N), but this was
    ##      done for all data captured by the Gaia x GALEX crossmatch in Bianchi & Shiao 2020 paper

    testFileNameList = ['80-85csv_GUVCat_AIS055_GaiaDR2_3arcsecAllColumns_galex_glat80_00N__85_00N.csv_1' ,
                    '80-85csv_GUVCat_AIS055_GaiaDR2_3arcsecAllColumns_galex_glat80_00N__85_00N.csv_2' ,
                    '80-85csv_GUVCat_AIS055_GaiaDR2_3arcsecAllColumns_galex_glat80_00N__85_00N.csv_3' ,
                    '85-90csv_GUVCat_AIS055_GaiaDR2_3arcsecAllColumns_galex_glat85_00N__90_00N.csv_1']

    #fullFileNameLstofLists = [testFileNameList, testFileNameList]
    fullFileNameLstofLists = [fileNameList, northFNameLst2]

    galexDataDF = pd.DataFrame()
    dfs = []

    for fname in fullFileNameLstofLists[0]:
        galexCsvDFTemp = dfFromTxtFile("C:/Users/pchar/Desktop/ncsuRoedererResearch24-25/datafiles/" + fname) ## FOR LOCAL
        #galexCsvDFTemp = dfFromTxtFile("/share/mpstars/pcharvu/datafiles/" + fname) ## FOR HAZEL

        underscore_index = fname.rfind('_')
        n = len(fname) - underscore_index - 1
        print("----------------------------------------------------- Starting on " + fname[0:5] + fname[-n-1:] + " galactic latitude file -----------------------------------------------------")

        galexDataDFTemp = getRidOfObjectsWithBlanks(galexCsvDFTemp)
        stringToFloat(galexDataDFTemp)
        galexDataDFTemp = gaiaStarCuts(galexDataDFTemp)
        insertReddeningCol(galexDataDFTemp) #calculates ratio of extinction in x band (r_x) wrt A0 (k_x = Ax/A0)
        deredden(galexDataDFTemp)
        stringToFloatRed(galexDataDFTemp)
        insertAbsMagCol(galexDataDFTemp)

        printTime()
        print("Getting PHOTometric metallicities and crossmatching DR2 with DR3")
        galexDataDFTemp = getGaiaDR3MetallicitiesGSPPhot(galexDataDFTemp)

        printTime()
        print("Getting SPECtroscopic metallicities and crossmatching DR2 with DR3")
        galexDataDFTemp = getGaiaDR3MetallicitiesGSPSpec(galexDataDFTemp)
        
        printTime()
        print("Calculating y-axis values for color-color plot")
        galexDataDFTemp = colorColorShiftYAxisDown(galexDataDFTemp)

        print("Done with " + fname[0:5] + fname[-n-1:] + " galactic latitude file")
        dfs.append(galexDataDFTemp)

    print("Done with all files. Concatenating dataframes")#-----------------------------------------------------------------------------------
    printTime()
    galexDataDF = pd.concat(dfs, ignore_index=True)
    print("Done concatenating dataframes")#-----------------------------------------------------------------------------------
    printTime()
    
    galexDataDF = getRidOfObjectsWithBlanksPost(galexDataDF)
    galexDataDF = turnStr2Float(galexDataDF)

    saveDFasCSV(galexDataDF, "cutAndCleanedGaiaGALEXData.csv")
    print(f"THERE ARE {len(galexDataDF['e_bv'].to_list())} OBJECTS TOTAL")

    galexDataDF = teffCut(galexDataDF)
    saveDFasCSV(galexDataDF, "cutAndCleanedPostTeffCutData.csv")
    print(f"THERE ARE {len(galexDataDF['e_bv'].to_list())} OBJECTS TOTAL AFTER Teff CUTS")

    galexDataDF = lowExtinctionCuts(galexDataDF)
    saveDFasCSV(galexDataDF, "lowExtinctionGALEXGaiaData.csv")
    print(f"THERE ARE {len(galexDataDF['e_bv'].to_list())} LOW EXTINCTION and Teff Cut OBJECTS")
    #plt.hist(galexDataDF['e_bv'].to_list(), 20)
    #plt.show()

    getKnownUMPs()
   
    ### NOTES: --------------------------------------------------------------------------------------------------------------------------------------------
