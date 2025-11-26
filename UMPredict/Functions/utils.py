import pandas as pd
import numpy as np
import os
import re



def get_folders_and_files(root_dir):
    '''
    Given a root directory. Extract all of the files and folders in the root directory (not recursive).
    
    Parameters:
     - root_dir (str): String with the path to the folder to inspect.
    
    Returns:
     - folders (list): List of strings with all folder names (relative to root_dir).
     - files (list): List of strings with all file names (relative to root_dir).
    '''
    
    folders = []
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folders.extend([os.path.join(dirpath, d) for d in dirnames])
        files.extend([os.path.join(dirpath, f) for f in filenames])
    return folders, files

def extract_target_name(filename):
    match = re.search(r'gaiaGalex(.*?)Dat', filename)
    return match.group(1) if match else None



def compile_data():
    '''
    Function to obtain the data tables from the 'Data' folder.
    Runs through all files in the folder and creates a pandas dataframe list.
    Assumes all 'Data' files are csv.
    
    Parameters:
     -
    Returns:
     - tables: List of pandas dataframe for every file in Data.
    
    '''
    data_folder = os.path.normpath(r'Data\Train_Data')
    
    _ , files = get_folders_and_files(data_folder)
    
    #bonifacio_data = os.path.normpath('UMPredict\Data\gaiaGalexBonifacioDatForPlotting.csv')
    #laidat_data = os.path.normpath('UMPredict\Data\gaiaGalexLaiDatForPlotting.csv')
    #liaoki_data = os.path.normpath('UMPredict\Data\gaiaGalexLiAokiDatForPlotting.csv')
    #yong_data = os.path.normpath('UMPredict\Data\gaiaGalexYongDatForPlotting.csv')
    
    tables = []
    for file in files:
        df = pd.read_csv(file) 
        survey_name = extract_target_name(file)
        df['Survey'] = survey_name
        tables.append(df)

    #print(tables)
    return tables
    
    
def extract_data(UMP_threshold = -4.0):
    
    tables = compile_data()
    
    #FeH NUVDered colorColorY bp_rp_dered Teff logg
    columns_of_interest = ['Survey', 'FeH', 'NUVDered', 'bp_rp_dered','e_bv', 'Teff', 'logg', 'colorColorY']
    
    for i, df in enumerate(tables):

        #Ensure all columns exists and if not, create a nan array.
        for col in columns_of_interest:
            if col not in df.columns:
                df[col] = np.nan
        
        #Select Columns of interest. 
        tables[i] = df[columns_of_interest]
      
    compiled_data = pd.concat(tables, ignore_index=True)

    #Now check for UMP Threshold
    compiled_data['UMP_flag'] = 0
    
    for index, row in compiled_data.iterrows():
        if row['FeH'] <= UMP_threshold:
            compiled_data.at[index, 'UMP_flag'] = 1
        else:
            compiled_data.at[index, 'UMP_flag'] = 0

    return compiled_data


def read_gaia_galex_data():
    data = pd.read_csv(r"Data\Test_Data\crossmatched_GalexGaia.csv")
    columns_of_interest = ['Gaia_ID', 'NUVDered', 'bp_rp_dered','e_bv' ,'Teff', 'logg', 'colorColorY']    
    return data[columns_of_interest]
    
    
