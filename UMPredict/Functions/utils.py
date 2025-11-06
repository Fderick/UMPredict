import pandas as pd
import numpy as np
import os



def get_folders_and_files(root_dir):
    folders = []
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folders.extend([os.path.join(dirpath, d) for d in dirnames])
        files.extend([os.path.join(dirpath, f) for f in filenames])
    return folders, files


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
    data_folder = os.path.normpath('UMPredict\\Data')
    
    _ , files = os.get_folders_and_files(data_folder)
    
    #bonifacio_data = os.path.normpath('UMPredict\Data\gaiaGalexBonifacioDatForPlotting.csv')
    #laidat_data = os.path.normpath('UMPredict\Data\gaiaGalexLaiDatForPlotting.csv')
    #liaoki_data = os.path.normpath('UMPredict\Data\gaiaGalexLiAokiDatForPlotting.csv')
    #yong_data = os.path.normpath('UMPredict\Data\gaiaGalexYongDatForPlotting.csv')
    
    tables = []
    for file in files:
        with pd.read_csv(file) as df:
            tables.append(df)
    
    return tables
    
    
def extract_data():
    
    tables = compile_data()
    
    pass


    #Total UMP