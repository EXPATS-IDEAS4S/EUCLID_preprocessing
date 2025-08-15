# %%
import numpy as np
import xarray as xr
import glob
import tarfile
import os
import shutil
import pandas as pd

def get_euclid_grid_settings():
    """
    Returns the grid settings for the EUCLID dataset.
    
    Returns:
    - longitudes: numpy array of longitudes
    - latitudes: numpy array of latitudes
    """
    # Define the grid settings
    LonMin = 5.00
    LonMax = 15.00
    LonStep = 0.04

    LatMin = 42.00
    LatMax = 51.52
    LatStep = 0.04
    
    return LonMin, LonMax, LonStep, LatMin, LatMax, LatStep

def create_msg_like_daily_file(year, month, day):

    lons = np.arange(5.00, 16.01, 0.04).astype(np.float32)
    lats = np.arange(42.00, 51.53, 0.04).astype(np.float32)
    # create timeseries for 1 day, every 15 minutes and convert to 'datetime64[ns]'
    times = pd.date_range(start=f"{year}-{month:02d}-{day:02d}", end=f"{year}-{month:02d}-{day:02d} 23:59", freq="15T").values.astype("datetime64[ns]")

    # create dataset containing only NaNs
    ds = xr.Dataset(
        {
            "euclid": (("time", "lat", "lon"), np.full((len(times), len(lats), len(lons)), np.nan, dtype=np.float32)),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds

# %%
def process_euclid_data(path, years, months, delete_extracted=True):
    """
    Process EUCLID data from tar files and convert to NetCDF format.
    
    Parameters:
    - path: str, path to the EUCLID data directory
    - years: list of int, years to process
    - months: list of int, months to process
    - delete_extracted: bool, whether to delete extracted files after processing
    """
    # loop over years
    for year in years:

        # define year path
        year_path = f"{path}/TEST_Euclid_{year}.tar"
        year_path_extracted = year_path.replace(".tar", "")

        # unpack year folder if not already exists
        if not os.path.exists(year_path_extracted):
            print("unpacking", year_path, flush=True)
            # Extract all subfolders to the specified directory
            with tarfile.open(year_path, "r") as tar:
                tar.extractall(path=year_path_extracted)
        else:
            print("already unpacked", year_path, flush=True)

        # List all subfolders in the extracted directory
        extracted_folders = sorted(glob.glob(f"{year_path_extracted}/data/*"))
        
        # loop over files
        for day_path in extracted_folders:
            # get month and day from filename
            date_str = day_path.split("/")[-1].split("_")[-1].split(".")[0]
            month = int(date_str[4:6])
            day = int(date_str[6:])

            # check if month is in study period
            if month in months:

                # path to extracted folder
                day_path_extracted = day_path.replace(".tar.gz", "")
                # untar day folder
                if not os.path.exists(day_path_extracted):
                    print(f"unpacking {month:02d}-{day:02d}", flush=True)
                    # extract all files to the specified directory
                    with tarfile.open(day_path, "r:gz") as tar:
                        tar.extractall(path=day_path_extracted)
                else:
                    print(f"already unpacked {month:02d}-{day:02d}", flush=True)

                # create empty dataset
                ds = create_msg_like_daily_file(year, month, day)

                # get all files in folder
                extracted_files = sorted(glob.glob(f"{day_path_extracted}/*.dat"))
                
                # loop over files
                for file in extracted_files[:1]:
                    print("__converting", file)
                    # read data from file
                    df = pd.read_csv(file, sep=" ", skiprows=4, lineterminator="\n", skip_blank_lines=True)
                    print(df)

                    # TODO: save data at corresponding timestamp

                
                print(ds)
                        
                # delete extracted day folder recursively
                if delete_extracted:
                    print(f"_deleting folder {month}-{day}", flush=True)
                    shutil.rmtree(day_path_extracted)
                break

        # delete extracted year folder
        if delete_extracted:
            print(f"deleting", year_path_extracted, flush=True)
            shutil.rmtree(year_path_extracted)
        break

# %%
euclid_path = "/net/morget/dcorradi/EUCLID/TESTING"
years = [2013]
months = np.arange(4, 10, 1)

process_euclid_data(euclid_path, years, months, delete_extracted=False)

# %%
