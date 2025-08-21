# %%
import numpy as np
import xarray as xr
import glob
import tarfile
import os
import shutil
import pandas as pd
import scipy.ndimage
import datetime

# Define the edges of the Grid (edges of outer most pixels)
LONMIN = 5.00
LONMAX = 16.00
LATMIN = 42.00
LATMAX = 51.52

# define width and height of pixels
LONSTEP = 0.04
LATSTEP = 0.04

# define pixel center coords
EUCLID_LONS = np.arange(5.00, 16.00, 0.04) + 0.02
EUCLID_LATS = np.arange(42.00, 51.50, 0.04) + 0.02

# time resolution of EUCLID data
TIME_RES = 5  # minutes

# number of lines in .dat file that contain general info
SKIPROWS = 4

# Grid settings of MSG data
MSG_LONS = np.arange(5.00, 16.01, 0.04).astype(np.float32)
MSG_LATS = np.arange(42.00, 51.53, 0.04).astype(np.float32)

# %%
def create_msg_like_daily_dataset(year, month, day):


    # create timeseries for 1 day, every 5 minutes and convert to 'datetime64[ns]'
    times = pd.date_range(start=f"{year}-{month:02d}-{day:02d} 00:00:00", 
                          end=f"{year}-{month:02d}-{day:02d} 23:59:00", 
                          freq=f"{TIME_RES}T").values.astype("datetime64[ns]")

    # create dataset containing only NaNs and add description to this variable
    ds = xr.Dataset(
        {
            "euclid": (("time", "euclid_lat", "euclid_lon"), np.full((len(times), len(EUCLID_LATS), len(EUCLID_LONS)), np.nan, dtype=np.float32)), 
            "euclid_msg_grid": (("time", "lat", "lon"), np.full((len(times), len(MSG_LATS), len(MSG_LONS)), np.nan, dtype=np.float32)), 
        },
        coords={
            "time": times,
            "lat": MSG_LATS,
            "lon": MSG_LONS,
            "euclid_lat": EUCLID_LATS,
            "euclid_lon": EUCLID_LONS,
        },
        attrs={
            "description": "EUCLID data over EXPATS domain with a spatial resolution of 0.04° (MSG).",
        }
    )

    # add attributes to variables
    ds.euclid.attrs.update({
        "long_name": "Total Lightning Count on EUCLID grid during given time interval",
        "units": "",
    })
    ds.euclid_msg_grid.attrs.update({
        "long_name": "Total Lightning Count regridded to MSG grid.",
        "description": "Each EUCLID lightning count is distributed evenly over the 4 neighboring MSG pixel. "
        "Treat the MSG pixel at the edges of the domain with caution as they only represent 1/2 (border pixel) "
        "or 1/4 (corner pixel) of the usual pixel size. "
        "To resample to 15 min MSG resolution run data_euclid.resample(time='15T').sum(dim='time').",
        "units": "",
    })

    # add description to coords
    ds.coords["time"].attrs.update({
        "description": "Starting time of counting interval. If the timestamp is e.g. 2023-07-24 00:00:00, "
        "this means the flashes were counted from 00:00:00 - 00:04:59.",
    })
    ds.coords["lat"].attrs.update({
        "long_name": "Latitude of MSG grid",
        "units": "degrees_north",
    })
    ds.coords["lon"].attrs.update({
        "long_name": "Longitude of MSG grid",
        "units": "degrees_east",
    })
    ds.coords["euclid_lat"].attrs.update({
        "long_name": "Latitude of EUCLID grid",
        "units": "degrees_north",
    })
    ds.coords["euclid_lon"].attrs.update({
        "long_name": "Longitude of EUCLID grid",
        "units": "degrees_east",
    })

    return ds

# %%
def process_euclid_data(path, years, months, days, output_path, delete_extracted=True):
    """
    Process EUCLID data from tar files and convert to NetCDF format.
    
    Parameters:
    - path: str, path to the EUCLID data directory
    - years: list of int, years to process
    - months: list of int, months to process
    - output_path: str, path to the output directory where to save the processed netcdf files
    - delete_extracted: bool, whether to delete extracted files after processing
    """
    # loop over years
    for year in years:

        # define year path
        year_path = f"{path}/Euclid_{year}.tar"
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
            if month in months and day in days:

                # path to extracted folder
                day_path_extracted = day_path.replace(".tar.gz", "")
                # untar day folder
                if not os.path.exists(day_path_extracted):
                    print(f"___unpacking {month:02d}-{day:02d}", flush=True)
                    # extract all files to the specified directory
                    with tarfile.open(day_path, "r:gz") as tar:
                        tar.extractall(path=day_path_extracted)
                else:
                    print(f"___already unpacked {month:02d}-{day:02d}", flush=True)

                # create empty dataset
                ds = create_msg_like_daily_dataset(year, month, day)

                # get all files in folder
                extracted_files = sorted(glob.glob(f"{day_path_extracted}/*.dat"))
                
                # loop over files
                for file in extracted_files:
                    # create array full of NaNs in expected shape
                    #data_timestamp = np.full((len(ds.lat), len(ds.lon)), np.nan)

                    # read lines of .dat file
                    with open(file, "r") as f:
                        for l, line in enumerate(f.readlines()):
                            if l == 0:
                                # extract timestamp from first line
                                timestamp = np.datetime64(line[:19])

                            if l >= SKIPROWS:
                                # first row is max latitude so we need to start with last index and decrease
                                lat_idx = len(ds.euclid_lat.values) - l + SKIPROWS - 1
                                # convert this line into array
                                ds["euclid"].loc[dict(time=timestamp)][lat_idx, :] = np.fromstring(line, sep=" ")

                    # regrid to MSG data and save in separate variable
                    # first add row of zeros at left and top of array
                    euclid_padded = np.pad(ds["euclid"].loc[dict(time=timestamp)].values, ((1, 0), (1, 0)), mode='constant', constant_values=0)
                    # convolve filter over array -> meaning to add 1/4 of each euclid pixel to each neighboring MSG pixel
                    k = np.array([[0.25,0.25],[0.25,0.25]])
                    ds["euclid_msg_grid"].loc[dict(time=timestamp)] = scipy.ndimage.convolve(euclid_padded, k, mode='constant', cval=0)


                # save daily netcdf file
                out_dir = f"{output_path}/{year}/{month:02d}"
                os.makedirs(out_dir, exist_ok=True)
                ds.to_netcdf(f"{out_dir}/EUCLID_total_lightning_{year}{month:02d}{day:02d}.nc")

                # delete extracted day folder recursively
                if delete_extracted:
                    print(f"___deleting folder {month}-{day}", flush=True)
                    shutil.rmtree(day_path_extracted)

        # delete extracted year folder
        if delete_extracted:
            print(f"deleting", year_path_extracted, flush=True)
            shutil.rmtree(year_path_extracted)


# %%
euclid_path = "/net/morget/dcorradi/EUCLID"
output_path = "/net/merisi/pbigalke/data/EUCLID"
years = [2023]
months = [7] # np.arange(4, 10, 1)
days = np.arange(1, 32, 1) # np.arange(1, 29, 1)

start_time = datetime.datetime.now()
ds = process_euclid_data(euclid_path, years, months, days, output_path=output_path, delete_extracted=False)
# print out runtime in hours
runtime = (datetime.datetime.now() - start_time).total_seconds()
print(f"Processing took {runtime / 3600} hours or {runtime / 60} minutes.", flush=True)

# %%
