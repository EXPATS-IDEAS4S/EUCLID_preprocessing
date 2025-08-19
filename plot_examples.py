# TODO: plot examples for 24th July 2023 and compare to MSG
# %%
import os
import sys
import xarray as xr 
sys.path.append("/home/pbigalke/Documents/Code/data-investigation/data-investigation/")

import plotting.plot_MSG as msg
import plotting.plot_EUCLID as eu
import helpers.datetime_helper as hlp

# %%
# define channels to plot
channelname = "IR_108"
min_val = 200
max_val = 300

example_msg = "/home/pbigalke/Documents/Code/Repos/EUCLID_preprocessing/example_MSG_file/20230724-EXPATS-RG.nc"


# %%
plot_path_msg = f"/home/pbigalke/Documents/Code/Repos/EUCLID_preprocessing/plots/msg"
os.makedirs(plot_path_msg, exist_ok=True)

# read msg data
data_msg = xr.open_dataset(example_msg)
# loop over timestamps
for timestamp in data_msg.time.values:
    print(timestamp)
    dt = hlp.get_datetimestring_from_npdatetime(timestamp)

    # get msg data for this timestamp
    msg_lons = data_msg.sel(time=timestamp).lon.values
    msg_lats = data_msg.sel(time=timestamp).lat.values
    msg_tb = data_msg.sel(time=timestamp).IR_108.values
    
    # plot msg
    title = f'MSG {channelname} - {dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[-4:-2]}:{dt[-2:]}'
    msg.plot_MSG_over_map(msg_lons, msg_lats, msg_tb, channelname, 
                vmin=min_val, vmax=max_val, alpha=0.8, clear_sky_thresh=max_val,
                title=title, path_out=f"{plot_path_msg}/msg_{dt}.png")

# %%
example_euclid = "/net/merisi/pbigalke/data/EUCLID/TESTING/2023/07/EUCLID_total_lightning_20230724.nc"

plot_path_euclid = f"/home/pbigalke/Documents/Code/Repos/EUCLID_preprocessing/plots/euclid/original"
plot_path_euclid_msg = f"/home/pbigalke/Documents/Code/Repos/EUCLID_preprocessing/plots/euclid/regrid"
os.makedirs(plot_path_euclid, exist_ok=True)

# read msg data
data_euclid = xr.open_dataset(example_euclid)
# loop over timestamps
for timestamp in data_euclid.time.values[:1]:
    print(timestamp)
    dt = hlp.get_datetimestring_from_npdatetime(timestamp)

    # get euclid data for this timestamp
    euclid_lons = data_euclid.sel(time=timestamp).euclid_lon.values
    euclid_lats = data_euclid.sel(time=timestamp).euclid_lat.values
    euclid_data = data_euclid.sel(time=timestamp).euclid.values

    # plot euclid
    title = f'EUCLID - {dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[-4:-2]}:{dt[-2:]}'
    eu.plot_euclid_over_map(euclid_lons, euclid_lats, euclid_data,  
                            title=title, path_out=f"{plot_path_euclid}/euclid_{dt}.png")
    
    # get regridded data for this timestamp
    msg_lons = data_euclid.sel(time=timestamp).lon.values
    msg_lats = data_euclid.sel(time=timestamp).lat.values
    euclid_regrid = data_euclid.sel(time=timestamp).euclid_msg_grid.values

    # plot regridded euclid
    title = f'EUCLID regridded - {dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[-4:-2]}:{dt[-2:]}'
    eu.plot_euclid_over_map(msg_lons, msg_lats, euclid_regrid,  
                            title=title, path_out=f"{plot_path_euclid}/euclid_regrid_{dt}.png")
    
# %%
