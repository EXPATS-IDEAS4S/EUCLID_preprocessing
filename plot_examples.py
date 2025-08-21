# TODO: plot examples for 24th July 2023 and compare to MSG
# %%
import os
import glob
import numpy as np
import sys
import xarray as xr 
sys.path.append("/home/pbigalke/Documents/Code/data-investigation/data-investigation/")

import plotting.plot_MSG as msg
import plotting.plot_EUCLID as eu
import helpers.datetime_helper as hlp
from plotting.gif_maker import gif_maker, convert_gifs_to_mp4

# %%
def make_msg_plots(example_msg, plot_path_msg, channelname, vmin, vmax):
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
                    vmin=vmin, vmax=vmax, alpha=0.8, clear_sky_threshold=vmax,
                    title=title, path_out=f"{plot_path_msg}/msg_{dt}.png")

# %%
def make_euclid_plots(example_euclid, plot_path_euclid, plot_path_euclid_msg):
    # read msg data
    data_euclid = xr.open_dataset(example_euclid)
    # loop over timestamps
    for timestamp in data_euclid.time.values[::3]:
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
                                title=title, path_out=f"{plot_path_euclid_msg}/euclid_regrid_{dt}.png")

def plot_euclid_over_msg(example_euclid, example_msg, plot_path, channelname, vmin_msg, vmax_msg, 
                         cmap_euclid, vmin_euclid, vmax_euclid, alpha_euclid):
    # read euclid data
    data_euclid = xr.open_dataset(example_euclid)
    # resample and sum over 15 min intervals
    data_euclid = data_euclid.resample(time='15T').sum(dim='time')

    # read msg data
    data_msg = xr.open_dataset(example_msg)

    # loop over timestamps
    for timestamp in data_euclid.time.values:
        print(timestamp)
        dt = hlp.get_datetimestring_from_npdatetime(timestamp)

        # get msg data for this timestamp
        msg_lons = data_msg.sel(time=timestamp).lon.values
        msg_lats = data_msg.sel(time=timestamp).lat.values
        msg_tb = data_msg.sel(time=timestamp).IR_108.values

        # get euclid data for this timestamp
        euclid_lons = data_euclid.sel(time=timestamp).euclid_lon.values
        euclid_lats = data_euclid.sel(time=timestamp).euclid_lat.values
        euclid_data = data_euclid.sel(time=timestamp).euclid.values

        # plot euclid
        title = f'EUCLID & MSG - {dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[-4:-2]}:{dt[-2:]}'
        os.makedirs(f"{plot_path}/original", exist_ok=True)
        eu.plot_euclid_over_MSG(msg_lons, msg_lats, msg_tb, channelname, 
                            euclid_lons=euclid_lons, euclid_lats=euclid_lats, euclid_data=euclid_data, 
                            vmin_msg=vmin_msg, vmax_msg=vmax_msg, 
                            cmap_euclid=cmap_euclid, vmin_euclid=vmin_euclid, vmax_euclid=vmax_euclid, alpha_euclid=alpha_euclid,
                            title=title, path_out=f"{plot_path}/original/euclid_over_msg_{dt}.png",)

        
        # get regridded data for this timestamp
        euclid_regrid = data_euclid.sel(time=timestamp).euclid_msg_grid.values

        # plot regridded euclid
        title = f'EUCLID regridded & MSG - {dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[-4:-2]}:{dt[-2:]}'  
        os.makedirs(f"{plot_path}/regrid", exist_ok=True)
        eu.plot_euclid_over_MSG(msg_lons, msg_lats, msg_tb, channelname, 
                            euclid_lons=msg_lons, euclid_lats=msg_lats, euclid_data=euclid_regrid, 
                            vmin_msg=vmin_msg, vmax_msg=vmax_msg, 
                            cmap_euclid=cmap_euclid, vmin_euclid=vmin_euclid, vmax_euclid=vmax_euclid, alpha_euclid=alpha_euclid,
                            title=title, path_out=f"{plot_path}/regrid/euclid_regrid_over_msg_{dt}.png",)
    

# %%
# define channels to plot
plot_path = "/home/pbigalke/Documents/Code/Repos/EUCLID_preprocessing/plots"
channelname = "IR_108"
min_val = 200
max_val = 300

cmap_euclid = 'RdPu'
vmin_euclid = 1
vmax_euclid = 20
alpha_euclid = 1.0

# %%
example_msg = "/home/pbigalke/Documents/Code/Repos/EUCLID_preprocessing/example_MSG_file/20230724-EXPATS-RG.nc"
plot_path_msg = f"{plot_path}/msg"
os.makedirs(plot_path_msg, exist_ok=True)
# make_msg_plots(example_msg, plot_path_msg, channelname=channelname, vmin=min_val, vmax=max_val)

example_euclid = "/net/merisi/pbigalke/data/EUCLID/TESTING/2023/07/EUCLID_total_lightning_20230724.nc"
plot_path_euclid = f"{plot_path}/euclid/original"
plot_path_euclid_regrid = f"{plot_path}/euclid/regrid"
plot_path_euclid_msg = f"{plot_path}/euclid/msg"
os.makedirs(plot_path_euclid, exist_ok=True)
os.makedirs(plot_path_euclid_regrid, exist_ok=True)
os.makedirs(plot_path_euclid_msg, exist_ok=True)
# make_euclid_plots(example_euclid, plot_path_euclid, plot_path_euclid_regrid)
dat = plot_euclid_over_msg(example_euclid, example_msg, plot_path_euclid_msg, 
                           channelname=channelname, vmin_msg=min_val, vmax_msg=max_val, 
                           cmap_euclid=cmap_euclid, vmin_euclid=vmin_euclid, vmax_euclid=vmax_euclid, alpha_euclid=alpha_euclid)

# %%
# make the folders into Gifs
# msg_imgs = sorted(glob.glob(os.path.join(plot_path_msg, '*.png')))
# print(f"making a gif from {len(msg_imgs)} MSG images.")
# gif_maker(msg_imgs, f'20230724_msg', plot_path, sec_per_frame=1)

# eu_imgs = sorted(glob.glob(os.path.join(plot_path_euclid, '*.png')))
# print(f"making a gif from {len(eu_imgs)} EUCLID images.")
# gif_maker(eu_imgs, f'20230724_euclid', plot_path, sec_per_frame=1)

# eu_regrid_imgs = sorted(glob.glob(os.path.join(plot_path_euclid_regrid, '*.png')))
# print(f"making a gif from {len(eu_regrid_imgs)} EUCLID regridded images.")
# gif_maker(eu_regrid_imgs, f'20230724_euclid_regrid', plot_path, sec_per_frame=1)

eu_msg_imgs = sorted(glob.glob(os.path.join(plot_path_euclid_msg, 'original/*.png')))
print(f"making a gif from {len(eu_msg_imgs)} EUCLID and MSG images.")
gif_maker(eu_msg_imgs, f'20230724_euclid_msg', plot_path, sec_per_frame=5)

eu_msg_imgs = sorted(glob.glob(os.path.join(plot_path_euclid_msg, 'regrid/*.png')))
print(f"making a gif from {len(eu_msg_imgs)} EUCLID and MSG images.")
gif_maker(eu_msg_imgs, f'20230724_euclid_regrid_msg', plot_path, sec_per_frame=5)

# %%
# convert gifs to mp4
convert_gifs_to_mp4(plot_path, plot_path)
# %%
