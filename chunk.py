import xarray as xr
import numpy as np

#Add in path loading based on data_access
file = TODO #"drive/MyDrive/CSKS4_GEC_B_WR_01_VV_RA_FF_20191126085014_20191126085029.MBI.tif.zarr"
data_array = xr.open_dataset(file, engine="zarr")

def process_chunk(chunk):
  pass

def pyramid(data_array, levels=[500, 250, 100]):
  for chunksize in levels:
    data = data_array.chunk({'y':chunksize,'x':chunksize, 'band':1}).chunks
    ix = data['x']
    iy = data['y']

    idx = [sum(ix[:i]) for i in range(len(ix)+1)]
    idy = [sum(iy[:i]) for i in range(len(iy)+1)]

    for i in range(0, len(idx)-1, 1):
      tgt_x = xr.DataArray(np.arange(idx[i], idx[i+1]), dims="x")
      for j in range(0, len(idy)-1, 1):
        #print(f"{idx[i]}, {idy[j]}")
        tgt_y = xr.DataArray(np.arange(idy[j], idy[j+1]), dims="y")
        chunk = data_array.isel(y=tgt_y, x=tgt_x, band=0)
        do_something = process_chunk(chunk)
        #print(da)

pyramid(data_array)
