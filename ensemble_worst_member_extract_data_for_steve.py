



import xarray as xr
from pylab import plt
import pandas as pd

from xarray_clim import standardize_dataset, sellonlatbox, weighted_areamean ,sellonlatpoint



var='167.128'
date='20190720_0000'
area=[-5,60,30,40]
leadtime=72 #h
perc = 95
# ipath='/data/ensemble_worst_member/'
ipath='/climstorage/sebastian/ensemble_worst_member/ecmwf_data/'

for date in pd.date_range('201907200000','201907221200', freq='12h'):
    date = date.strftime('%Y%m%d_%H00')
    ifile = f'{ipath}/ecmwf_ens_{date}_{var}_for_ens_worst_member.nc.nc'

    data = xr.open_dataset(ifile)
    data = standardize_dataset(data)
    data = data['t2m']
    data = data - 273.15


    lat = 50
    lon = 10

    fc = sellonlatpoint(data,lon=lon, lat=lat )

    fc.to_netcdf(f'ecmwf_enss_lat{lat}_lon{lon}_{date}.nc')

    df = fc.to_pandas()
    df.to_csv(f'ecmwf_enss_lat{lat}_lon{lon}_{date}.csv')


