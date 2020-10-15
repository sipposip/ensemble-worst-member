

import os
import matplotlib
matplotlib.use('agg')
import xarray as xr
from pylab import plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from scipy import stats
import seaborn as sns


from xarray_clim import standardize_dataset, sellonlatbox, weighted_areamean

plt.rcParams['savefig.bbox'] = 'tight'
plt.ioff()

if not os.path.exists('plots/'):
    os.mkdir('plots/')

var='167.128'

area=[-5,60,30,40]
# leadtime=72 #h
perc = 95
ipath='/climstorage/sebastian/ensemble_worst_member/ecmwf_data/'
# ipath='/data//ensemble_worst_member/'
# date=pd.to_datetime('201907200000')
leadtime = 72
for date in pd.date_range('201906011200','201908251200', freq='5d'):
    for leadtime in np.arange(6,24*6,6):
        datestr = date.strftime("%Y%m%d_%H%M")
        ifile = f'{ipath}/ecmwf_ens_{datestr}_{var}_for_ens_worst_member.nc.nc'


        data = xr.open_dataset(ifile)
        data = standardize_dataset(data)
        data = data['t2m']
        data = data.astype('float64')
        data = data - 273.15
        data = sellonlatbox(data, *area)

        # make sure that the data is indeed 6 hourly
        assert(np.all(pd.to_timedelta(data.time.diff('time')) == pd.to_timedelta('6h')))

        leadidx = leadtime//6

        fc = data.isel(time=leadidx)

        date_valid = pd.to_datetime(fc.time.values)
        date_init = pd.to_datetime(data.time[0].values)

        # the ensemble dimension is called "number
        ensmean = fc.mean('number')
        perc_empirical = fc.quantile(dim='number', q=perc/100)

        # 95th via fitting a normal distribution
        perc_factor = stats.norm.ppf(perc/100) # this is for a standardized distribution
        stds = fc.std('number')
        perc_fitted = ensmean + perc_factor * stds


        # worst (=warmest) member
        ameans = fc.mean(('lat','lon'))
        worst_mem_idx = np.argmax(ameans.values)
        worst_member = fc.isel(number=worst_mem_idx)

        worst_5_mem_idcs = np.argsort(ameans.values)[-5:]
        worst_5 = fc.isel(number=worst_5_mem_idcs).mean('number')

        # DCA
        anom = fc - ensmean
        anom_sum = anom.sum(('lat', 'lon'))
        anom_mean = anom.mean(('lat', 'lon'))
        plt.figure()
        sns.kdeplot(anom_mean)
        plt.xlabel('anom_mean')
        plt.savefig('kde_anom_mean.svg')

        # flatten ("stack" in xarray)
        #flat_anom = anom.stack(z=('lat','lon'))
        flat_anom_xr = anom.stack(z=('lat','lon'))
        flat_anom = flat_anom_xr.values
        # change to (space, time)
        flat_anom = flat_anom.T
        n_flat = flat_anom.shape[0]
        # here we use the not-normalized definition of the covariance matrix,
        # which is simply a dot product of the data-matrix with its transpose
        # cov_matrix = np.dot(flat_anom ,flat_anom.T)
        cov_matrix = np.cov(flat_anom)
        #dca_flat = np.dot(cov_matrix, np.ones(n_flat).T)
        # more efficient implementation
        dca_flat = np.sum(cov_matrix,1)
        # scaling
        g = dca_flat

        ameans_anom = ameans - ameans.mean()
        # convert from xarray to numpy
        ameans_anom = ameans_anom.values

        # scale g
        # on amplitude of worst member
        dca_scaled_worstmem = g * np.max(ameans_anom) / np.mean(g)
        # an mean amplitude of top 5 members
        dca_scaled_top5 = g * np.mean(np.sort(ameans_anom)[-5:] )/ np.mean(g)

        # convert to array with right coordinates (we select member 1 since the coords are the same for all mems)
        dca_scaled_worstmem = xr.DataArray(dca_scaled_worstmem.T, coords=flat_anom_xr.isel(number=0).coords,
                                   dims=flat_anom_xr.isel(number=0).dims).unstack('z')
        dca_scaled_top5 = xr.DataArray(dca_scaled_top5.T, coords=flat_anom_xr.isel(number=0).coords,
                                   dims=flat_anom_xr.isel(number=0).dims).unstack('z')



        projection = ccrs.PlateCarree()

        perc_empirical_amean_anom = np.mean(perc_empirical - ensmean).values
        perc_fitted_amean_anom = np.mean(perc_fitted - ensmean).values
        worst_member_amean_anom = np.mean(worst_member - ensmean).values
        worst_5_amean_anom = np.mean(worst_5 - ensmean).values
        dca_scaled_worst_amean_anom = np.mean(dca_scaled_worstmem).values
        dca_scaled_top5_amean_anom = np.mean(dca_scaled_top5).values

        clevs = np.arange(14,35,1)
        cmap = plt.cm.magma_r

        plt.figure(figsize=(7,6))
        ax = plt.subplot(111, projection = projection)
        worst_member.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'worst member')
        plt.savefig(f'plots/ensemble_worst_member_worstmem_{datestr}_{leadtime}h.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        perc_empirical.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'empirical {perc}th percentile')
        plt.savefig(f'plots/ensemble_worst_member_perc_{datestr}_{leadtime}h.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        worst_5.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'mean of 5 worst members')
        plt.savefig(f'plots/ensemble_worst_member_top5_{datestr}_{leadtime}h.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        (dca_scaled_worstmem+ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'dca scaled on worst member')
        plt.savefig(f'plots/ensemble_worst_member_dca_scaled_worst_{datestr}_{leadtime}h.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        (dca_scaled_top5+ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'dca scaled on top5')
        plt.savefig(f'plots/ensemble_worst_member_dca_scaled_top5_{datestr}_{leadtime}h.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        ensmean.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title('ensmean')
        plt.suptitle(f'init:{date_init} valid:{date_valid}')
        plt.savefig(f'plots/ensemble_worst_member_ensmean_{datestr}_{leadtime}h.png')



        clevs = np.arange(-3,3,0.25)
        cmap = plt.cm.RdBu_r

        plt.figure(figsize=(7,6))
        ax = plt.subplot(111, projection = projection)
        (worst_member-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'worst member amean={worst_member_amean_anom:.2f}')
        plt.savefig(f'plots/ensemble_worst_member_worstmem_{datestr}_{leadtime}h_anom.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        (perc_empirical-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'empirical {perc}th percentile amean={perc_empirical_amean_anom:.2f}')
        plt.savefig(f'plots/ensemble_worst_member_perc_{datestr}_{leadtime}h_anom.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        (worst_5-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'mean of 5 worst members amean={worst_5_amean_anom:.2f}')
        plt.savefig(f'plots/ensemble_worst_member_top5_{datestr}_{leadtime}h_anom.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        (dca_scaled_worstmem).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'dca scaled on worst member amean={dca_scaled_worst_amean_anom:.2f}')
        plt.savefig(f'plots/ensemble_worst_member_dca_scaled_worst_{datestr}_{leadtime}h_anom.png')

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection=projection)
        (dca_scaled_top5).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'dca scaled on worst5 amean={dca_scaled_top5_amean_anom:.2f}')
        plt.savefig(f'plots/ensemble_worst_member_dca_scaled_top5_{datestr}_{leadtime}h_anom.png')


        plt.close('all')
