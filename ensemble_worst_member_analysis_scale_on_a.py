

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
import scipy.spatial

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
        # get the 95th percentile of the area mean anomalies
        amean_anom_perc = np.percentile(ameans_anom,95)

        # scale g to have the same area mean anomaly
        dca_scaled = g * amean_anom_perc / np.mean(g)

        # convert to array with right coordinates (we select member 1 since the coords are the same for all mems)
        dca_flat_xr = xr.DataArray(dca_scaled.T, coords=flat_anom_xr.isel(number=0).coords,
                                   dims=flat_anom_xr.isel(number=0).dims)
        dca = dca_flat_xr.unstack('z')


        projection = ccrs.PlateCarree()

        perc_empirical_amean_anom = np.mean(perc_empirical - ensmean).values
        perc_fitted_amean_anom = np.mean(perc_fitted - ensmean).values
        worst_member_amean_anom = np.mean(worst_member - ensmean).values
        worst_5_amean_anom = np.mean(worst_5 - ensmean).values
        dca_amean_anom = np.mean(dca).values

        clevs = np.arange(14,35,1)
        cmap = plt.cm.magma_r

        plt.figure(figsize=(15,12))

        ax = plt.subplot(322, projection = projection)
        worst_member.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'worst member')

        ax = plt.subplot(323, projection = projection)
        perc_empirical.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'empirical {perc}th percentile')

        # ax = plt.subplot(324, projection = projection)
        # perc_fitted.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        # ax.coastlines()
        # plt.title(f'fitted {perc}th percentile')

        ax = plt.subplot(324, projection = projection)
        worst_5.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'mean of 5 worst members')

        ax = plt.subplot(321, projection = projection)
        (dca+ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'dca')
        ax = plt.subplot(325, projection = projection)
        ensmean.plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title('ensmean')
        plt.suptitle(f'init:{date_init} valid:{date_valid}')

        plt.savefig(f'ensemble_worst_member_scale_a_{datestr}_{leadtime}h.png')



        clevs = np.arange(-3,3,0.25)
        cmap = plt.cm.RdBu_r

        plt.figure(figsize=(15,8))
        ax = plt.subplot(223, projection = projection)
        (perc_empirical-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'empirical {perc}th percentile amean={perc_empirical_amean_anom:.2f}')

        # ax = plt.subplot(224, projection = projection)
        # (perc_fitted-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        # ax.coastlines()
        # plt.title(f'fitted {perc}th percentile amean={perc_fitted_amean_anom:.2f}')
        #
        ax = plt.subplot(224, projection = projection)
        (worst_5-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'mean of 5 worst members amean={worst_5_amean_anom:.2f}')

        ax = plt.subplot(222, projection = projection)
        (worst_member-ensmean).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'worst member amean={worst_member_amean_anom:.2f}')

        ax = plt.subplot(221, projection = projection)
        (dca).plot.contourf(ax=ax, transform=projection, levels=clevs, cmap=cmap, extend='both')
        ax.coastlines()
        plt.title(f'dca amean={dca_amean_anom:.2f}')


        plt.suptitle(f'init:{date_init} valid:{date_valid}')
        plt.tight_layout()
        plt.savefig(f'plots/ensemble_worst_member_scale_a_{datestr}_{leadtime}h_anom.png')


        plt.close('all')


        # compute angles with repect to unit vector
        ref = np.ones(n_flat)
        def angle(v,w):
            return np.arccos(np.dot(v,w) / (np.linalg.norm(v)* np.linalg.norm(w)))

        angle_dca = angle(dca.values.flatten(),ref)
        angle_worst_member = angle(worst_member.values.flatten(),ref)
        angle_dca = angle(dca.values.flatten(),ref)





