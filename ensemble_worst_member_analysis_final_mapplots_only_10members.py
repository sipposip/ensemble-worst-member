

import os
import xarray as xr
from pylab import plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from scipy import stats
import seaborn as sns
import pickle


from xarray_clim import standardize_dataset, sellonlatbox, weighted_areamean

plt.rcParams['savefig.bbox'] = 'tight'
plt.ioff()

if not os.path.exists('plots/'):
    os.mkdir('plots/')


var='167.128'

area=[-5,60,30,40]
# leadtime=72 #h
perc = 95
# ipath='/climstorage/sebastian/ensemble_worst_member/ecmwf_data/'
ipath='/data//ensemble_worst_member/'
m2_res_all = []
leadtime = 72
for date in (pd.to_datetime('201906011200'),pd.to_datetime('201907211200')):

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

        # select only first 10 members
        fc = fc.sel(number=np.arange(1,11))

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

        if date==pd.to_datetime('201906011200'):
                vmax=4
        else:
                vmax=3
        clevs = np.arange(-vmax,vmax,0.25)
        # remove the zero tick
        #clevs = clevs[clevs!=0]
        cmap = plt.cm.RdBu_r

        # sampe, but all combined in one plot
        fig = plt.figure(figsize=(14,12))
        ax1 = plt.subplot(321, projection = projection)
        (worst_member-ensmean).plot.contourf(ax=ax1, transform=projection, levels=clevs, cmap=cmap, extend='both',
                                             add_colorbar=False)
        ax1.coastlines()
        plt.title(f'worst member amean={worst_member_amean_anom:.2f}')


        ax2 = plt.subplot(322, projection=projection)
        (worst_5-ensmean).plot.contourf(ax=ax2, transform=projection, levels=clevs, cmap=cmap, extend='both',
                                        add_colorbar=False)
        ax2.coastlines()
        plt.title(f'mean of 5 worst members amean={worst_5_amean_anom:.2f}')

        ax3 = plt.subplot(323, projection=projection)
        (dca_scaled_worstmem).plot.contourf(ax=ax3, transform=projection, levels=clevs, cmap=cmap, extend='both',
                                         add_colorbar=False)
        ax3.coastlines()
        plt.title(f'dca scaled on worst member amean={dca_scaled_worst_amean_anom:.2f}')

        ax4 = plt.subplot(324, projection=projection)
        p = (dca_scaled_top5).plot.contourf(ax=ax4, transform=projection, levels=clevs, cmap=cmap, extend='both',
                                        add_colorbar=False)
        ax4.coastlines()
        plt.title(f'dca scaled on worst5 amean={dca_scaled_top5_amean_anom:.2f}')


        ax5 = plt.subplot(325, projection=projection)
        (perc_empirical-ensmean).plot.contourf(ax=ax5, transform=projection, levels=clevs, cmap=cmap, extend='both',
                                               add_colorbar=False)
        ax5.coastlines()
        plt.title(f'empirical {perc}th percentile amean={perc_empirical_amean_anom:.2f}')


        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                             wspace=0.02, hspace=0.08)
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(p, cax=cb_ax)
        cbar.set_ticks(np.arange(-vmax,vmax,0.5))
        cbar.set_label('T2m [K]')
        def get_axis_limits(ax, scale=.9):
                return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale


        ax1.set_title("a)", loc='left', fontdict=({'fontweight':'bold'}))
        ax2.set_title("b)", loc='left', fontdict=({'fontweight':'bold'}))
        ax3.set_title("c)", loc='left', fontdict=({'fontweight':'bold'}))
        ax4.set_title("d)", loc='left', fontdict=({'fontweight':'bold'}))
        ax5.set_title("e)", loc='left', fontdict=({'fontweight':'bold'}))
        plt.savefig(f'plots/ensemble_worst_member_allpanels_{datestr}_{leadtime}h_anom_only_10_members.png')




