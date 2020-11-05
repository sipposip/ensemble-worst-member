

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

from tqdm import trange

from xarray_clim import standardize_dataset, sellonlatbox, weighted_areamean

plt.rcParams['savefig.bbox'] = 'tight'
plt.ioff()

if not os.path.exists('plots/'):
    os.mkdir('plots/')
if not os.path.exists('data/'):
    os.mkdir('data/')


var='167.128'

area=[-5,60,30,40]
perc = 95
ipath='/climstorage/sebastian/ensemble_worst_member/ecmwf_data/'
# ipath='/data//ensemble_worst_member/'
# date=pd.to_datetime('201907200000')
leadtime = 48
for date in pd.date_range('201906011200','201908251200', freq='5d'):
    print(date)
    datestr = date.strftime("%Y%m%d_%H%M")
    ifile = f'{ipath}/ecmwf_ens_{datestr}_{var}_for_ens_worst_member.nc.nc'


    data = xr.open_dataset(ifile)
    data = standardize_dataset(data)
    data = data['t2m']
    data = data.astype('float64')
    data = data - 273.15
    data = sellonlatbox(data, *area)

    n_ens = len(data.number)

    # make sure that the data is indeed 6 hourly
    assert(np.all(pd.to_timedelta(data.time.diff('time')) == pd.to_timedelta('6h')))

    leadidx = leadtime//6

    # bootstrapping
    n_bootstrap = 50
    bootstrap_res = []

    for i_bootstrap in trange(n_bootstrap):
        fc = data.isel(time=leadidx)

        mem_idcs = np.random.randint(0,n_ens, size=n_ens)

        fc = fc.isel(number=mem_idcs)

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



        perc_empirical_amean_anom = np.mean(perc_empirical - ensmean).values
        perc_fitted_amean_anom = np.mean(perc_fitted - ensmean).values
        worst_member_amean_anom = np.mean(worst_member - ensmean).values
        worst_5_amean_anom = np.mean(worst_5 - ensmean).values
        dca_scaled_worst_amean_anom = np.mean(dca_scaled_worstmem).values
        dca_scaled_top5_amean_anom = np.mean(dca_scaled_top5).values


        # compute angles with repect to unit vector
        ref = np.ones(n_flat)
        # https://math.stackexchange.com/questions/53291/how-is-the-angle-between-2-vectors-in-more-than-3-dimensions-defined
        def angle(v,w):
            return np.arccos(np.dot(v,w) / (np.linalg.norm(v)* np.linalg.norm(w)))

        angle_dca_scaled_worst = angle(dca_scaled_worstmem.values.flatten(),ref)
        angle_dca_scaled_top5 = angle(dca_scaled_top5.values.flatten(),ref)
        angle_worst_member = angle((worst_member-ensmean).values.flatten(),ref)
        angle_worst5 = angle((worst_5-ensmean).values.flatten(),ref)
        angle_perc = angle((perc_empirical-ensmean).values.flatten(),ref)


        df = pd.DataFrame({'angle':[angle_dca_scaled_worst,
                                    angle_dca_scaled_top5,
                                    angle_perc,
                                    angle_worst_member,
                                    angle_worst5],
                           'a':[dca_scaled_worst_amean_anom,
                                dca_scaled_top5_amean_anom,
                                perc_empirical_amean_anom,
                                worst_member_amean_anom,
                                worst_5_amean_anom,
                                ],
                           'method':['dca_scaled_worst','dca_scaled_worst5','perc95','worst_member','worst_5'],
                           'i_bootstrap':i_bootstrap
                           })

        bootstrap_res.append(df)



    bootstrap_res = pd.concat(bootstrap_res)
    # 'a' has array type, convert to float
    bootstrap_res['a'] = bootstrap_res['a'].astype('float')
    bootstrap_res['cos'] = np.cos(bootstrap_res['angle'])

    plt.figure()
    sns.scatterplot('angle','a', hue='method', data=bootstrap_res, alpha=0.3)
    plt.savefig(f'plots/angle_vs_amplitude_bootstrap_{datestr}_{leadtime}h.svg')

    bootstrap_res.to_pickle(f'data/angle_vs_amplitude_bootstrap_{datestr}_{leadtime}h.pkl')

    print(bootstrap_res.groupby('method').std()[['angle','a']])
    print(bootstrap_res.groupby('method').std()[['angle','a']],
          file=open(f'data/angle_vs_amplitude_bootstrap_{datestr}_{leadtime}h_stdev.txt','w'))