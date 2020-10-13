

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
from tqdm import tqdm
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

area_base=np.array([-5,60,30,40])

# create perturbed areas. all combinations of adding n, substracting n or leaving the same,
# at all 4 sides, with n as 2
areas = []
for n in [2]:
    for sig1 in [1,0,-1]:
        for sig2 in [1, 0, -1]:
            for sig3 in [1, 0, -1]:
                for sig4 in [1, 0, -1]:
                    areas.append(area_base + np.array([n*sig1,n*sig2,n*sig3,n*sig4]))


perc = 95
ipath='/climstorage/sebastian/ensemble_worst_member/ecmwf_data/'
leadtime = 72
for date in tqdm(pd.date_range('201906011200','201908251200', freq='5d')):
    print(date)
    datestr = date.strftime("%Y%m%d_%H%M")
    ifile = f'{ipath}/ecmwf_ens_{datestr}_{var}_for_ens_worst_member.nc.nc'


    data = xr.open_dataset(ifile)
    data = standardize_dataset(data)
    data = data['t2m']
    data = data.astype('float64')
    data_fullarea = data - 273.15

    n_ens = len(data.number)
    leadidx = leadtime // 6
    res = []
    for i_area,area in tqdm(enumerate(areas)):

        data = sellonlatbox(data_fullarea, *area)

        # make sure that the data is indeed 6 hourly
        assert(np.all(pd.to_timedelta(data.time.diff('time')) == pd.to_timedelta('6h')))



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


        perc_empirical_amean_anom = np.mean(perc_empirical - ensmean).values
        perc_fitted_amean_anom = np.mean(perc_fitted - ensmean).values
        worst_member_amean_anom = np.mean(worst_member - ensmean).values
        worst_5_amean_anom = np.mean(worst_5 - ensmean).values
        dca_amean_anom = np.mean(dca).values


        # compute angles with repect to unit vector
        ref = np.ones(n_flat)
        # https://math.stackexchange.com/questions/53291/how-is-the-angle-between-2-vectors-in-more-than-3-dimensions-defined
        def angle(v,w):
            return np.arccos(np.dot(v,w) / (np.linalg.norm(v)* np.linalg.norm(w)))

        angle_dca = angle(dca.values.flatten(),ref)
        angle_worst_member = angle((worst_member-ensmean).values.flatten(),ref)
        angle_worst5 = angle((worst_5-ensmean).values.flatten(),ref)
        angle_perc = angle((perc_empirical-ensmean).values.flatten(),ref)


        df = pd.DataFrame({'angle':[angle_dca,
                                    angle_perc,
                                    angle_worst_member,
                                    angle_worst5],
                           'a':[dca_amean_anom,
                                perc_empirical_amean_anom,
                                worst_member_amean_anom,
                                worst_5_amean_anom,
                                ],
                           'method':['dca','perc95','worst_member','worst_5'],
                           'i_area':i_area
                           })

        res.append(df)



    res = pd.concat(res)
    # 'a' has array type, convert to float
    res['a'] = res['a'].astype('float')
    res['cos'] = np.cos(res['angle'])

    plt.figure()
    sns.scatterplot('angle','a', hue='method', data=res, alpha=0.3)
    plt.savefig(f'plots/angle_vs_amplitude_areasens_{datestr}_{leadtime}h.svg')

    res.to_pickle(f'data/angle_vs_amplitude_areasens_{datestr}_{leadtime}h.pkl')
    res.to_csv(f'data/angle_vs_amplitude_areasens_{datestr}_{leadtime}h.csv')

    print(res.groupby('method').std()[['angle','a']])
    print(res.groupby('method').std()[['angle','a']],
          file=open(f'data/angle_vs_amplitude_areasens_{datestr}_{leadtime}h_stdev.txt','w'))

    plt.close('all')