


from pylab import plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
from xarray_clim import standardize_dataset, sellonlatbox, weighted_areamean

area=[-5,60,30,40]
for datestr in ('20190601_0000', '20190721_0000'):
    data = xr.open_dataset(f'data_efi/ecmwf_ens_{datestr}_167.132_for_ens_worst_member.grb.nc')
    data = data.squeeze()
    data = standardize_dataset(data)
    data = data['t2i']
    data = sellonlatbox(data, *area)
    projection = ccrs.PlateCarree()

    clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    cmap = plt.cm.RdBu_r

    plt.figure(figsize=(7, 6))
    ax = plt.subplot(111, projection=projection)
    data.plot.contourf(ax=ax, transform=projection,levels=clevs,
                       cmap=cmap, extend='both')
    ax.coastlines()
    plt.title(f'EFI')
    plt.savefig(f'plots/EFI_{datestr}.png')