

import os
import matplotlib
from pylab import plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns



plt.rcParams['savefig.bbox'] = 'tight'


if not os.path.exists('plots2/'):
    os.mkdir('plots2/')

colors = ['#a6cee3', '#1f78b4','#b2df8a','#33a02c']
leadtime = 72

sens_methods = ['bootstrap', 'bootstrap_subens_25', 'mvn_uncertainty']
methods = ['worst_member','dca_scaled_worst', 'worst_5','dca_scaled_worst5']
df = []
for date in pd.date_range('201906011200','201908251200', freq='5d'):
    datestr = date.strftime("%Y%m%d_%H%M")

    for sens_method in sens_methods:
        ifile = f'data/angle_vs_amplitude_{sens_method}_{datestr}_{leadtime}h.pkl'
        df = pd.read_pickle(ifile)
        # drop perc95
        df = df[df['method'] != 'perc95']
        plt.figure()
        sns.scatterplot('angle', 'a', hue='method', data=df, alpha=0.9, palette=colors,
                        hue_order=methods)
        # plot group centers
        sns.scatterplot('angle','a',data=df.groupby('method').mean().reset_index(),
                        s=100,linewidth=2, marker='+',palette=colors, hue='method',
                        legend=False, hue_order=methods)
        datestr = pd.to_datetime(date).strftime("%Y%m%d_%H%M")
        plt.title(sens_method)
        plt.xlabel(r'$\alpha$')
        sns.despine()
        plt.savefig(f'plots2/angle_vs_amplitude_{sens_method}_{datestr}_{leadtime}h.svg')
        plt.close()
