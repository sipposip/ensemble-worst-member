

import os
import matplotlib
from pylab import plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns



plt.rcParams['savefig.bbox'] = 'tight'
sns.set_palette('colorblind')

if not os.path.exists('plots2/'):
    os.mkdir('plots2/')


leadtime = 72

sens_methods = ['bootstrap', 'bootstrap_subens_5', 'mvn_uncertainty']
methods = ['worst_member','dca_scaled_worst', 'worst_5','dca_scaled_worst5']
df = []
for date in pd.date_range('201906011200','201908251200', freq='5d'):
    datestr = date.strftime("%Y%m%d_%H%M")

    for sens_method in sens_methods:
        ifile = f'data/angle_vs_amplitude_{sens_method}_{datestr}_{leadtime}h_stdev_smallense10.txt'
        sub_df = pd.read_table(ifile, sep=r"\s+", skiprows=2, names=['angle', 'a'],
                               index_col=0)

        sub_df['method']=sub_df.index
        sub_df['date'] = date
        sub_df['sens_method'] = sens_method
        df.append(sub_df)


df = pd.concat(df)
# drop perc95
df = df[df['method']!='perc95']

# paired colorblindsafe palette from colorbrewer2.org
colors = ['#a6cee3', '#1f78b4','#b2df8a','#33a02c']
p = sns.catplot('sens_method','a', hue='method', data=df, kind='box', hue_order=methods, palette=colors)
p.set_xticklabels(rotation=15)
plt.ylabel('stdev $a$')
plt.xlabel('robustness procedure')
plt.savefig('plots2/sensitivity_barplot_amplitude_smallensemble.svg')
p=sns.catplot('sens_method','angle',hue='method', data=df, kind='box', hue_order=methods, palette=colors)
p.set_xticklabels(rotation=15)
plt.ylabel(r'stdev $\alpha$')
plt.xlabel('robustness procedure')
plt.savefig('plots2/sensitivity_barplot_angle_smallensemble.svg')

