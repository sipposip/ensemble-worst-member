

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

methods = ['dca', 'worst_member', 'worst_5']
df = []
for date in pd.date_range('201906011200','201908251200', freq='5d'):
    datestr = date.strftime("%Y%m%d_%H%M")

    ifile = f'data/angle_vs_amplitude_areasens_{datestr}_{leadtime}h_stdev.txt'
    sub_df = pd.read_table(ifile, sep=r"\s+", skiprows=2, names=['angle', 'a'],
                           index_col=0)

    sub_df['method']=sub_df.index
    sub_df['date'] = date
    df.append(sub_df)


df = pd.concat(df)

sns.catplot('method','a', hue='method', data=df, kind='box')
plt.ylabel('stdev amplitude')
plt.savefig('plots2/areasens_barplot_amplitude.svg')
sns.catplot('method','angle',hue='method', data=df, kind='box')
plt.ylabel('stdev angle')
plt.savefig('plots2/areasens_barplot_angle.svg')


# statistical tests
stat_res = []
for vvar in ['a', 'angle']:
        for method1 in methods:
            for method2 in methods:
                if method1 != method2:
                    m1 = df.query('method==@method1')[vvar].values
                    m2 = df.query('method==@method2')[vvar].values
                    # our values are not indepenent, therefore we make a 1 sample
                    # test with the differences
                    _,p = stats.ttest_1samp(m1-m2, popmean=0)
                    stat_res.append(pd.DataFrame(
                        {'p':p,'method1':method1, 'method2':method2,
                         'vvar':vvar}, index=[0]
                    ))

stats_res = pd.concat(stat_res)

# stats_res.fillna(0, inplace=True)

for vvar in ['a', 'angle']:
    plt.figure()
    sub = stats_res.query('vvar==@vvar')[['p', 'method1', 'method2']]
    sns.heatmap(sub.pivot_table(values='p', index='method1', columns='method2'), annot=True,
                vmin=0, vmax=1)
    plt.title(f'areasensitivity  {vvar} p-values')
    plt.savefig(f'plots2/ttes_res_areasens_{vvar}.png')

