

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

sens_methods = ['bootstrap', 'bootstrap_subens_25', 'mvn_uncertainty']
methods = ['dca', 'worst_member', 'worst_5']
df = []
for date in pd.date_range('201906011200','201908251200', freq='5d'):
    datestr = date.strftime("%Y%m%d_%H%M")

    for sens_method in sens_methods:
        ifile = f'data/angle_vs_amplitude_{sens_method}_{datestr}_{leadtime}h_stdev.txt'
        sub_df = pd.read_table(ifile, sep=r"\s+", skiprows=2, names=['angle', 'a'],
                               index_col=0)

        sub_df['method']=sub_df.index
        sub_df['date'] = date
        sub_df['sens_method'] = sens_method
        df.append(sub_df)


df = pd.concat(df)

sns.catplot('sens_method','a', hue='method', data=df, kind='box')
plt.ylabel('stdev amplitude')
plt.savefig('plots2/sensitivity_barplot_amplitude.svg')
sns.catplot('sens_method','angle',hue='method', data=df, kind='box')
plt.ylabel('stdev angle')
plt.savefig('plots2/sensitivity_barplot_angle.svg')


# statistical tests
stat_res = []
for vvar in ['a', 'angle']:
    for sens_method in sens_methods:
        sub = df.query('sens_method==@sens_method ')
        for method1 in methods:
            for method2 in methods:
                if method1 != method2:
                    m1 = sub.query('method==@method1')[vvar].values
                    m2 = sub.query('method==@method2')[vvar].values
                    # our values are not indepenent, therefore we make a 1 sample
                    # test with the differences
                    _,p = stats.ttest_1samp(m1-m2, popmean=0)
                    stat_res.append(pd.DataFrame(
                        {'p':p,'method1':method1, 'method2':method2,
                         'vvar':vvar, 'sens_method':sens_method}, index=[0]
                    ))

stats_res = pd.concat(stat_res)

# stats_res.fillna(0, inplace=True)

for vvar in ['a', 'angle']:
    for i,sens_method in enumerate(sens_methods):
        plt.figure()
        sub = stats_res.query('sens_method==@sens_method & vvar==@vvar')[['p', 'method1', 'method2']]
        sns.heatmap(sub.pivot_table(values='p', index='method1', columns='method2'),
                    annot=True, vmin=0, vmax=0.5)
        plt.title(f'sens_method:{sens_method} variable:{vvar} p-values')
        plt.savefig(f'plots2/ttes_res_{sens_method}_{vvar}.png')

