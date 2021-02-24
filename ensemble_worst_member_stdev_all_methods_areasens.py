

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
# drop perc95
df = df[df['method']!='perc95']

# change names
df = df.replace({'method':{'dca_scaled_worst':'DCA1', 'dca_scaled_worst5':'DCA5', 'worst_5':'W5', 'worst_member':'W1'}})
methods = ['W1','DCA1', 'W5','DCA5']
# paired colorblindsafe palette from colorbrewer2.org
colors = ['#a6cee3', '#1f78b4','#b2df8a','#33a02c']
p=sns.catplot('method','a', data=df, kind='box', order=methods, palette=colors)
p.set_xticklabels(rotation=15)
plt.ylabel('stdev $a$')
plt.savefig('plots2/areasens_barplot_amplitude.svg')
p=sns.catplot('method','angle', data=df, kind='box', order=methods, palette=colors)
p.set_xticklabels(rotation=15)
plt.ylabel(r'stdev $\alpha$')
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

