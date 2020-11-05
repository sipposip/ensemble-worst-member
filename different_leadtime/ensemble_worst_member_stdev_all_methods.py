

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


leadtime = 48

sens_methods = ['bootstrap', 'bootstrap_subens_25', 'mvn_uncertainty']
methods = ['worst_member','dca_scaled_worst', 'worst_5','dca_scaled_worst5']
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
# drop perc95
df = df[df['method']!='perc95']

# paired colorblindsafe palette from colorbrewer2.org
colors = ['#a6cee3', '#1f78b4','#b2df8a','#33a02c']
p = sns.catplot('sens_method','a', hue='method', data=df, kind='box', hue_order=methods, palette=colors)
p.set_xticklabels(rotation=15)
plt.ylabel('stdev amplitude')
plt.savefig(f'plots2/sensitivity_barplot_amplitude_{leadtime}h.svg')
p=sns.catplot('sens_method','angle',hue='method', data=df, kind='box', hue_order=methods, palette=colors)
p.set_xticklabels(rotation=15)
plt.ylabel('stdev angle')
plt.savefig(f'plots2/sensitivity_barplot_angle_{leadtime}h.svg')



# statistical tests
stat_res = []

for vvar in ['a', 'angle']:
    for sens_method in sens_methods:
        sub = df.query('sens_method==@sens_method ')
        methods_remaining = methods[:]
        for method1 in methods:
            methods_remaining.remove(method1)
            for method2 in methods_remaining:
                #if method1 != method2:
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


# make tables
# for amplitude, there is only one non-redundtan combination, because
# worst_member and dca_scaled_worst have the sampe amplitude, and
# worst_5 and dca_scaled_worst_5 as well

sub_a = stats_res[stats_res['vvar']=='a']
sub_a = sub_a.query('method1=="worst_member" & method2=="worst_5"')

# for angle, only the two dca methods are redundant
sub_angle = stats_res[stats_res['vvar']=='angle']
sub_angle = sub_angle.query('method1!="dca_scaled_worst5" & method2!="dca_scaled_worst5"')
sub_angle = sub_angle.replace({'dca_scaled_worst':'dca'})

sub_a[['sens_method','p']].to_csv(f'stat_test_a_{leadtime}h.csv', index=False)
sub_angle[['method1','method2','sens_method','p']].round(3).to_csv(f'stat_test_angle_{leadtime}h.csv', index=False)
