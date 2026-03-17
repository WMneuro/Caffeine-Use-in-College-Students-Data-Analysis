# Analysis script for caffeine survey dataset.
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import pingouin as pg
from pathlib import Path
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.stattools import durbin_watson
import numpy as np
from itertools import combinations

script_dir = Path(__file__).resolve().parent
data_path = script_dir / "Caffeine_Project_Data" / "caffeine_survey_responses.xlsx"
caffeine_data = pd.read_excel(data_path) 

gender_map = {'Cis Male': 'Cis Male', # grouped gender identity responses by Cis Male, Cis Female, and tgd to increase statistical power
              'Cis Female': 'Cis Female', 
              'Gender Fluid': 'Trans and Gender-Diverse', 
              'Non Binary': 'Trans and Gender-Diverse',
              'Agender': 'Trans and Gender-Diverse', 
              'Trans Male': 'Trans and Gender-Diverse',
              'Trans Female': 'Trans and Gender-Diverse'}

caffeine_data['gender_grouped'] = caffeine_data['gender'].map(gender_map) # a new column (gender_grouped) is made by mapping the gender responses to the above dictionary

# Treat outlier data through z-score (only for data used in current analysis). Also remove misssing data
caffeine_data_clean = caffeine_data.dropna(subset= ['days_well_rested_per_week',
                                                    'maximum_coffee_cups_day_all_caffeine',
                                                    'gender_grouped'
]).reset_index(drop= True)

n_before = len(caffeine_data_clean)
z_scores = np.abs(stats.zscore(caffeine_data_clean[['days_well_rested_per_week', 'maximum_coffee_cups_day_all_caffeine']]))
caffeine_data_clean = caffeine_data_clean[(z_scores < 3).all(axis=1)].reset_index(drop=True)
n_after = len(caffeine_data_clean)
print(f"Outliers removed: {n_before - n_after} ({n_before} -> {n_after} participants)")

# Comparison of maximum caffeine consumption between gender identity. Compute Statistics 
normality_results_max = pg.normality(
    data= caffeine_data_clean,
    dv= 'maximum_coffee_cups_day_all_caffeine',
    group= 'gender_grouped'
)

homoscedasticity_results_max = pg.homoscedasticity(
    data= caffeine_data_clean,
    dv= 'maximum_coffee_cups_day_all_caffeine',
    group= 'gender_grouped'
)

print('maximum caffeine intake over 1 day','\n'*2)
print(f"Normality results:\n{normality_results_max}\n")
print(f"Homoscedasticity results:\n{homoscedasticity_results_max}\n")

if (normality_results_max['pval'] < 0.05).any():
    print("Normality violated — running Kruskal-Wallis")
    results_max = pg.kruskal(
        data= caffeine_data_clean,
        dv= 'maximum_coffee_cups_day_all_caffeine',
        between = 'gender_grouped'
)

elif not homoscedasticity_results_max['equal_var'].all():
    print("Homoscedasticity violated — running Welch's ANOVA")
    results_max = pg.welch_anova(
        data= caffeine_data_clean,
        dv= 'maximum_coffee_cups_day_all_caffeine',
        between= 'gender_grouped'
)

else:
    print("Assumptions met — running standard ANOVA")
    results_max = pg.anova(
        data= caffeine_data_clean,
        dv= 'maximum_coffee_cups_day_all_caffeine',
        between= 'gender_grouped',
        detailed= True
)

if 'eta-sq' in results_max.columns and not np.isnan(results_max['eta-sq'].values[0]):
    effect_size_max = results_max['eta-sq'].values[0]
    effect_label_max = 'η²'
elif 'np2' in results_max.columns:
    effect_size_max = results_max['np2'].values[0]
    effect_label_max = 'partial η²'
else:
    # Manual epsilon-squared for Kruskal-Wallis
    H = results_max['H'].values[0]
    n = len(caffeine_data_clean)
    effect_size_max = H / ((n**2 - 1) / (n + 1))
    effect_label_max = 'ε²'

print(f"Effect size ({effect_label_max}) = {effect_size_max:.3f}")

print(results_max) 
pval_col= 'p-unc'

if 'p-unc' in results_max.columns:
    pval_col= 'p-unc'
elif 'p_unc' in results_max.columns:
    pval_col= 'p_unc'
elif 'pval' in results_max.columns:
    pval_col= 'pval'
else:
    raise ValueError("Could not find p-value column in results")

if results_max[pval_col].values[0] < 0.05:
    print("Significant — running post-hoc tests")
    parametric= not (normality_results_max['pval'] < 0.05).any()
    posthoc_max= pg.pairwise_tests(
        data= caffeine_data_clean,
        dv= 'maximum_coffee_cups_day_all_caffeine',
        between= 'gender_grouped',
        parametric= parametric,
        padjust= 'bonferroni'
)
    
    print(posthoc_max)
else:
    print("No significant difference found")

# Barplot of Gender data
sns.set_theme(style= "ticks")
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11}
)

fig, ax = plt.subplots(figsize=(10, 6))
bar_labels= ['Cis Female','Cis Male', 'Trans and Gender-Diverse',]
bar_colors= ['#50ad9f', '#f0c571', '#f55f74']
color_map= dict(zip(bar_labels, bar_colors)) # zip combines lists to makes tuples
groups= ['Cis Female', 'Cis Male', 'Trans and Gender-Diverse',]
gender_plot= sns.barplot(data= caffeine_data_clean,
                         x= 'gender_grouped',
                         y= 'maximum_coffee_cups_day_all_caffeine',
                         order= groups,
                         palette= color_map, 
                         capsize= .2,
                         errorbar= 'se',
                         errcolor= 'black',
                         edgecolor= 'black',
                         linewidth= 2,
                         ax= ax
)

sns.stripplot(data= caffeine_data_clean,
             x= 'gender_grouped', 
             y= 'maximum_coffee_cups_day_all_caffeine', 
             order= groups,
             facecolor= 'none',
             color= 'black',
             linewidth= 1.5,
             jitter= .25,
             clip_on= True,
             ax= gender_plot,
             alpha= .4
)
gender_plot.tick_params(axis='x', length=0) # remove x-axis tick marks
sns.despine()
gender_plot.set_title(None)
gender_plot.set_ylabel('Maximum Caffeine Intake Over 1 Day (Cups of Coffee)')
gender_plot.set_xlabel(None)
plt.tight_layout()

fig_dir = Path("Caffeine_Project_Figures") # create a path for the figure folder
fig_dir.mkdir(exist_ok=True) # fig_dir represents current director, mkdir = make directory, exist_ok tells mkdir to do nothing if the folder already exists
fig.savefig(fig_dir / "comparison_gender_max_caffeine.png", dpi=300)

# Compare days well rested between gender identity. Compute Statistics 
normality_results_rested = pg.normality(
    data= caffeine_data_clean,
    dv= 'days_well_rested_per_week',
    group= 'gender_grouped'
)

homoscedasticity_results_rested = pg.homoscedasticity(
    data=caffeine_data_clean,
    dv='days_well_rested_per_week',
    group='gender_grouped'
)
print('days well rested assumption results','\n'*2)
print(f"Normality results:\n{normality_results_rested}\n")
print(f"Homoscedasticity results:\n{homoscedasticity_results_rested}\n")

if (normality_results_rested['pval'] < 0.05).any():
    print("Normality violated — running Kruskal-Wallis")
    results_rested = pg.kruskal(
        data= caffeine_data_clean,
        dv= 'days_well_rested_per_week',
        between= 'gender_grouped'
)
elif not homoscedasticity_results_rested['equal_var'].all():
    print("Homoscedasticity violated — running Welch's ANOVA")
    results_rested = pg.welch_anova(
        data=caffeine_data_clean,
        dv='days_well_rested_per_week',
        between='gender_grouped'
)

else:
    print("Assumptions met — running standard ANOVA")
    results_rested = pg.anova(
        data= caffeine_data_clean,
        dv= 'days_well_rested_per_week',
        between= 'gender_grouped',
        detailed= True
)

if 'eta-sq' in results_rested.columns and not np.isnan(results_rested['eta-sq'].values[0]):
    effect_size_rested = results_rested['eta-sq'].values[0]
    effect_label_rested = 'η²'

elif 'np2' in results_rested.columns:
    effect_size_rested = results_rested['np2'].values[0]
    effect_label_rested = 'partial η²'

else:
    # Manual epsilon-squared for Kruskal-Wallis
    H = results_rested['H'].values[0]
    n = len(caffeine_data_clean)
    effect_size_rested = H / ((n**2 - 1) / (n + 1))
    effect_label_rested = 'ε²'

print(f"Effect size ({effect_label_rested}) = {effect_size_rested:.3f}")

print(results_rested) 
pval_col= 'p-unc'

if 'p-unc' in results_rested.columns:
    pval_col= 'p-unc'
elif 'p_unc' in results_rested.columns:
    pval_col= 'p_unc'
elif 'pval' in results_rested.columns:
    pval_col= 'pval'
else:
    raise ValueError("Could not find p-value column in results")

if results_rested[pval_col].values[0] < 0.05:
    print("Significant — running post-hoc tests")
    parametric= not (normality_results_rested['pval'] < 0.05).any()
    posthoc_rested= pg.pairwise_tests(
        data= caffeine_data_clean,
        dv= 'days_well_rested_per_week',
        between= 'gender_grouped',
        parametric= parametric,
        padjust= 'bonferroni'
)
    
    print(posthoc_rested)
else:
    print("No significant difference found")

# Visualize the days well rested per week data.
fig, ax = plt.subplots(figsize=(10, 6))
gender_plot= sns.barplot(data= caffeine_data_clean,
                         x= 'gender_grouped',
                         y= 'days_well_rested_per_week',
                         order= groups,
                         palette= color_map, 
                         capsize= .2,
                         errorbar= 'se',
                         errcolor= 'black',
                         edgecolor= 'black',
                         linewidth= 2,
                         ax= ax
)

sns.stripplot(data= caffeine_data_clean,
             x= 'gender_grouped', 
             y= 'days_well_rested_per_week', 
             order= groups,
             facecolor= 'none',
             color= 'black',
             linewidth= 1.5,
             jitter= .25,
             clip_on= True,
             ax= gender_plot,
             alpha= .4
)
gender_plot.tick_params(axis='x', length=0)
sns.despine()
gender_plot.set_title(None)
gender_plot.set_ylabel('Days Well Rested Per Week (Day)')
gender_plot.set_xlabel(None)
plt.tight_layout()
fig.savefig(fig_dir / "comparison_gender_days_wellrested_perweek.png", dpi=300)

# Compare time of caffeine use between gender identity. Compute statistics. 
caffeine_data_chi2 = caffeine_data_clean[caffeine_data_clean['time_caffeine_consumed'] != 'I do not consume caffeine'].reset_index(drop= True)
chi2_results = pg.chi2_independence(
    data= caffeine_data_chi2,
    x= 'gender_grouped',
    y= 'time_caffeine_consumed'
)
print(chi2_results, 2*'\n', 'Assumptions violated, using monte carlo chi square')

# The assumptions for chi-squared are violated, so instead we will Monte Carlo Chi-Square Test
chi2_contingency_table = pd.crosstab(caffeine_data_chi2['gender_grouped'], 
                                     caffeine_data_chi2['time_caffeine_consumed'],
)

print(chi2_contingency_table)

np.random.seed(31) # set a random seed for reproducibility
chi2_obs, p, dof, expected = stats.chi2_contingency(chi2_contingency_table.values)

chi2_sim = [] 
expected_prob = expected / expected.sum()
sample_n = len(caffeine_data_chi2['gender_grouped'])

for i in range(10000):
    sim_counts = np.random.multinomial(sample_n, expected_prob.flatten())
    sim_table = sim_counts.reshape(chi2_contingency_table.values.shape)
    try: 
        chi2_sim.append(stats.chi2_contingency(sim_table)[0])
    except ValueError: # to skip tables that generate zero cells, due to the small size to the tgd group
        continue

chi2_sim = np.array(chi2_sim)
print(chi2_contingency_table.shape)
print(chi2_obs, 2*'\n')
p_mc = np.mean(chi2_sim >= chi2_obs)
print(f"Simulations completed: {len(chi2_sim)} / 10000") # number of simulations completed
print("Observed chi-square:", chi2_obs)
print("Monte Carlo p-value:", p_mc)

# the monte carlo simulation gives a signifigant p-value, so examine effect size and post-hoc.
chi2_stats = chi2_results[2]
cramers_v = chi2_stats.loc[chi2_stats['test'] == 'pearson', 'cramer'].values[0]
p_pearson = chi2_stats.loc[chi2_stats['test'] == 'pearson', 'pval'].values[0]
chi2_val  = chi2_stats.loc[chi2_stats['test'] == 'pearson', 'chi2'].values[0]
dof = chi2_stats.loc[chi2_stats['test'] == 'pearson', 'dof'].values[0]
print(f"X²({int(dof)}) = {chi2_val:.3f}, p = {p_pearson:.3f}, Cramér's V = {cramers_v:.3f}")

pairs = list(combinations(groups, 2))
n_comparisons = len(pairs)   # 3 comparisons for Bonferroni

chi2_posthoc = []

for g1, g2 in pairs:
    subset = caffeine_data_chi2[caffeine_data_chi2['gender_grouped'].isin([g1, g2])]
    table = pd.crosstab(subset['gender_grouped'], subset['time_caffeine_consumed'])
    try:
        chi2_pair, p_val, dof_posthoc, expected = stats.chi2_contingency(table)
        p_bonferroni = min(p_val * n_comparisons, 1.0)
        chi2_posthoc.append({
        'Group 1':      g1,
        'Group 2':      g2,
        'chi2':         chi2_pair,
        'p_raw':        p_val,
        'p_bonferroni': p_bonferroni,
        'significant':  p_bonferroni < 0.05
        })
    except ValueError:
        chi2_posthoc.append({
        'Group 1': g1,
        'Group 2': g2,
        'chi2': np.nan,
        'p_raw': np.nan,
        'p_bonferroni': np.nan,
        'significant': False
        })

posthoc_chi2_df = pd.DataFrame(chi2_posthoc)
print(posthoc_chi2_df)

# Visualize the frequency of time of caffeine use by gender identity data.
chi2_plot = pd.crosstab(caffeine_data_chi2['gender_grouped'], caffeine_data_chi2['time_caffeine_consumed'], normalize= 'index')
chi2_plot = chi2_plot[['Morning', 'Afternoon', 'Evening']]

fig, ax = plt.subplots(figsize=(10, 6))

chi2_plot.plot(
    kind= 'bar',
    stacked= True,
    color= ['#298c8c', '#a00000', '#b8b8b8'], 
    edgecolor= 'black',
    linewidth= 0.2,
    ax= ax
)

ax.tick_params(axis= 'x', length=0)
ax.set_title(None)
ax.set_ylabel('Proportion')
ax.set_xlabel(None)
ax.set_xticklabels(['Cis Female', 'Cis Male', 'Trans and Gender-Diverse'], rotation=0)
ax.legend(title= 'Time of Consumption', frameon= False, bbox_to_anchor= (1, 1), loc= 'upper left')
ax.set_ylim(0, 1.0)
sns.despine(ax= ax)
fig.tight_layout()
fig.savefig(fig_dir / "comparison_caffeine_time_gender.png", dpi= 300)