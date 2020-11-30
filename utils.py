import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats
import numpy as np
import seaborn as sns
from typing import NamedTuple

def formatFloat(fmt, val):
  """
  Format as a decimal but removing leading 0s
  
  Parameters
  ----------
  fmt : str
      The format to use (which may include leading zeros)
  val: float
      The float to format 

  Returns
  -------
  str
      Formatted string
  """

  ret = fmt % val
  if ret.startswith("0."):
    return ret[1:]
  if ret.startswith("-0."):
    return "-" + ret[2:]

  return ret

def apaStyleCorrelation(val, name = "r"):
  return name + ' = ' + formatFloat("%.3f", val)

def apaStyleStatistic(val, name = "Z", format = "%.2f"):
  return name + ' = ' + formatFloat(format, val)

def apaStylePValue(val):
  if val < 0.001: 
    return "p < .001"  
  return "p = " + formatFloat("%.3f", val)

def apaStyleCombined(r, p, df = None, rname = "r"):
  if df is not None:
      rname = f"{rname}({df})"
  rs = apaStyleCorrelation(r, rname)
  ps = apaStylePValue(p)
  return f"({rs}, {ps})"

def set_style(markers=False):
    # First reset to the matplotlib defaults
    mpl.rcdefaults() 
    # Now apply the paper style to get font sizes and then our styles
    plt.style.use(["eedi.mplstyle"])
    
    if markers == True:
      plt.rcParams['axes.prop_cycle'] = cycler('color', ['0.20', '0.40', '0.60', '0.80']) + cycler('marker', ['o', 's', 'D', 'v'])

def export_fig(fig, filepath):
    for ax in fig.axes:
        ax.set_title("")    
    formats  = ["pdf", "png"]
    for i in formats:
        fig.savefig(f"{filepath}.{i}", format=i)

def conf_scatter(x, y, s, title, xlabel, ylabel, ax, alpha=0.3):
    r, p = stats.spearmanr(x, y)
    
    # Degrees of freedom is number of pairs minus 2
    df = len(x) - 2
    
    rs = apaStyleCorrelation(r, f"rs({df})")
    ps = apaStylePValue(p)
        
    ax.scatter(x, y, s=s, alpha=alpha, edgecolors='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks([0,25,50,75,100])
    ax.set_xticklabels([0,25,50,75,100])

    ax.set_yticks([0,25,50,75,100])
    ax.set_yticklabels([0,25,50,75,100])
    
    print(f"{title} ({rs}, {ps})")
    
    ax.plot([0,100],[0,100], color='grey', linestyle='--', linewidth=2)

    l1 = ax.scatter([],[], s=10, edgecolors='none', alpha=alpha, color="C0")
    l2 = ax.scatter([],[], s=50, edgecolors='none', alpha=alpha, color="C0")
    l3 = ax.scatter([],[], s=100, edgecolors='none', alpha=alpha, color="C0")
    l4 = ax.scatter([],[], s=200, edgecolors='none', alpha=alpha, color="C0")

    labels = ["10", "50", "100"]

    leg = ax.legend([l1, l2, l3], labels, title='No. Answers', labelspacing=1.0)
  
def conf_distplot(x, xlabel, ax, ylabel, title):
    bins = np.linspace(-75, 75, 50)
    sns.distplot(x, kde=False, bins=bins)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
  
def conf_dualplot(x, y, s, title, xlabel, ylabel, d, dlabel, ylabel2, title2, alpha=0.3, filepath=None, figsize=(10,10)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        
    conf_scatter(x, y, s, title, xlabel, ylabel, axes[0], alpha)
    
    conf_distplot(d, dlabel, axes[1], ylabel2, title2)
        
    # Setting equal display aspect (rather than equal axis range aspect)    
    # https://stackoverflow.com/a/48143380/1138558
    dispRatio = 1
    for i, ax in enumerate(axes.flat, start=1):
        ax.set(aspect=1.0/ax.get_data_ratio()*dispRatio)
        
    if filepath is not None:
        export_fig(fig, filepath)


def conf_distplot2(x, xlabel, ax, ylabel, title):
    bins = np.linspace(-75, 75, 50)
    sns.distplot(x, kde=False, bins=bins)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    
class WilcoxonRankBiserialCorrelation(NamedTuple):
    favorable: int
    unfavorable: int
    total: int
    rbc: float


def wilcoxon_signed_matched_pairs_rank_biserial_correlation(x, y):
    """
    Calculate the Matched Pairs Rank Biserial Correlation for the Wilcoxon 
    Signed Ranks test. 

    This is used instead of Cohen's d for Wilcoxon Signed-Rank tests.

    https://journals.sagepub.com/doi/pdf/10.2466/11.IT.3.1

    Parameters
    ----------
    x : array
        First set of measurements.
    y : array
        Second set of measurements.

    Returns
    -------
    float
        Wilcoxon Signed Matched Pairs Rank Biserial Correlation
    """

    d = y - x

    d = np.extract(np.not_equal(d, 0), d)

    count = len(d)

    r = stats.rankdata(abs(d))

    favorable = np.sum((d > 0) * r, axis=0)
    unfavorable = np.sum((d < 0) * r, axis=0)
    
    # The total should equal 0.5 * count * (count + 1)
    total = np.sum(r, axis=0)
    
    rbc = favorable / total - unfavorable / total
    
    # Confidence interval using bootstrapped. The library only accepts a single valued array
    # so we work with d.

    return WilcoxonRankBiserialCorrelation(favorable, unfavorable, total, rbc)

class MannWhitneyRankBiserialCorrelation(NamedTuple):
    ties: int
    favorable: int
    unfavorable: int
    u: float
    cles: float
    rbc: float

def mann_whitney_rank_biserial_correlation(x, y):
    """
    Calculate the Rank Biserial Correlation for two independent groups for the
    Mann-Whitney U test. 

    This is used instead of Cohen's d for Mann-Whitney U tests.

    https://journals.sagepub.com/doi/pdf/10.2466/11.IT.3.1

    Make a matrix where x are the rows and y are the columns and the cells are 
    the differences. Count the number of negatives, positives and zeros (i.e.
    ties).

    Calculate CLES = (positive + 0.5 * zeros) / (len(x) * len(y))

    Note: If we already know the Mann-Whitney U value for a two-sided test then
    the CLES value is just U / (n1 * n2).

    Parameters
    ----------
    x : array
        Array of values for control group
    y : array
        Array of values for treatment group

    Returns
    -------
    MannWhitneyRankBiserialCorrelation
        Mann Whitney Rank Biserial Correlation
    """
    
    n1 = len(x)
    n2 = len(y)
    
    mx = np.tile(np.array([x]), (len(y), 1))
    my = np.tile(np.array([y]).transpose(), (1, len(x)))

    ties = (mx == my).sum()
    
    favorable = (mx < my).sum()
    unfavorable = (mx > my).sum()

    u = favorable + 0.5 * ties

    cles = u / (n1 * n2)

    rbc = favorable / (n1 * n2) - unfavorable / (n1 * n2)

    return MannWhitneyRankBiserialCorrelation(ties, favorable, unfavorable, u, cles, rbc)

def students_describe(df):
    """
    Return the mean, std and median statistics for a dataframe of students.
    
    Parameters
    ----------
    df : dataframe
        The students

    Returns
    -------
    df_desc
        Mean, std and median statistics.
    """
    df_desc = (df[["AnswersCount", "Facility", "MeanConfidence"]].describe().T)[["mean","std","50%"]]
    
    df_desc.rename(columns = { "50%": "Median" }, inplace=True)

    for key, value in df_desc.items():
        df_desc[key] = df_desc[key].apply("{:.1f}".format)

    return df_desc

def students_kruskal(dfs, col):
    data = []

    for df in dfs:
        data.append(df[col])
    
    s, p = stats.kruskal(*data, nan_policy="omit")

    d = len(dfs) - 1

    print(f"The Kruskal-Wallis test statistic is H({d}) = {s:.1f}, {apaStylePValue(p)}.")

def students_mann_whitney(df1, df2, col):
    
    d1 = df1[col]
    d2 = df2[col]

    s, p = stats.mannwhitneyu(d1, d2, alternative="two-sided")

    mdn1 = np.median(d1)
    mdn2 = np.median(d2)

    print(f"The Mann-Whitney test statistic is U = {s:.1f}, {apaStylePValue(p)}. The medians are {mdn1:.1f} and {mdn2:.1f}.")
