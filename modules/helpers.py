import pandas as pd

"""
A set of helper functions for kiva loan default prediction
"""

def import_data(file, file_format='csv'):

    """Import data as a pandas dataframe.

    Import file with csv, tsv, excel or txt format and display helpful information about the dataframe.

    Args:
        file: file pathname
        file_format: file format, default to 'csv'

    Returns:
        A pandas dataframe.
    """

    if file_format == 'csv':
        data = pd.read_csv(file)
    elif file_format == 'tsv':
        data = pd.read_csv(file, sep='\t')
    elif file_format == 'excel':
        data = pd.read_excel(file)
    elif file_format == 'txt':
        data = pd.read_table(file)
    else:
        raise ValueError("Invalid file format. Available options are 'csv', 'tsv', 'excel' and 'txt'.")

    print('Reading in the {} dataset'.format(file))
    print('Dataset has {0} instances and {1} columns.'.format(*data.shape))
    print('It has {} duplicated entries.'.format(data.duplicated().sum()))
    print('\nColumn names:\n{}'.format(data.columns))
    print('\nMissing values:\n{}'.format(data.isnull().sum()))

    return data



def plot_stacked_bar(data, var, by_var, sort=True):

    """Plot stacked bar chart.

    Args:
    var: variable name to plot
    by_var: variable name to group by
    sort: whether to sort the bar by value, default to true

    Returns:
    A stacked bar chart with kiva color palette.
    """

    # Define custom palette that aligns with Kiva's corp color book
    kiva_pal = ["#549E39", "#8AB833", "#C0CF3A", "#029676", "#A6A6A6"]

    order = data[var].value_counts().index
    data = data.groupby([var, by_var]).size()

    if sort:
        data = data.reindex(index=order, level=0).unstack()
    else:
        data = data.unstack()

    data.plot.bar(stacked=True, color=kiva_pal[-1:-6:-4])
