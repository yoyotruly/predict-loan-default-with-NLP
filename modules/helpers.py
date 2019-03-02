import re
import pandas as pd
import unidecode
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def preproc_text1(text):

    """Initial text cleaning.

    Convert text to all lowercase and strip it of unwanted HTML tags and content, unwanted REGEX patterns, natural line breaks and non-unicode characters.

    Args:
    text: a string.

    Returns:
    A cleaned text string.
    """

    bad_tags = ['i', 'h4', 'b']
    bad_regex_list = ['translated[^\.]+\.',
                      'previous (profile|loan)[^\.]+',
                      'http\S+',
                      'www\S+',
                      'mifex offers its clients[^\.]+\.',
                      'for more information[^\<]+']
    bad_regex = re.compile('|'.join(bad_regex_list))

    # remove unwanted html content contained in BAD_TAGS
    soup = BeautifulSoup(text, 'lxml')
    content_to_remove = [s.get_text() for s in soup.find_all(bad_tags)]
    if content_to_remove:
        text = ''.join([text.replace(c, "") for c in content_to_remove])
    else:
        text = text

    text = text.lower()
    text = BeautifulSoup(text, 'lxml').text  # remove html tags
    text = bad_regex.sub("", text) # remove unwanted REGEX patterns
    text = ' '.join(text.splitlines())  # remove natural line breaks
    text = unidecode.unidecode(text)  # remove non-English characters

    return text


def preproc_text2(text):

    """Further text cleaning.

    Strip text of punctuations, numbers and customized stopwords, then lemmatize nouns and verbs.

    Args:
    text: a string.

    Returns:
    A cleaned string.
    """

    stop_words = stopwords.words('english') + stopwords.words('spanish')
    stop_words.extend(['loan', 'also', 'kiva', 'am', 'pm'])

    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
    text = re.sub(r'\d+', '', text)  # remove numbers
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [WordNetLemmatizer().lemmatize(token, pos='n') for token in tokens]
    tokens = [WordNetLemmatizer().lemmatize(token, pos='v') for token in tokens]
    text = ' '.join(tokens)

    return text
