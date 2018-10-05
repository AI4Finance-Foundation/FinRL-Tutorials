import numpy as np
import pandas as pd
import sys
from collections import deque
# data scraping packages
from bs4 import BeautifulSoup
# satisfy compatilibty between python2 and python3
from six.moves.urllib.request import urlopen
version = int(sys.version[0])
# the symbols of S&P500 and S&P 100 are ^GSPC and ^OEX
def get_sap_symbols(name='sap500'):
    """Get ticker symbols constituting S&P
    
    Args:
        name(str): should be 'sap500' or 'sap100'
    """
    if name == 'sap500':
        site = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    elif name == 'sap100':
        site = 'https://en.wikipedia.org/wiki/S%26P_100'
    else:
        raise NameError('invalid input: name should be "sap500" or "sap100"')
    # fetch data from yahoo finance
    page = urlopen(site)
    soup = BeautifulSoup(page, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            symbol = col[0].string.replace('.', '-')
            symbols.append(str(symbol))
    return symbols

def date_parse(date):
    """split time data into the list of ingegers
    Args:
        date: 'y-m-d', e.g.'2015-04-06'
    Return:
        List(int)
    """
    parsed = date.split('-')
    converted = [int(t) for t in parsed]
    return converted

def get_data(symbol, st, end):
    """Get historical data from yahoo-finance as pd.DataFrame
    
    Args:
        symbol(str): ticker symbol, e.g. 'AFL'
        st, end: start and end date for data, e.g. '2015-04-06'
    Return:
        DataFrame
    """
    # split date into integers
    ys, ms, ds = date_parse(st)
    ye, me, de = date_parse(end)
    # fetch data from yahoo finance
    url = urlopen('http://chart.finance.yahoo.com/table.csv?s=%s&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=d&ignore=.csv' \
                           % (symbol, ms, ds, ys, me, de, ye))
    history = url.read()
    if version == 3:
        history = str(history).split('\\n')
        # get rid of 'b'
        keys = history[0].split("'")[1]
    else:
        history = str(history).split('\n')
        keys = history[0]
    keys = keys.split(',')[1:]
    # convert fetched data into the DataFrame
    values = deque()
    dates = deque()
    for x in history[1:-1]:
        x = x.split(',')
        dates.appendleft(pd.Timestamp(x[0]))
        values.appendleft([float(value) for value in x[1:]])
    return pd.DataFrame(list(values), columns=keys, index=list(dates))

def get_data_list_full(symbols, st, end):
    """Get all of symbols's data as list"""
    data = []
    for s in symbols:
        data.append(get_data(s, st, end))
    return data

def get_data_list_key(symbols, st, end, key='Open'):
    """Get historical data of key attribute
    from yahoo-finance as pd.DataFrame
    
    Args:
        symbols: list of ticker symbols, e.g. ['AFL', 'AAPL', ....]
        st, end: start and end date for data, e.g. '2015-04-06'
        key: attribute of data, e.g. Open, Close, Volume, Adj Close...
    Return:
        DataFrame
    """
    values= []
    sucess_symbols = []
    fail_symbols = []
    # length of data 
    max_len = 0
    for s in symbols:
        try:
            x = get_data(s, st, end)
            n_data = len(x.index)
            if n_data >= max_len:
                if n_data != max_len:
                    max_len = n_data
                    date = x.index
                    fail_symbols += sucess_symbols
                    values= []
                    sucess_symbols = []
                values.append(x[key])
                sucess_symbols.append(s)
            else:
                fail_symbols.append(s)            
        except:
            fail_symbols.append(s)
            pass
    if len(fail_symbols) > 0:
        print('we cound not fetch data from the following companies')
        print(fail_symbols)
    return pd.DataFrame(np.array(values).T, index = date, columns=sucess_symbols)

def testscore(prediction, target):
    return np.mean(np.abs(prediction - target) / target)
