"""Model selection utilities.

Synthetizes common data load and preprocessing to alleviate exploration and
modeling. Choices made cannot fit all needs.

Examples
--------
Suppose we want a split with climate and month attached to observations (and 
by default one hot encoded categorical variables).

>>> from ozpydry.model_selection import load_df, sample_tts
>>> load_df(extras='climate')
>>> X_train, X_test, y_train, y_test = sample_tts(dt='month')

Note that all split after calling `load_df` with the *extras* parameter will
contains the given extras (climate in our example). So can get another split
with balanced target (undersampled) as this :

>>> X_train, X_test, y_train, y_test = sample_tts(resample='under', dt='month')
"""

import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from .display import md

# random state
seed = 756

df = None
df_locations = None

def load_location_extras():
    """Load additionnal locations informations.

    You may don't have to call this function directly since this is done for you
    when you apply `load_df` with extras parameter.

    Returns
    -------
    DataFrame
    """
    global df_locations

    ozshape = gpd.read_file("../data/geo/STE_2021_AUST_SHP_GDA94/STE_2021_AUST_GDA94.shp")
    ozshape.to_crs(epsg=4326, inplace=True)

    # Climate geodata
    climate = gpd.read_file('../data/geo/other_climate_2007_koppen_geiger/other_climate_2007_koppen_geiger.shp')
    climate.to_crs(epsg=4326, inplace=True)
    climate = climate.sjoin(ozshape).drop_duplicates(subset='identity')
    
    # Merge cleaned Koppen Geiger data with its geodata
    tmp = pd.read_csv("../data/koppen-table.csv", sep=";")
    climate = climate.assign(KCode=climate.climate.str.split(" ").str[0])
    climate = climate.merge(tmp, on='KCode')

    # Load extended locations and and merge climate
    locations = gpd.read_file('../data/locations-regions.csv')
    locations['geometry'] = gpd.GeoSeries([Point((l.Lon, l.Lat)) for i, l in locations.iterrows()], crs=4326)
    climate = climate[['geometry', 'EN', 'KCode', 'Color']].rename({'EN':'Climate'}, axis=1).to_crs(epsg=3035)
    df_locations = locations.to_crs(epsg=3035).sjoin_nearest(climate).to_crs(epsg=4326).drop(columns=['index_right', 'geometry'])
    return df_locations

def load_df(extras=None, path="../data/weatherAUS.csv"):
    """Load main dataset.

    Parameters
    ----------
    extras : {'climate', 'region', 'full'}, optional
        Whether to attach additionals informations to observations.

        - 'climate': climate's Koppen code is attached (KCode var)
        - 'region': observation's region is attached (Region var)
        - 'full: the two above plus climate Koppen description (Climate var)

        .. note:: first call with extras parameter may take some time due to
                  geodata precessing.

    path : string, optional
        File path of a specific dataset to load

    Returns
    -------
    DataFrame
    """
    global df, df_locations

    df = pd.read_csv(path, parse_dates=['Date'])

    if extras in ['climate', 'region', 'full']:
        if df_locations is None: # data cached by load_location_extras
            load_location_extras()

        sel = ['Region', 'Climate', 'KCode'] if extras == 'full' else  ['Region'] \
                    if extras == 'region' else ['KCode']
        df = df.merge(df_locations[['Location', *sel]], on='Location')

    return df

def sample_tts(resample=None, dt=False, drop=[], encode='onehot', test_size=.2, verbose=False):
    """Samples random train and test subsets.

    Wraps `train_test_split` on main dataset cleaned from NAs to apply variables
    selection, resampling and a subset of preprocessing.
    Note that classes of target's variable beings binary encoded and the *Date*
    variable is droped.

    Parameters
    ----------
    resample : {'under', 'over'}, optional
        Resample according to RandomUnderSampler or RandomOverSampler.

    dt : {'month'}, optional
        Whether to attach observation's month (Month var).

    drop : list, optional
        The list of variables to drop from the set.

    encode: {'onehot', 'discrete'}, optional
        Encoding method for categorical variables.

        'onehot' : applies `get_dummies`
        'discrete' : converts to int labels in range[1,n]

    test_size : float or int, optional
        see `train_test_split`.

    verbose : bool, optional
        Whether to print statistics : number of samples in train and test subset,
        target's classe balance.

    Returns
    -------
    List containing train-test split of inputs.
    """

    if df is None:
        load_df()

    csel = list(set(df.columns).difference(drop))
    data = df[csel].dropna()

    if dt and 'm' in dt:
        data['Month'] = df.Date.dt.month

    data = data.replace(['Yes', 'No'], [1, 0])
    targ = data.RainTomorrow
    data = data.drop(columns=['RainTomorrow', 'Date'])

    # Categorical vars encoding
    if encode == 'onehot':
        data = pd.get_dummies(data)
    elif encode == 'discrete':
        repl = data.select_dtypes('O').apply(pd.unique)
        data = data.replace(repl.to_dict(),
                            repl.apply(lambda x: np.arange(1, len(x) + 1)).to_dict())

    # Split train/test & resample (train only)
    Xs, Xt, ys, yt = train_test_split(data, targ, test_size=test_size,
                                      random_state=seed, stratify=targ)
    if resample:
        if resample == 'under':
            Xs, ys = RandomUnderSampler(random_state=seed).fit_resample(Xs, ys)
        else:
            Xs, ys = RandomOverSampler(random_state=seed).fit_resample(Xs, ys)
    
    # Print sampling info
    if verbose:
        tact = ys.value_counts(normalize=True).round(4) * 100
        md("_Taking __%d__ samples of __%d__ features, target balance %.f%% / %.f%%_" %(*Xs.shape, tact[1], tact[0]))

    return Xs, Xt, ys, yt