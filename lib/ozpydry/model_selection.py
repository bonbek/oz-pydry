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
from sklearn.preprocessing import StandardScaler
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

def sample_tts(resample=None, dt=False, drop=[], encode='onehot', na='drop',
               prior=None, scale=False, test_size=.2, verbose=False):
    """Samples random train and test subsets.

    Wraps `train_test_split` on main dataset cleaned from NAs to apply variables
    selection, resampling and a subset of preprocessing.
    Note that classes of target's variable beings binary encoded and the *Date*
    variable is droped.

    Parameters
    ----------
    resample : {'under', 'over'}, optional
        Resample according to RandomUnderSampler or RandomOverSampler.

    dt : bool or string, optional
        When True, keep original *Date* variable.
        When string, wether to attach date's components to observations according
        to the following (only one format per component allowed, ex: 'Yma').
        Note that *Date* variable is droped when a date's component is attached.

        'Y' : year with century (Year var)
        'm' : month as decimal number  [1,12] (Month var)
        'B' : full month name (Month var)
        'd' : day of the month as a decimal number [01,31] (Day var)
        'w' : weekday as a decimal number [0,6] (Day var)
        'a' : abbreviated weekday name (Day var)

    drop : list, optional
        The list of variables to drop from the set.

    encode: {'onehot', 'discrete'}, optional
        Encoding method for categorical variables.

        'onehot' : applies `get_dummies`
        'discrete' : converts to int labels in range[1,n]

    na : {'keep', 'drop', 'fill'}, optional
        Keep, drops or fill NAs. Categorical are filled with the variable's modes
        and numerical with their means.

    prior : callable, optional
        A callable applied before any preprocessing that take the DataFrame
        and return a Dataframe.

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

    data = df

    if callable(prior): # Apply prior hook
        data = prior(data)

    # Attach Date components
    if isinstance(dt, str):
        if 'Y' in dt:
            data = data.assign(Year=df.Date.dt.year)
        if 'm' in dt:
            data = data.assign(Month=df.Date.dt.month)
        if 'B' in dt:
            data = data.assign(Month=df.Date.dt.month_name())
        if 'd' in dt:
            data = data.assign(Day=df.Date.dt.day)
        if 'w' in dt:
            data = data.assign(Day=df.Date.dt.weekday)
        if 'a' in dt:
            data = data.assign(Day=df.Date.dt.day_name())
        data = data.drop(columns=['Date'])
    elif not dt:
        data = data.drop(columns=['Date'])

    # Time to drop before fill na
    data = data.drop(columns=drop)

    if na == 'drop':
        data = data.dropna()
    elif na == 'fill':
        nums = data.select_dtypes('number')
        data[nums.columns] = nums.fillna(nums.mean())
        cats = data.select_dtypes('O')
        data[cats.columns] = cats.fillna(cats.mode())

    # Prevent trailing NAs error
    data = data.dropna(subset=['RainTomorrow', 'RainToday'])
    data = data.replace(['Yes', 'No'], [1, 0])
    targ = data.RainTomorrow
    data = data.drop(columns=['RainTomorrow'])

    # Categorical vars encoding
    if encode is not None:
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

    # 'Normalize' data
    if scale:
        sc = StandardScaler().fit(Xs)
        Xs.iloc[:,:] = sc.transform(Xs)
        Xt.iloc[:,:] = sc.transform(Xt)

    # Print sampling info
    if verbose:
        tact = ys.value_counts(normalize=True).round(4) * 100
        md("_Taking __%d__ samples of __%d__ features, target balance %.f%% / %.f%%_" %(*Xs.shape, tact[1], tact[0]))

    return Xs, Xt, ys, yt