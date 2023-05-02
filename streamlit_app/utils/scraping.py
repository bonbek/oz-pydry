import streamlit as st
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from PIL import Image

bommapos_ = {
    'Adelaide':                     [570, 510],
    'Albany':                       [224, 516],
    'Albury':                       [700, 533],
    'AliceSprings':                 [494, 280],
    'BadgerysCreek':                [790, 472],
    'Ballarat':                     [648, 560],
    'Bendigo':                      [657, 547],
    'Brisbane':                     [821, 377],
    'Cairns':                       [710, 155],
    'Canberra':                     [737, 526],
    'Cobar':                        [702, 433],
    'CoffsHarbour':                 [817, 432],
    'Dartmoor':                     [606, 568],
    'Darwin':                       [431, 56],
    'GoldCoast':                    [827, 392],
    'Hobart':                       [694, 673],
    'Katherine':                    [453, 93],
    'Launceston':                   [694, 649],
    'Melbourne':                    [666, 569],
    'MelbourneAirport':             [666, 569],
    'Mildura':                      [623, 495],
    'Moree':                        [763, 409],
    'MountGambier':                 [599, 563],
    'MountGinini':                  [730, 531],
    'Newcastle':                    [789, 481],
    'Nhil':                         [611, 540],
    'NorahHead':                    [779, 491],
    'NorfolkIsland':                None,
    'Nuriootpa':                    [572, 497],
    'PearceRAAF':                   [183, 464],
    'Penrith':                      [772, 498],
    'Perth':                        [184, 471],
    'PerthAirport':                 [184, 471],
    'Portland':                     [609, 574],
    'Richmond':                     [773, 495],
    'Sale':                         [700, 580],
    'SalmonGums':                   [278, 472],
    'Sydney':                       [772, 500],
    'SydneyAirport':                [772, 500],
    'Townsville':                   [728, 206],
    'Tuggeranong':                  [736, 527],
    'Uluru':                        [433, 314],
    'WaggaWagga':                   [713, 516],
    'Walpole':                      [204, 523],
    'Watsonia':                     [669, 569],
    'Williamtown':                  [788, 481],
    'Witchcliffe':                  [177, 512],
    'Wollongong':                   [767, 510],
    'Woomera':                      [531, 424]
}

stations_ = {
    'Albury':'IDCJDW2002',
    'CoffsHarbour':'IDCJDW2030',
    'BadgerysCreek':'IDCJDW2062',
    'Newcastle':'IDCJDW2097',
    'Moree':'IDCJDW2084',
    'NorfolkIsland':'IDCJDW2100',
    'NorahHead':'IDCJDW2099',
    'Richmond':'IDCJDW2119',
    'Penrith':'IDCJDW2111',
    'WaggaWagga':'IDCJDW2139',
    'SydneyAirport':'IDCJDW2125',
    'Wollongong':'IDCJDW2014',
    'Williamtown':'IDCJDW2145',
    'GoldCoast':'IDCJDW4050',
    'Brisbane':'IDCJDW4019',
    'Sydney':'IDCJDW2124',
    'Cobar':'IDCJDW2029',
    'Tuggeranong':'IDCJDW2802',
    'Canberra':'IDCJDW2801',
    'Ballarat':'IDCJDW3005',
    'MountGinini':'IDCJDW2804',
    'Sale':'IDCJDW3021',
    'Bendigo':'IDCJDW3008',
    'Melbourne':'IDCJDW3050',
    'MelbourneAirport':'IDCJDW3049',
    'Hobart':'IDCJDW7021',
    'Watsonia':'IDCJDW3079',
    'Mildura':'IDCJDW3051',
    'Launceston':'IDCJDW7025',
    'SalmonGums':'IDCJDW6119',
    'Nhil':'IDCJDW3059',
    'Dartmoor':'IDCJDW3101',
    'Portland':'IDCJDW2110',
    'MountGambier':'IDCJDW5041',
    'Adelaide':'IDCJDW5081',
    'Albany':'IDCJDW6001',
    'Nuriootpa':'IDCJDW5049',
    'Walpole':'IDCJDW6138',
    'Witchcliffe':'IDCJDW6081',
    'Townsville':'IDCJDW4128',
    'Cairns':'IDCJDW4154',
    'Katherine':'IDCJDW8048',
    'Darwin':'IDCJDW8014',
    'AliceSprings':'IDCJDW8002',
    'Woomera':'IDCJDW5072',
    'PearceRAAF':'IDCJDW6108',
    'Uluru':'IDCJDW8056',
    'Perth':'IDCJDW2166',
    'PerthAirport':'IDCJDW6110',
}

# Augment location dataset with station ids and bom map coords
locations = pd.read_csv('../data/locations-full.csv')
locations = locations.rename(columns={'Lat':'LAT', 'Lon':'LON'})
locations = locations.set_index('Location')
locations = locations.join(pd.Series(stations_, name="Station"))
locations = locations.join(pd.Series(bommapos_, name="Bomapos"))
# Monkey patch
locations.dropna(subset='Bomapos', inplace=True)

headers_ = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

def get_location(location):
    """Returns location data as a Series"""
    location = locations[(locations.index == location) | (locations.LocationName == location)]
    return location.reset_index().loc[0]

@st.cache_data(show_spinner=False)
def by_location(location, year, month):
    """Load samples for a given location from BoM website.

    .. note:: only past 14 month are available

    Parameters
    ----------
    location : string
        The location name as in locations dataset `Location`or `LocationName`
    
    year : int
        Year of the samples
    
    month : int
        Month of the samples

    Returns
    -------
    pandas.DataFrame
    """
    location = locations[(locations.index == location) | (locations.LocationName == location)]
    if (len(location) == 0):
        raise Exception("Sorry, %s not found in locations" % location)

    location = location.index[0]
    sid = locations.at[location, 'Station']
    month = '0' + str(month) if month < 10 else str(month)
    url = f"http://www.bom.gov.au/climate/dwo/{year}{month}/text/{sid}.{year}{month}.csv"

    with requests.Session() as s:
        req = s.get(url, headers=headers_)
        if req.status_code != 200:
            return None

        cols_order = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow', 'KCode']
        cols_scrap = ['TMP0', 'Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'TMP1', 'Temp9am', 'Humidity9am', 'Cloud9am', 'WindDir9am', 'WindSpeed9am', 'Pressure9am', 'Temp3pm', 'Humidity3pm', 'Cloud3pm', 'WindDir3pm', 'WindSpeed3pm', 'Pressure3pm']

        csv = BytesIO(req.content)
        while csv.readline().decode('ISO-8859-1').find(',"Date"') == -1:
            pass # seek to first data line (not all csv are not equaly formated)
        tdf = pd.read_csv(csv, encoding='ISO-8859-1', header=None, names=cols_scrap)

        rtd = tdf.Rainfall.fillna(0)
        rtd = tdf.Rainfall.apply(lambda x: 'Yes' if x >= 1 else 'No')
        kc = locations.at[location, 'KCode']
        tdf = tdf.assign(Location=location, KCode=kc, RainToday=rtd, RainTomorrow=rtd.shift(-1, fill_value=np.nan))
        # special cases
        tdf[['WindSpeed9am', 'WindSpeed3pm']] = tdf[['WindSpeed9am', 'WindSpeed3pm']].replace(['Calm'], [0]).astype(float)
        return tdf[cols_order]

# Scores and colors correspondence of BoM map
bomap_colors = ['2E8421', '47AA22', '55C127', 'BFDE14', 'FCE6D0', 'FEFEFE']
bomap_colors = np.array([[int(hex[i:i+2], 16) for i in (0,2,4)] for hex in bomap_colors])
bomap_scores = [100, 75, 65, 55, 50, 45]

@st.cache_data(show_spinner=False)
def compute_bom_accuracy(map_url, location):
    res = requests.get(map_url, headers=headers_)
    img = Image.open(BytesIO(res.content))
    px, py = bommapos_.get(location, [0, 0])

    pick = np.array(img.getpixel((px,py)))
    knn = [np.abs(pick - c) for c in bomap_colors]
    idx_scores = np.unique(np.argmin(np.array(knn), axis=0))

    if (len(idx_scores) > 1):
        s1 = bomap_scores[idx_scores[0]]
        s2 = bomap_scores[idx_scores[1]]
        rx = (idx_scores[1] - idx_scores[0]) / (s1 - s2)
        ofs = np.abs(knn[idx_scores[0]] - knn[idx_scores[1]]).sum() * rx
        print("ofs", ofs)
        print("f score", s1 - ofs)

    knn = [c.sum() for c in knn]
    a1 = np.argmin(knn)
    return bomap_scores[a1]
    # st.write(bomap_scores[a1])
    # if 's1' in locals():
        # st.write("%d" % int(s1 - ofs))