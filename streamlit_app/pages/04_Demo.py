import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import config
from models.models import models, acc_yes_nop
import utils.scraping as scrap
import utils.tween as tw
import utils.layout as layout

# base stylesheet
layout.base_styles()

# Init stuff
if 'last_acc' not in st.session_state:
    st.session_state.last_acc = 0
    st.session_state.last_bac = 0
    st.session_state.last_yep = 0
    st.session_state.last_nop = 0

# We start today !
today = datetime.date.today()

# bom.gov has only 14 months before today data
print(today.day)
predictable_month = {d.strftime('%b %Y'): d for d in [today - relativedelta(months=i) for i in range(1, 15)]}

# WIP: handle total predictions in session
# st.write(st.session_state.disabled)

# Sidebar: selection for location, date and bom accuracy
# TODO: add boom accuracy selection
location_label = "Choose a location"
tomorrow_label = "Today"
with st.sidebar:
    location_selected = st.selectbox("", [location_label, "All locations"] + sorted(scrap.locations['LocationName']), label_visibility='collapsed')
    date_selected = option_menu(None, [tomorrow_label] + [d for d in predictable_month.keys()],
        icons=[],
        menu_icon="cast",
        orientation='horizontal',
        styles={
            'nav':{'font-size':'15px', 'display':'grid', 'grid-template-columns': 'repeat(3, 1fr)'},
            'nav-item':{'padding': '0px 0px', 'display': 'flex', 'align-items': 'center','font-weight':'normal'},
            # 'nav-link': {'font-weight':'normal'},
            'icon':{'display':'none'}
        })

# layout.show_members()

# Background placeholder
bg_placeholder = st.empty()

welcome = st.empty()

# Header columns
header = st.container()

# Metrics columns
scores_cols = st.columns(3)

date = predictable_month.get(date_selected) if date_selected != tomorrow_label else today

# TODO: if Today, we predict for one day vs BoM forecast

# Trigger predictions
if location_selected != location_label:

    start_time = time.time()
    predict_today = date == today
    predict_alloc = location_selected == 'All locations'

    with header:
        header_cols = st.columns([1,1])
        header_col1 = header_cols[1].empty()

    with header_cols[0]:
        st.write(f'<h2 style="text-align:right">{location_selected}</h2>', unsafe_allow_html=True)
        if predict_today:
            st.write(f'<h4 style="text-align:right">Will it rain tomorrow ?</h4>', unsafe_allow_html=True)
        else:
            st.write(f'<h4 style="text-align:right">Next day rain for {date.strftime("%B %Y")}</h4>', unsafe_allow_html=True)

    if predict_alloc:
        with header_col1.container():
            st.write('<span class="spinner"><i class="spin"></i><span class="label">Loading</span></span>', unsafe_allow_html=True)
            load_state = st.empty()
            st.write("""<style>.stAlert{display: inline-block; position: absolute;}</style>""",
                unsafe_allow_html=True)
    else:
        header_col1.write('<span class="spinner"><i class="spin"></i><span class="label">Loading</span></span>', unsafe_allow_html=True)

    st_month = date.month if date.month >= 10 else '0' + str(date.month)

    # Display maps

    if predict_alloc:
        lat = -28.274398
        lon = 133.775136
        col = 'FFF' if predict_today else '000'
        pin = ["pin-s+%s(%f,%f)" % (col,loc.LON, loc.LAT) for _, loc in scrap.locations.iterrows()]
        pin = ','.join(pin)
        zoom = '3.6,0,0'
    else:
        location = scrap.get_location(location_selected)
        lat = location['LAT']
        lon = location['LON']
        pin = f'pin-s+000({lon},{lat})'
        zoom = 5

    map_background = f'https://api.mapbox.com/styles/v1/mapbox/light-v11/static/{pin}/{lon},{lat},{zoom}/1085x726?access_token={config.MAPBOX_TOKEN}'

    if predict_today or predict_alloc:
        bg_placeholder.write(""" 
        <style>
        section.main {
            background-image: linear-gradient(to bottom, #FFF 10%%, rgba(255,255,255,0) 90%%),
                url(''), url(''), url('%s');
        }
        </style>
        """ % (map_background), unsafe_allow_html=True)
    else:
        bom_background = f'http://www.bom.gov.au/climate/ahead/outlooks/skill/plots/rain.skill.wpc.median.monthly.ini25.lt1.{st_month}.aus.hr.png'
        bg_placeholder.write(""" 
        <style>
        section.main {
            background-image: linear-gradient(to bottom, #FFF 10%%, rgba(255,255,255,0) 90%%),
                url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="1085" height="726"><circle cx="%d" cy="%d" r="25" fill="white" stroke="black" stroke-width="4"/></svg>'),
                url('%s'), url('%s');
        }
        </style>
        """ % (location['Bomapos'][0], location['Bomapos'][1], bom_background, map_background), unsafe_allow_html=True)
        # Need only for accuracy comparison
        bom_acc = scrap.compute_bom_accuracy(bom_background, location['Location'])

    if predict_alloc:
        location = scrap.locations.index
        data = pd.DataFrame()
        # batch_obs = monthrange(date.year, date.month)[1]
        # total_obs = batch_obs * len(location)
        with load_state:
            for i, loc in enumerate(location):
                load = scrap.by_location(loc, date.year, date.month)
                if predict_today:
                    load = load.tail(1)
                data = pd.concat([data, load])
                # loaded = (i + 1) * batch_obs
                st.info("%d / %d loaded" % (i + 1, len(location)))
    else:
        data = scrap.by_location(location['Location'], date.year, date.month)

    if data is None:
        st.warning("""Sorry, no data found for %s on %s. Please try another date or location.
        """ % (location['Location'], date.strftime("%B %Y")))
        header_col1.write('')
    else:

        # rest_time = 2 - (time.time() - start_time)
        # if  rest_time > 0:
        #     time.sleep(rest_time)

        model = models[0]
    
        if predict_alloc:
            zone = 'All climates'
            kcod = 'A, B, C'
        else:
            zone = location['Climate']
            kcod = location['KCode'][0]

        header_col1.write('<div class="model-info"><span>%s</span><span>%s model %s</span></<div>' % (zone, model.name, kcod),
                            unsafe_allow_html=True)

        # Switch tomorrow / all|single location
        if predict_today:
            if not predict_alloc:
                data = data.tail(1)
                pred = model.predict_one(data)
                st.write("""<h2 class="pred %s">%s
                <span class="msg">...but we have to wait tomorrow to see if it's true</span></h2>
                """ % (pred['Pred'], pred['Pred']), unsafe_allow_html=True)
            else:
                data = data.reset_index(drop=True)
                pins = []
                yes = 0
                nop = 0
                for _, row in data.iterrows():
                    loca = scrap.get_location(row['Location'])
                    pred = model.predict_one(row.to_frame().T)
                    zone = loca['KCode'][0].lower()
                    if pred['Pred'] == 'Yes':
                        yes+= 1
                        # hexa = 'b6b3e1'
                        hexa = '8ebcdf'
                    else:
                        nop+= 1
                        # hexa = 'ecbe74'
                        hexa = 'f9d656'
                    pins.append("pin-s-%s+%s(%f,%f)" % (zone, hexa, loca['LON'], loca['LAT']))

                lat = -28.274398
                lon = 133.775136
                pin = ','.join(pins)
                zoom = '3.6,0,0'

                map_background = f'https://api.mapbox.com/styles/v1/mapbox/light-v11/static/{pin}/{lon},{lat},{zoom}/1085x726?access_token={config.MAPBOX_TOKEN}'
                bg_placeholder.write(""" 
                <style>
                section.main {
                    background-image: linear-gradient(to bottom, #FFF 10%%, rgba(255,255,255,0) 90%%),
                        url(''), url(''), url('%s');
                }
                </style>
                """ % (map_background), unsafe_allow_html=True)

                cols = scores_cols
                col1 = cols[0].empty()
                col2 = cols[1].empty()
                col3 = cols[2].empty()

                yes = yes / len(data) * 100
                nop = nop / len(data) * 100

                for t in tw.tween(.8):
                    # col1.metric(label='Accuracy',
                    #             value="%.1f%%" % (lac + (acc - lac) * t),
                    #             )
                    col2.metric(label="Yes",
                                value="%.1f%%" % (yes * t),
                                )
                    col3.metric(label="No",
                                value="%.1f%%" % (nop * t),
                                )

        elif predict_alloc:
            start_time = time.time()
            predsA = model.predict(data[data.KCode.str[0] == 'A'])
            predsB = model.predict(data[data.KCode.str[0] == 'B'])
            predsC = model.predict(data[data.KCode.str[0] == 'C'])
            acc, yes, nop = acc_yes_nop(pd.concat([predsA, predsB, predsC]))

            st.write("`%d` predictions in `%.2fs`" % (len(data), time.time() - start_time))

            lac = st.session_state.last_acc
            lye = st.session_state.last_yep
            lno = st.session_state.last_nop

            # Prep scores columns
            cols = scores_cols
            col1 = cols[0].empty()
            col2 = cols[1].empty()
            col3 = cols[2].empty()

            for t in tw.tween(.8):
                col1.metric(label='Accuracy',
                            value="%.1f%%" % (lac + (acc - lac) * t),
                            )
                col2.metric(label="Yes",
                            value="%.1f%%" % (lye + (yes - lye) * t),
                            )
                col3.metric(label="No",
                            value="%.1f%%" % (lno + (nop - lno) * t),
                            )

            st.session_state.last_acc = acc
            st.session_state.last_yep = yes
            st.session_state.last_nop = nop
        else:
            preds = model.predict(data)
            acc, yes, nop = acc_yes_nop(preds)
            bac = acc - bom_acc

            lac = st.session_state.last_acc
            lba = st.session_state.last_bac
            lye = st.session_state.last_yep
            lno = st.session_state.last_nop

            # Prep scores columns
            cols = scores_cols
            col1 = cols[0].empty()
            col2 = cols[1].empty()
            col3 = cols[2].empty()

            for t in tw.tween(.8):
                col1.metric(label='Accuracy',
                            value="%.1f%%" % (lac + (acc - lac) * t),
                            delta="%.1f%%" % (lba + (bac - lba) * t))
                col2.metric(label="Yes",
                            value="%.1f%%" % (lye + (yes - lye) * t),
                            )
                col3.metric(label="No",
                            value="%.1f%%" % (lno + (nop - lno) * t),
                            )

            st.session_state.last_acc = acc
            st.session_state.last_bac = bac
            st.session_state.last_yep = yes
            st.session_state.last_nop = nop

# No location selected display help
else:
    lat = -25.274398
    lon = 133.775136
    zoom = 4
    map_background = f'https://api.mapbox.com/styles/v1/mapbox/light-v11/static/{lon},{lat},{zoom}/1085x726?access_token={config.MAPBOX_TOKEN}'
    
    bg_placeholder.write(""" 
    <style>
    section.main {
        background-image: linear-gradient(to bottom, #FFF 10%%, rgba(255,255,255,0) 90%%),
            url(''), url(''), url('%s');
    }
    </style>
    """ % (map_background), unsafe_allow_html=True)

    with welcome.container():
        st.write("""
        <style>
        .demo-modal {
            background: rgba(255,255,255,.7);
            padding: 20px 40px 10px 40px;
            box-shadow: 0px 0px 10px rgba(0,0,0,.2);
            /*margin-top: -80px;*/
        }
        </style>
        <div class="demo-modal">
            <h3>Welcome to the demo</h3>
            <p>From here you should be able to test our model at predicting rain tomorrow in Australia for the past 14 months. See the screenshot bellow for an overview of the display.</p>
            <p>To start, choose a location from the left and <i>Happy predictions !</i></p>
        </div>
        """, unsafe_allow_html=True)

        st.image('assets/demo-screenshot.png')
