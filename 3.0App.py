import streamlit as st

import os
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly as plty
import seaborn as sns
import country_converter as coco
from bokeh.io import output_file, show, output_notebook, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Viridis as palette
from bokeh.transform import factor_cmap

import warnings

warnings.filterwarnings("ignore")


# ======================================================================== Load and Cache the data
# @st.cache(persist=True)
def getdata():
    WORLD_CONFIRMED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    WORLD_DEATHS_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    WORLD_RECOVERED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

    world_confirmed = pd.read_csv(WORLD_CONFIRMED_URL)
    world_deaths = pd.read_csv(WORLD_DEATHS_URL)
    world_recovered = pd.read_csv(WORLD_RECOVERED_URL)

    sets = [world_confirmed, world_deaths, world_recovered]

    # yesterday's date
    yesterday = pd.to_datetime(world_confirmed.columns[-1]).date()
    today_date = str(pd.to_datetime(yesterday).date() + datetime.timedelta(days=1))

    return (yesterday, today_date, sets)


yesterday, today_date, sets = getdata()[0], getdata()[1], getdata()[2]


# =========================================================================================  clean

def drop_neg(df):
    # Drop negative entries entries
    idx_l = df[df.iloc[:, -1] < 0].index.tolist()
    for i in idx_l:
        df.drop([i], inplace=True)
    return df.reset_index(drop=True)


sets = [drop_neg(i) for i in sets]

for i in range(3):
    sets[i].rename(columns={'Country/Region': 'Country', 'Province/State': 'State'}, inplace=True)
    sets[i][['State']] = sets[i][['State']].fillna('')
    sets[i].fillna(0, inplace=True)
    # Change dates to datetime format
    sets[i].columns = sets[i].columns[:4].tolist() + [pd.to_datetime(sets[i].columns[j]).date()
                                                      for j in range(4, len(sets[i].columns))]

sets_grouped = []
cases = ['confirmed cases', 'deaths', 'recovered cases']
for i in range(3):
    o = sets[i].groupby('Country').sum()
    o.rename(index={'US': 'United States'}, inplace=True)
    sets_grouped.append(o)

# get continent names
for df in sets_grouped:
    continent = coco.convert(names=df.index.tolist(), to='Continent')
    df['Continent'] = continent


# =========================================================================================  top countries

def bokehB(dataF, case):
    # Bokeh bar plots. The function takes a dataframe, datF, as the one provided by the raw data
    # (dates as columns, countries as rows). It first takes the last column as yesterday's date.

    df = dataF.iloc[:, -2:].sort_values(by=dataF.columns[-2], ascending=False).head(20)
    df['totals'] = df.iloc[:, 0]
    df.drop(df.columns[0], axis=1, inplace=True)
    cont_cat = len(df['Continent'].unique())

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    source = ColumnDataSource(df)

    select_tools = ['save']
    tooltips = [
        ('Country', '@Country'), ('Totals', '@totals{0,000}')
    ]

    p = figure(x_range=df.index.tolist(), plot_width=840, plot_height=600,
               x_axis_label='Country',
               y_axis_label='Totals',
               title="Top Countries with {} as of ".format(case) + today_date,
               tools=select_tools)

    p.vbar(x='Country', top='totals', width=0.9, alpha=0.7, source=source,
           legend_field="Continent",
           color=factor_cmap('Continent', palette=palette[cont_cat], factors=df.Continent.unique()))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = 1
    p.left[0].formatter.use_scientific = False

    p.add_tools(HoverTool(tooltips=tooltips))

    return p


def bokehB_mort(num=100):
    # Bokeh bar plots. The function already includes the confirmed and deaths dataframes,
    # and operates over them to calculate th mortality rate depending on num (number of
    # minimum deaths to consider for a country). The rest is equivalent to the BokehB()
    # function.

    # top countries by deaths rate with at least num deaths
    top_death = sets_grouped[1][yesterday].sort_values(ascending=False)
    top_death = top_death[top_death > num]

    # Inner join to the confirmed set, compute mortality rate and take top 20
    df_mort = pd.concat([sets_grouped[0][yesterday], top_death], axis=1, join='inner')
    mort_rate = round(df_mort.iloc[:, 1] / df_mort.iloc[:, 0] * 100, 2)
    mort_rate = mort_rate.sort_values(ascending=False).to_frame().head(20)

    # take yesterday's data
    df = mort_rate.iloc[:, -1].sort_values(ascending=False).head(20).to_frame()
    df['totals'] = df.iloc[:, -1]
    df.drop(df.columns[0], axis=1, inplace=True)

    continent = coco.convert(names=df.index.to_list(), to='Continent')
    df['Continent'] = continent
    cont_cat = len(df['Continent'].unique())

    source = ColumnDataSource(df)

    select_tools = ['save']
    tooltips = [
        ('Country', '@Country'), ('Rate', '@totals{0.00}%')
    ]

    p = figure(x_range=df.index.tolist(), plot_width=840, plot_height=600,
               x_axis_label='Country',
               y_axis_label='Rate (%)',
               title="Mortality rate of countries with at least {} deaths " \
                     "as of ".format(num) + today_date,
               tools=select_tools)

    p.vbar(x='Country', top='totals', width=0.9, alpha=0.7, source=source,
           legend_field="Continent",
           fill_color=factor_cmap('Continent', palette=palette[cont_cat], factors=df.Continent.unique()))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = 1
    p.left[0].formatter.use_scientific = False

    p.add_tools(HoverTool(tooltips=tooltips))

    return p


# =========================================================================================  daily cases

roll = 15


def daily():
    # Classify countries into continents
    countries = sets_grouped[0].index.tolist()
    continents = coco.convert(names=countries, to='Continent')
    df_cont = pd.DataFrame({'coun': countries, 'cont': continents})
    America = df_cont[df_cont.cont == 'America'].coun.tolist()
    Asia = df_cont[df_cont.cont == 'Asia'].coun.tolist()
    Europe = df_cont[df_cont.cont == 'Europe'].coun.tolist()
    Africa = df_cont[df_cont.cont == 'Africa'].coun.tolist()
    Oceania = df_cont[df_cont.cont == 'Oceania'].coun.tolist()
    America.remove('Ecuador')

    bycontinent_conf = []
    for continent in [Europe, America, Asia, Africa, Oceania]:
        df_cat = sets_grouped[0].loc[continent]
        bycontinent_conf.append(df_cat)

    bycontinent_death = []
    for continent in [Europe, America, Asia, Africa, Oceania]:
        df_cat = sets_grouped[1].loc[continent]
        bycontinent_death.append(df_cat)

    # compute daily values for the n_top countries
    dfs_conf = [df.sort_values(by=yesterday, ascending=False).iloc[:, 2:-1].diff(axis=1).T
                for df in bycontinent_conf]

    dfs_death = [df.sort_values(by=yesterday, ascending=False).iloc[:, 2:-1].diff(axis=1).T
                 for df in bycontinent_death]

    dfs = dfs_conf + dfs_death

    # replace negative values by the previous day value
    for df in dfs:
        for i in df.columns:
            idx = df.loc[df[i] < 0, i].index
            df.loc[idx, i] = df.loc[idx - datetime.timedelta(days=1), i].tolist()

    return dfs


def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))

    # Calculate a mask for the differences that are > 5 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)

    # If the trend is rising, replace values with the previous value plus the mean of the previous
    # 7 differences
    # If the trend is falling off, replace values with the previous value minus the mean of the previous
    # 7 differences
    for date in series[this_mask].index.to_list():
        if series[date - datetime.timedelta(days=2)] - series[date - datetime.timedelta(days=0)] < 0:
            series[date] = np.abs(series[date + datetime.timedelta(days=-1)] +
                                  np.mean([series[date - datetime.timedelta(days=j)] -
                                           series[date - datetime.timedelta(days=j - 1)]
                                           for j in reversed(range(2, 8))]))
        else:
            series[date] = np.abs(series[date + datetime.timedelta(days=-1)] -
                                  np.mean([series[date - datetime.timedelta(days=j)] -
                                           series[date - datetime.timedelta(days=j - 1)]
                                           for j in reversed(range(2, 8))]))

    return series


def rolling(dfs, n_since=30, roll=roll):
    # transform to rolling average
    daily_rolled = []
    for i in range(len(dfs)):  # Transform each dataset at a time
        dF = dfs[i].apply(replace_outliers)
        # get the rolling mean
        dF = dF.rolling(roll).mean().reset_index(drop=True)
        # for each column in a DF, get indexes where rows >= n_since
        since = [pd.DataFrame(dF[j][dF[j] >= n_since]).index for j in dF.columns]
        since = [(k, since[k][0]) for k in range(len(since)) if len(since[k]) > 0]
        # restart dataframes starting from since and reset index
        dfs_ = [dF.iloc[since[i][1]:, since[i][0]].reset_index(drop=True) for i in range(len(since))]
        # concatenate the columns and remove outliers
        if len(dfs_) != 0:
            out = pd.concat(dfs_, axis=1, join='outer').reset_index(drop=True)
            # change values < 1 by 0.5
            out[out < 0.5] = 0.5
            # append
            daily_rolled.append(out)

    return daily_rolled


def bokeh_plot(dataF, cat, n_since, tickers, cont, format_axes=False):
    ''' Customizations for the Bokeh plots '''
    # cat = {'confirmed', 'deaths', 'recoveries'}
    # n_since = number of cases since we start counting
    # n_top = number of top countries to show
    # tickers = customized tickers for the logy axis. It is simpler to manually define
    # them than to compute them for each case.

    from bokeh.io import output_notebook, output_file, show, reset_output
    from bokeh.plotting import figure, save
    from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool, Span
    from bokeh.palettes import Category10

    # Specify the selection tools to be made available
    select_tools = ['box_zoom', 'pan', 'wheel_zoom', 'reset', 'crosshair', 'save']

    # Format the tooltip
    tooltips = [
        ('', '$name'),
        ('Days since', '$x{(0)}'),
        ('{}'.format(cat), '$y{(0)}')
    ]

    #
    num = dataF.iloc[:, 0].sort_values(ascending=False).values[0]
    s = str(int(num))[0]
    l = len(str(int(num)))
    a = str(int(s) + 2) + '0' * (l - 1)
    a = int(a)

    if format_axes:
        y_range = [0.49, a]
    else:
        y_range = None
    #

    p = figure(y_range=y_range,
               x_range=[-2, 340],
               y_axis_type="log", plot_width=840, plot_height=600,
               x_axis_label='Days since average daily {} passed {}'.format(cat, n_since),
               y_axis_label='',
               title=
               'Daily {} ({}-day rolling average) by number of days ' \
               'since {} cases - {}'.format(cat, roll, n_since, cont),
               toolbar_location='above', tools=select_tools, toolbar_sticky=False)

    if len(dataF.columns) > 10:
        count = 10
    else:
        count = len(dataF.columns)

    for i in range(count):
        p.line(dataF.index[:], dataF.iloc[:, i], line_width=2.5, color=Category10[10][i], alpha=0.8,
               legend_label=dataF.columns[i], name=dataF.columns[i])
        p.line(1)
        p.circle(dataF.index[:], dataF.iloc[:, i], color=Category10[10][i], fill_color='white',
                 size=3, alpha=0.8, legend_label=dataF.columns[i], name=dataF.columns[i])
        hline = Span(location=1, dimension='width', line_width=2.5, line_dash='dashed', line_color='gray')

    for i in range(count, len(dataF.columns)):
        p.line(dataF.index[:], dataF.iloc[:, i], line_width=2, alpha=0.2, color='gray',
               name=dataF.columns[i])

    p.renderers.extend([hline])
    p.yaxis.ticker = tickers

    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'

    p.add_tools(HoverTool(tooltips=tooltips))

    return p


# =========================================================================================  geo visualizations

# Construct a data set with daily cases acounting for Lat and Lon without States
sets_daily = []
for i in range(3):
    # Countries to to take care of
    c_to_change = sets[0][sets[0].State != ''].Country.unique().tolist()
    # Get Lat and Lon of Australia's, Canada's and Hubei's capitals
    mask_1 = (sets[i].State == 'Australian Capital Territory') | (sets[i].State == 'Ontario') | (
            sets[i].State == 'Hubei')
    # Get Lat and Lon for Denmark, France, Netherlands and UK
    mask_2 = (sets[0].Country == 'Denmark') | (sets[0].Country == 'France') | (sets[0].Country == 'Netherlands') | (
            sets[0].Country == 'United Kingdom')
    # Lat and Lon of countries to take care of
    c_lat_lon_1 = sets[i][mask_2][sets[i][mask_2].State == ''].loc[:, ['Country', 'Lat', 'Long']].set_index(
        'Country')
    c_lat_lon_2 = sets[i][mask_1].loc[:, ['Country', 'Lat', 'Long']].set_index('Country')
    c_lat_lon = pd.concat([c_lat_lon_1, c_lat_lon_2])
    # Records for the countries
    c_records = sets_grouped[i].loc[c_to_change].drop(['Lat', 'Long'], axis=1)
    # Full DF of countries to take care of
    full = pd.concat([c_lat_lon, c_records], axis=1)
    # Sets grouped without the countries
    df_no_c = sets_grouped[i].drop(c_to_change)
    # Concat with the full countries DF
    df_final = pd.concat([df_no_c, full]).reset_index().rename(columns={'index': 'Country'})
    # Get daily records
    df_1 = df_final.iloc[:, 3:].diff(axis=1)
    df_2 = df_final.iloc[:, :3]
    df_final = pd.concat([df_2, df_1], axis=1)
    # Drop negative values
    df_final = drop_neg(df_final)
    # Change date-time name columns by string names
    df_final.columns = df_final.columns.map(str)
    df_final = df_final.rename(columns={str(yesterday): 'New cases'})


    #
    def drop_neg(df):
        # Drop negative entries entries
        idx_l = df[df.iloc[:, -2] < 0].index.tolist()
        for i in idx_l:
            df.drop([i], inplace=True)
        return df.reset_index(drop=True)


    drop_neg(df_final)
    sets_daily.append(df_final)

fig1 = px.scatter_geo(sets_daily[0], width=1200, height=500,
                      lat="Lat", lon="Long", color='New cases',
                      hover_name="Country", size='New cases',
                      size_max=40,  # hover_data=["State"],
                      template='seaborn', projection="natural earth")
# title="New worldwide confirmed cases")

fig1.update_layout(
    geo=dict(showframe=True, showcoastlines=False, projection_type='equirectangular'),
    coloraxis_colorbar=dict(title="Confirmed", lenmode='fraction', len=0.9, y=0.52)
)
fig1.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, pad=0))

fig1.update_geos(resolution=110, showcountries=True)  # ,
#    lataxis_range=[-55, 90], lonaxis_range=[-180, 180])


fig2 = px.scatter_geo(sets_daily[1], width=1200, height=500,
                      lat="Lat", lon="Long", color='New cases',
                      hover_name="Country", size='New cases',
                      size_max=40,  # hover_data=["Country"],
                      template='seaborn', projection="natural earth")
# title="New worldwide deaths")

fig2.update_layout(
    geo=dict(showframe=True, showcoastlines=False, projection_type='equirectangular'),
    coloraxis_colorbar=dict(title="Confirmed", lenmode='fraction', len=0.9, y=0.52)
)
fig2.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, pad=0))

fig2.update_geos(resolution=110, showcountries=True)  # ,
#    lataxis_range=[-55, 90], lonaxis_range=[-180, 180])


# =========================================================================================  App
# =========================================================================================
# =========================================================================================


# ======================================================================== Sidebar

# Sidebar
st.sidebar.markdown('# Covid-19 tracker')
st.sidebar.markdown('#FlattenTheCurve #StayHome')
st.sidebar.markdown('### Latest update: {}'.format(today_date))
st.sidebar.markdown('---')
st.sidebar.markdown("Select below whether you want to see yesterday's numbers, the historical ""numbers, or the top "
                    "countries by case:")
sections = ('World new cases', 'Historical trends', 'Top countries')
selection = st.sidebar.radio('', sections)
st.sidebar.markdown('__Pro tip__: click on Menu Settings -> Show app in wide mode -> Save')
st.sidebar.markdown('---')
st.sidebar.markdown('### [Source Code](https://github.com/hectoramirez/Covid19)')
st.sidebar.markdown(
    '### [Medium Story](https://towardsdatascience.com/your-live-covid-19-tracker-with-airflow-and-github-pages'
    '-658c3e048304)')
st.sidebar.markdown('### [Contact](https://www.linkedin.com/in/harr/)')

# ======================================================================== Main
container = True


def footer():
    st.markdown('---')
    st.markdown('## Data source:')
    st.markdown(
        'We use the live COVID19 data from the [GitHub data repository](https://github.com/CSSEGISandData/COVID-19) '
        'for the 2019 Novel Coronavirus Visual Dashboard by the Johns '
        'Hopkins University Center for Systems Science and Engineering '
        '(JHU CSSE).')
    st.markdown('## Notes:')
    st.markdown('On April 17th, [_China outbreak city Wuhan raised death toll by '
                '50%_](https://www.bbc.com/news/world-asia-china-52321529) (BBC). '
                'As a consequence, outliers of this type were removed '
                '― values whose distance from the mean is larger than 5σ are replaced '
                'by the previous-day value plus/minus the mean of the previous 7-day '
                'direct differences depending on whether the trend is rising/falling off.')


if selection == sections[0]:
    st.markdown('# World new cases')
    st.markdown('---')

    opt = {'New worldwide confirmed cases': fig1, 'New worldwide deaths': fig2}

    case = st.selectbox('Select option:', ('New worldwide confirmed cases', 'New worldwide deaths'))
    st.plotly_chart(opt[case], use_container_width=True)  # container)

    footer()

elif selection == sections[1]:
    st.markdown('# Historical trends')
    st.markdown('---')

    opt = ('Confirmed cases', 'Deaths')

    daily_rolled_conf = rolling(daily()[:5])
    Europe = [df.apply(replace_outliers) for df in daily()[7:8]]
    daily_rolled_death = rolling(daily()[5:7] + Europe + daily()[8:], n_since=3)
    # daily_rolled_death = rolling(daily()[5:], n_since=3)

    case = st.selectbox('Select option:', opt)
    cont_str = ('Europe', 'America', 'Asia', 'Africa', 'Oceania')

    #    yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000]

    #    if case == 'Confirmed cases':
    #       optcont = {'Europe':bokeh_plot(daily_rolled_conf[2], 'confirmed', n_since=30, tickers=yticks, cont=cont_str[2], format_axes=True),
    #                 'America':bokeh_plot(daily_rolled_conf[0], 'confirmed', n_since=30, tickers=yticks, cont=cont_str[0], format_axes=True),
    #               'Asia':bokeh_plot(daily_rolled_conf[1], 'confirmed', n_since=30, tickers=yticks, cont=cont_str[1], format_axes=True),
    #               'Africa':bokeh_plot(daily_rolled_conf[3], 'confirmed', n_since=30, tickers=yticks, cont=cont_str[3], format_axes=True),
    #              'Oceania':bokeh_plot(daily_rolled_conf[4], 'confirmed', n_since=30, tickers=yticks, cont=cont_str[4], format_axes=True)}
    #  case = st.selectbox('', cont_str)
    # st.bokeh_chart(optcont[case], use_container_width=container)

    # if case == 'Deaths':
    #    optcont = {'Europe':bokeh_plot(daily_rolled_death[2], 'deaths', n_since=3, tickers=yticks, cont=cont_str[2], format_axes=True),
    #              'America':bokeh_plot(daily_rolled_death[0], 'deaths', n_since=3, tickers=yticks, cont=cont_str[0], format_axes=True),
    #            'Asia':bokeh_plot(daily_rolled_death[1], 'deaths', n_since=3, tickers=yticks, cont=cont_str[1], format_axes=True),
    #            'Africa':bokeh_plot(daily_rolled_death[3], 'deaths', n_since=3, tickers=yticks, cont=cont_str[3], format_axes=True),
    #           'Oceania':bokeh_plot(daily_rolled_death[4], 'deaths', n_since=3, tickers=yticks, cont=cont_str[4], format_axes=True)}
    # yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000]
    # case = st.selectbox('', cont_str)
    # st.bokeh_chart(optcont[case], use_container_width=container)

    if case == 'Confirmed cases':
        for i, df in enumerate(daily_rolled_conf):
            yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            st.markdown('## {}'.format(cont_str[i]))
            st.bokeh_chart(
                bokeh_plot(df, 'confirmed', n_since=30, tickers=yticks, cont=cont_str[i], format_axes=True),
                use_container_width=container)

    elif case == 'Deaths':
        for i, df in enumerate(daily_rolled_death):
            yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            st.markdown('## {}'.format(cont_str[i]))
            st.bokeh_chart(
                bokeh_plot(df, 'deaths', n_since=3, tickers=yticks, cont=cont_str[i], format_axes=True),
                use_container_width=container)

    footer()

elif selection == sections[2]:
    #    opt = {'Confirmed':bokehB(sets_grouped[0], cases[0]),
    #           'Deaths':bokehB(sets_grouped[1], cases[1]),
    #           'Recoveries':bokehB(sets_grouped[2], cases[2]),
    #           'Mortality rate':bokehB_mort(100)}

    #    case = st.selectbox('', ('Confirmed', 'Deaths', 'Recoveries', 'Mortality rate'))
    #    st.bokeh_chart(opt[case], use_container_width=container)

    st.markdown('# Top countries')
    st.markdown('---')

    cases2 = ('Confirmed', 'Deaths', 'Recoveries', 'Mortality rate')
    for i in range(3):
        st.markdown('## {}'.format(cases2[i]))
        st.bokeh_chart(bokehB(sets_grouped[i], cases[i]), use_container_width=container)
    st.markdown('## {}'.format(cases2[-1]))
    st.bokeh_chart(bokehB_mort(100), use_container_width=container)

    footer()
