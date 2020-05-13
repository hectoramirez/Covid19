def CovidPlots():

    # Function to reproduce the interactive plots from:
    # https://hectoramirez.github.io/covid/COVID19.html
    # The code is explained in:
    # https://github.com/hectoramirez/Covid19

    import os
    import pandas as pd
    import numpy as np
    import datetime
    import plotly.express as px
    import plotly as plty
    import seaborn as sns
    #
    sns.set()
    sns.set_style("whitegrid")
    custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    }
    sns.set_style(custom_style)

    os.chdir('/Users/hramirez/GitHub/Covid19/Automated')

    # =========================================================================================  import

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
    # print('\nAccording to the latest imput, the data was updated on ' + today_date + '.')

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
        sets_grouped.append(sets[i].groupby('Country').sum())

    # =========================================================================================  top countries

    def bokehB(dataF, case):

        # Bokeh bar plots. The function takes a dataframe, datF, as the one provided by the raw data
        # (dates as columns, countries as rows). It first takes the last column as yesterday's date.

        from bokeh.io import output_file, show, output_notebook, save
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.palettes import Viridis as palette
        from bokeh.transform import factor_cmap

        df = dataF.iloc[:, -1].sort_values(ascending=False).head(20).to_frame()
        df['totals'] = df.iloc[:, -1]
        df.drop(df.columns[0], axis=1, inplace=True)

        # get continent names
        import country_converter as coco
        continent = coco.convert(names=df.index.to_list(), to='Continent')
        df['Continent'] = continent
        cont_cat = len(df['Continent'].unique())

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

        output_file('top_{}.html'.format(case))

        return save(p, 'top_{}.html'.format(case))

    def bokehB_mort(num=100):

        # Bokeh bar plots. The function already includes the confirmed and deaths dataframes,
        # and operates over them to calculate th mortality rate depending on num (number of
        # minimum deaths to consider for a country). The rest is equivalent to the BokehB()
        # function.

        from bokeh.io import output_file, show, output_notebook, save
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.palettes import Viridis as palette
        from bokeh.transform import factor_cmap

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

        import country_converter as coco
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

        output_file('top_mortality.html')

        return save(p, 'top_mortality.html')

    for i in range(3):
        bokehB(sets_grouped[i], cases[i])

    bokehB_mort(100)

    # =========================================================================================  daily cases

    roll = 7

    def daily(n_top=15):
        # compute daily values for the n_top countries
        dfs = [df.sort_values(by=yesterday, ascending=False).iloc[:n_top, 2:].diff(axis=1).T
               for df in sets_grouped]

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
        this_mask = absolute_differences_from_mean > (np.std(series) * 5)

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

    def rolling(n_since=100, roll=roll, quit_outliers=False):

        # transform to rolling average
        dFs = daily()

        sets_grouped_daily_top_rolled = []
        for i in range(3):  # Transform each dataset at a time
            dF = dFs[i].apply(replace_outliers)
            top_countries = dF.columns
            # get the rolling mean
            dF = dF.rolling(roll).mean().reset_index(drop=True)
            # for each column in a DF, get indexes where rows >= n_since
            since = [pd.DataFrame(dF[i][dF[i] >= n_since]).index[0] for i in top_countries]
            # restart dataframes starting from since and reset index
            dfs = [dF.iloc[since[i]:, i].reset_index(drop=True) for i in range(len(dF.columns))]
            # concatenate the columns and remove outliers
            if quit_outliers:
                out = pd.concat(dfs, axis=1, join='outer').reset_index(drop=True).apply(replace_outliers)
            else:
                out = pd.concat(dfs, axis=1, join='outer').reset_index(drop=True)
            # change values < 1 by 0.5
            out[out < 0.5] = 0.5

            # append
            sets_grouped_daily_top_rolled.append(out)

        return sets_grouped_daily_top_rolled

    def bokeh_plot(dataF, cat, n_since, tickers, n_top=12, format_axes=False):

        ''' Customizations for the Bokeh plots '''
        # cat = {'confirmed', 'deaths', 'recoveries'}
        # n_since = number of cases since we start counting
        # n_top = number of top countries to show
        # tickers = customized tickers for the logy axis. It is simpler to manually define
        # them than to compute them for each case.

        from bokeh.io import output_notebook, output_file, show, reset_output
        from bokeh.plotting import figure, save
        from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool, Span
        from bokeh.palettes import Category20

        # Specify the selection tools to be made available
        select_tools = ['box_zoom', 'pan', 'wheel_zoom', 'reset', 'crosshair', 'save']

        # Format the tooltip
        tooltips = [
            ('', '$name'),
            ('Days since', '$x{(0)}'),
            ('{}'.format(cat), '$y{(0)}')
        ]

        if format_axes:
            y_range = [0.49, 4000]
        else:
            y_range = None

        p = figure(y_range=y_range,
                   y_axis_type="log", plot_width=840, plot_height=600,
                   x_axis_label='Days since average daily {} passed {}'.format(cat, n_since),
                   y_axis_label='',
                   title=
                   'Daily {} ({}-day rolling average) by number of days ' \
                   'since {} cases - top {} countries ' \
                   '(as of {})'.format(cat, roll, n_since, n_top, today_date),
                   toolbar_location='above',tools=select_tools, toolbar_sticky=False)

        for i in range(n_top):
            p.line(dataF.index[:], dataF.iloc[:, i], line_width=2, color=Category20[20][i], alpha=0.8,
                   legend_label=dataF.columns[i], name=dataF.columns[i])
            p.line(1)
            p.circle(dataF.index[:], dataF.iloc[:, i], color=Category20[20][i], fill_color='white',
                     size=3, alpha=0.8, legend_label=dataF.columns[i], name=dataF.columns[i])
            hline = Span(location=1, dimension='width', line_width=2, line_dash='dashed', line_color='gray')

        p.renderers.extend([hline])
        p.yaxis.ticker = tickers

        p.legend.location = 'top_right'
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(tooltips=tooltips))

        output_file('Daily_{}.html'.format(cat))

        return save(p, 'Daily_{}.html'.format(cat))

    # Remember: rolling() throws a list of dataframes where {'confirmed': 0, 'deaths': 1, 'confirmed':2}

    yticks = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000]
    bokeh_plot(rolling(n_since=30)[0], 'confirmed', n_since=30, tickers=yticks)

    yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
    bokeh_plot(rolling(n_since=3)[1], 'deaths', n_since=3, tickers=yticks, format_axes=True)

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
        sets_daily.append(df_final)

    fig = px.scatter_geo(sets_daily[0],
                         lat="Lat", lon="Long", color='New cases',
                         hover_name="Country", size='New cases',
                         size_max=40,  # hover_data=["State"],
                         template='seaborn', projection="natural earth",
                         title="COVID-19 new worldwide confirmed cases")

    fig.update_geos(
        resolution=110,
        # showcoastlines=True, coastlinecolor="RebeccaPurple",
        # showland=True, landcolor="LightGreen",
        # showocean=True, oceancolor="LightBlue",
        showcountries=True
        # showlakes=True, lakecolor="Blue",
        # showrivers=True, rivercolor="Blue"
    )

    plty.offline.plot(fig, filename='Geo_confirmed.html', auto_open=False)

    fig = px.scatter_geo(sets_daily[1],
                         lat="Lat", lon="Long", color='New cases',
                         hover_name="Country", size='New cases',
                         size_max=40,  # hover_data=["Country"],
                         template='seaborn', projection="natural earth",
                         title="COVID-19 new worldwide deaths")

    fig.update_geos(
        resolution=110,
        # showcoastlines=True, coastlinecolor="RebeccaPurple",
        # showland=True, landcolor="LightGreen",
        # showocean=True, oceancolor="LightBlue",
        showcountries=True
        # showlakes=True, lakecolor="Blue",
        # showrivers=True, rivercolor="Blue"
    )

    plty.offline.plot(fig, filename='Geo_deaths.html', auto_open=False)

    return


CovidPlots()
