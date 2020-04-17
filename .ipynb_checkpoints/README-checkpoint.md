# Live COVID-19 tracker with Airflow

Set of charts automatically updated daily with [Apache Airflow](https://airflow.apache.org). COVID-19 data provided by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). The data can be found in [this GitHub data repository](https://github.com/CSSEGISandData/COVID-19).

* __[Link](https://hectoramirez.github.io/covid/COVID19.html) to the live tracker__

* __A Medium Story of this project was featured in [Towards Data Science](https://medium.com/p/your-live-covid-19-tracker-with-airflow-and-github-pages-658c3e048304?source=email-2e35a42940fd--writer.postDistributed&sk=343b8c88e348ff738b1f947c38076c97)__

[![Licence](https://img.shields.io/badge/Licence-MIT-red)]((https://opensource.org/licenses/MIT))
[![COVID-19 Tracker](https://img.shields.io/badge/COVID--19-Tracker-green)](https://hectoramirez.github.io/covid/COVID19.html)
[![Medium](https://img.shields.io/badge/Medium-Story-informational)](https://medium.com/p/your-live-covid-19-tracker-with-airflow-and-github-pages-658c3e048304?source=email-2e35a42940fd--writer.postDistributed&sk=343b8c88e348ff738b1f947c38076c97)

![Illustration of photon/dark photon passage through inhomogeneities.](plots/gif.gif)

We use the [bokeh](https://bokeh.org) and [plotly](https://plotly.com) visualization libraries.

### COVID19_notebook.ipynb

Along the notebook, we
1. Load and clean the data.
2. Show bokeh interactive bar plots for the top countries by confirmed cases, deaths, recoveries and mortality rate.
3. Present the world totals.
4. Compute the daily cases and show bokeh interactive time series plots.
5. We show plotly geographical, interactive maps.

### Automated/

Series of scripts used for Airflow:
* <b>covid19_dag.py</b> Airflow DAG that automates the execution of:
    * <b>covid_func.py</b> Reproduces the code in the Jupyter notebook.
    * <b>git_push.py</b> Commit/push of the plots to the GitHub Pages repository.
    
The tracker is updated daily at 01:00 UTC.