import chart_studio.tools as tls
import chart_studio.plotly as py
import plotly.io as pio

# plotly authentication
username = 'HectoRamirez' # your username
api_key = 'Qo8mqaA9T3oSbWg6cvdc' # your api key - go to profile > settings > regenerate key

tls.set_credentials_file(username=username, api_key=api_key)

fig = 
filename = 

# push plot (gets link)
link = py.plot(fig, filename = filename, auto_open=False)

# gets HTML code to embed plotly graphs
tls.get_embed(link)