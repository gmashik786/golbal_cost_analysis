import numpy as np
from numpy.lib.shape_base import tile
import pandas as pd
import streamlit as st
import random
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pydeck as pdk
from plotly.subplots import make_subplots
import folium
from folium import Circle
from streamlit_folium import folium_static

from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown("<h2 style='text-align: center; color: black;'><b> Global Living Cost Analysis and Recommender App</b></h2>", unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
city = pd.read_pickle('city.pkl')
city.at[63, 'Country']="United States"
city.at[63, 'Location']="Tbilisi, GA, "+"United States"
country_list=list(set(city['Country']))
coption = st.multiselect('Add countries you want general comparisons:',country_list,["Canada", "United States"])
w=[False]*len(city['Country'])
w=pd.Series(w)
l=[]
for i in coption:
  l=l+list(city[city['Country']==i].index)
for item in l:
  w[item]=True
dfcop=city[w][['Location','Food', 
                'Travel','Living','Lifestyle','Education',
                    'Income']].set_index('Location').sort_values(by = 'Income',
                                ascending = False).style.background_gradient(cmap = 'viridis')
st.dataframe(dfcop)
if st.button('Show Data'):
    st.header('Data for all countries')
    df1=city[['Country','Food','Travel',
      'Living','Lifestyle','Education','Income']].groupby(['Country']).agg('mean').style.background_gradient(cmap = 'icefire')
    st.dataframe(df1)

fig = make_subplots(rows=2, cols=3,subplot_titles=("Food","Travel","living",'Lifestyle','Education','Income'))
x = city[['Country','Food']].groupby('Country').mean().sort_values(by = 'Food', ascending = False).head(6)
x=x.reset_index()
fig.add_trace(
  
   go.Bar(name="Food",x=x['Country'], y=x['Food'],marker=dict(color=[i for i in range(len(x))])),
              1, 1)
x = city[['Country','Travel']].groupby('Country').mean().sort_values(by = 'Travel', ascending = False).head(6)
x=x.reset_index()
fig.add_trace(
    
    go.Bar(x=x['Country'], y=x['Travel'],marker=dict(color=[i for i in range(len(x))])),
              1, 2)

x = city[['Country','Living']].groupby('Country').mean().sort_values(by = 'Living', ascending = False).head(6)
x=x.reset_index()
fig.add_trace(
    
    go.Bar(x=x['Country'], y=x['Living'],marker=dict(color=[i for i in range(len(x))])),
              1, 3)
x = city[['Country','Lifestyle']].groupby('Country').mean().sort_values(by = 'Lifestyle', ascending = False).head(6)
x=x.reset_index()
fig.add_trace(
    
    go.Bar(x=x['Country'], y=x['Lifestyle'],marker=dict(color=[i for i in range(len(x))])),
              2, 1)
x = city[['Country','Education']].groupby('Country').mean().sort_values(by = 'Education', ascending = False).head(6)
x=x.reset_index()
fig.add_trace(
    
    go.Bar(x=x['Country'], y=x['Education'],marker=dict(color=[i for i in range(len(x))])),
              2, 2)
x = city[['Country','Income']].groupby('Country').mean().sort_values(by = 'Income', ascending = False).head(6)
x=x.reset_index()
fig.add_trace(
    
    go.Bar(x=x['Country'], y=x['Income'],marker=dict(color=[i for i in range(len(x))])),
            2, 3)


fig.update_layout(height=600, width=800, title_text="Most Expensive countries (Euros)",showlegend=False)
st.plotly_chart(fig)
fig1 = make_subplots(rows=2, cols=3,subplot_titles=("Food","Travel","living",'Lifestyle','Education','Income'))
x = city[['Country','Food']].groupby('Country').mean().sort_values(by = 'Food', ascending = True).head(6)
x=x.reset_index()
fig1.add_trace(
  
   go.Bar(name="Food",x=x['Country'], y=x['Food'],marker=dict(color=["ivory", "khaki", "lavender", "lavenderblush", "lawngreen"])), 
              1, 1)
x = city[['Country','Travel']].groupby('Country').mean().sort_values(by = 'Travel', ascending = True).head(6)
x=x.reset_index()
fig1.add_trace(
    
    go.Bar(x=x['Country'], y=x['Travel'],marker=dict(color=["springgreen", "steelblue", "tan", "teal", "thistle", "tomato"])),
              1, 2)

x = city[['Country','Living']].groupby('Country').mean().sort_values(by = 'Living', ascending = True).head(6)
x=x.reset_index()
fig1.add_trace(
    
    go.Bar(x=x['Country'], y=x['Living'],marker=dict(color=["cornsilk", "crimson", "cyan", "darkblue", "darkcyan"])),
              1, 3)
x = city[['Country','Lifestyle']].groupby('Country').mean().sort_values(by = 'Lifestyle', ascending = True).head(6)
x=x.reset_index()
fig1.add_trace(
    
    go.Bar(x=x['Country'], y=x['Lifestyle'],marker=dict(color=["bisque","darkorchid", "darkred", "darksalmon", "darkseagreen"])),
              2, 1)
x = city[['Country','Education']].groupby('Country').mean().sort_values(by = 'Education', ascending = True).head(6)
x=x.reset_index()
fig1.add_trace(
    
    go.Bar(x=x['Country'], y=x['Education'],marker=dict(color=[i for i in range(6,6+len(x))])),
              2, 2)
x = city[['Country','Income']].groupby('Country').mean().sort_values(by = 'Income', ascending = True).head(6)
x=x.reset_index()
fig1.add_trace(
    
    go.Bar(x=x['Country'], y=x['Income'],marker=dict(color=["sandybrown", "seagreen", "seashell", "sienna", "silver"])),
            2, 3)


fig1.update_layout(height=600, width=800, title_text="Least Expensive countries (Euros)",showlegend=False)
st.plotly_chart(fig1)
ser1=city['Country'].value_counts()
country_list=[]
c=list(ser1.index)
for i in range(len(ser1)):
  if ser1[i]>=2:
    country_list.append(c[i])
country_list.remove("Canada")
country_list.remove("United States")
country_list.remove("Australia")
country_list.remove("India")
country_list.append("India")
country_list.append("Australia")
country_list.append("United States")
country_list.append("Canada")
country_list.reverse()
scountry = st.selectbox('Select a country for the analysis of its different cities',country_list)
city['City'] = city['Location'].str.split(', ')
city['City'] = city['City'].apply(lambda x: x[0])
x = city[city['Country'] == scountry]

fig2 = make_subplots(rows=3, cols=2,subplot_titles=("Food","Travel","living",'Lifestyle','Education','Income'))

fig2.add_trace(
  
   go.Bar(y=x['City'], x=x['Food'],orientation='h',marker=dict(color=[i for i in range(6,6+len(x))])), 
              1, 1)

fig2.add_trace(
    
    go.Bar(y=x['City'], x=x['Travel'],orientation='h',marker=dict(color=[i for i in range(6,6+len(x))])),
              1, 2)

fig2.add_trace(
    
    go.Bar(y=x['City'], x=x['Living'],orientation='h',marker=dict(color=[i for i in range(6,6+len(x))])),
              2, 1)
fig2.add_trace(
    
    go.Bar(y=x['City'], x=x['Lifestyle'],orientation='h',marker=dict(color=[i for i in range(6,6+len(x))])),
              2, 2)
fig2.add_trace(
    
    go.Bar(y=x['City'], x=x['Education'],orientation='h',marker=dict(color=[i for i in range(6,6+len(x))])),
              3, 1)

fig2.add_trace(
    
    go.Bar(y=x['City'], x=x['Income'],orientation='h',marker=dict(color=[i for i in range(6,6+len(x))])),
            3, 2)
fig2.update_layout(height=1100, width=950, title_text="Comparisons between different cities of "+scountry,showlegend=False)
st.plotly_chart(fig2)
xt=x
xt.set_index(list(xt)[-1],inplace=True)
if st.button('Show Full Tabular Data'):
    st.header('Data Tabular Form for all the cities')
    st.dataframe(xt[['Food','Travel','Living','Lifestyle','Education','Income']])
plt.rcParams['figure.figsize'] = (16, 7)
plt.style.use('fivethirtyeight')
f11,ax=plt.subplots()
ax=sns.heatmap(city[['Food','Travel','Living','Lifestyle','Education','Income']].corr(),
            cmap = 'viridis', 
            annot = True, linecolor='black', linewidths = 10)
st.header("Correlation between different expenses")
st.pyplot(f11)
x = city[['Food','Travel','Living','Lifestyle','Education','Income']]
mm = MinMaxScaler()
data = mm.fit_transform(x)
data = pd.DataFrame(data)
data.columns = x.columns
data['Total Cost Score'] = (data['Food'] + data['Travel'] + data['Living'] + 
                       data['Lifestyle'] + data['Education'] + data['Income'])/6

# concat city
cities = city[['City', 'Country']]
data = pd.concat([data, cities], axis = 1)
# lets sort the values
st.header("Most Expensive Places in the World")
temp1=data[['Country','City','Total Cost Score']]
temp1.set_index('City',inplace=True)
temp1=temp1.sort_values(by = 'Total Cost Score', ascending = False).head(10).style.background_gradient(cmap = 'copper')
st.dataframe(temp1)
st.header("Cheapest Places in the World")
temp2=data[['Country','City','Total Cost Score']]
temp2.set_index('City',inplace=True)
temp2=temp2.sort_values(by = 'Total Cost Score', ascending = True).head(10).style.background_gradient(cmap = 'twilight')
st.dataframe(temp2)
qlife = pd.read_csv('movehubqualityoflife.csv')
st.header("Other ranking scores analysis for different cities in the world ")
qlif1=qlife[['City','Quality of Life']].sort_values(by = 'Quality of Life',
                            ascending = False).head(10).set_index('City').style.background_gradient(cmap = 'Greens')
st.write("**10 best cities having best quality of life score**")
st.dataframe(qlif1)
qlif2=qlife[['City','Quality of Life']].sort_values(by = 'Quality of Life',
                            ascending = True).head(10).set_index('City').style.background_gradient(cmap = 'Reds')
st.write("**10 worst cities having low quality of life score**")
st.dataframe(qlif2)
qlif3=qlife[['City','Crime Rating']].sort_values(by = 'Crime Rating',
                            ascending = False).head(10).set_index('City').style.background_gradient(cmap = 'Reds')
st.write("**10 cities having highest crime rating score**")
st.dataframe(qlif3)
qlif4=qlife[['City', 'Crime Rating']].sort_values(by = 'Crime Rating',
                            ascending = True).head(10).set_index('City').style.background_gradient(cmap = 'bone')
st.write("**10 cities having lowest crime rating score (safest cities)**")
st.dataframe(qlif4)
qlif5=qlife[['City','Health Care']].sort_values(by = 'Health Care',
                            ascending = False).head(10).set_index('City').style.background_gradient(cmap = 'Greens')
st.write("**10 cities having best health care facility**")
st.dataframe(qlif5)
qlif6=qlife[['City','Health Care']].sort_values(by = 'Health Care',
                            ascending = True).head(10).set_index('City').style.background_gradient(cmap = 'Reds')
st.write("**10 cities having worst health care facility**")
st.dataframe(qlif6)
st.markdown("<h2 style='text-align: center; color: black;'><b> City Recommendation</b></h2>", unsafe_allow_html=True)
st.write("**Select a city and a factor from below to get the recommendation**")
def recommend_better_cities(citi, factor = 'Lifestyle'):
    x = city[['City','Food','Education','Lifestyle','Travel', 'Income','Country']]
    food = x[x['City'] == citi]['Food']
    edu = x[x['City'] == citi]['Education']
    life = x[x['City'] == citi]['Lifestyle']
    travel = x[x['City'] == citi]['Travel']
    income = x[x['City'] == citi]['Income']
    best_cities = x[(x['Food'] <= food.values[0]) & (x['Education'] <= edu.values[0]) & 
                   (x['Lifestyle'] <= life.values[0]) & (x['Travel'] <= travel.values[0]) &
                   (x['Income'] > income.values[0])]
    if len(best_cities)==0:
        return "**Based on the city and factor you provided we didn't find any city.**"
    best = best_cities.sort_values(by = factor, ascending = False).head(10)
    return best[['City','Country']].reset_index(drop = True)
city_list=list(set(city['City']))
city_list.remove("Toronto")
city_list.remove("New York")
city_list.remove("Vancouver")
city_list.append("Vancouver")
city_list.append("New York")
city_list.append("Toronto")
city_list.reverse()
scity = st.selectbox('Select a city for the recommendation:',city_list)
scri = st.selectbox('Select a factor:',('Lifestyle','Food','Education','Travel', 'Income'))
st.write(recommend_better_cities(scity, scri))
st.markdown("<h2 style='text-align: center; color: black;'><b> Geographical view of diffenent cost in different areas</b></h2>", unsafe_allow_html=True)
st.write("**In this section different factors such as education, food, lifestyle cost distribution have shown in the different \
              areas of the world. Red and dark red indicates high expense region and on the other hand green and golden are indicating \
                  cheap and afordable expenses region in the map.** ")
map = folium.Map(location=[city['latitude'].mean(),
                           city['longitude'].mean()],
                 tiles='Stamen Terrain',
                 zoom_start=2)
def color_func(val,item):
    if val <= city[item].quantile(.25):
        return 'forestgreen'
    elif val <= city[item].quantile(.50):
        return 'goldenrod'
    elif val <= city[item].quantile(.75):
        return 'darkred'
    else:
        return 'red'
for i in range(0,len(city)):
    Circle(
        location=[city.iloc[i]['latitude'], city.iloc[i]['longitude']],
        radius=120000,
        color=color_func(city.iloc[i]['Food'],'Food')).add_to(map)
st.header("Food price in Different location")
folium_static(map)
map = folium.Map(location=[city['latitude'].mean(),
                           city['longitude'].mean()],
                 tiles='CartoDB dark_matter',
                 zoom_start=2)

item = 'Education'

# Add a bubble map to the base map
for i in range(0,len(city)):
    Circle(
        location=[city.iloc[i]['latitude'], city.iloc[i]['longitude']],
        radius=120000,
        color=color_func(city.iloc[i][item],item)).add_to(map)
st.header("Education Expenses in different regions")
folium_static(map)
map = folium.Map(location=[city['latitude'].mean(),
                           city['longitude'].mean()],
                 tiles='Stamen Toner',
                 zoom_start=2)

item = 'Income'
# Add a bubble map to the base map
for i in range(0,len(city)):
    Circle(
        location=[city.iloc[i]['latitude'], city.iloc[i]['longitude']],
        radius=120000,
        color=color_func(city.iloc[i][item],item)).add_to(map)
st.header("Income in different region")
folium_static(map)
map = folium.Map(location=[city['latitude'].mean(),
                           city['longitude'].mean()],
                 tiles='Open Street Map',
                 zoom_start=2)

item = 'Living'

# Add a bubble map to the base map
for i in range(0,len(city)):
    Circle(
        location=[city.iloc[i]['latitude'], city.iloc[i]['longitude']],
        radius=120000,
        color=color_func(city.iloc[i][item],item)).add_to(map)
st.header("Living cost in different regions")
folium_static(map)
map = folium.Map(location=[city['latitude'].mean(),
                           city['longitude'].mean()],
                 tiles='CartoDB Positron',
                 zoom_start=2)

item = 'Lifestyle'

# Add a bubble map to the base map
for i in range(0,len(city)):
    Circle(
        location=[city.iloc[i]['latitude'], city.iloc[i]['longitude']],
        radius=120000,
        color=color_func(city.iloc[i][item],item)).add_to(map)
st.header("Lifestyle expenses in different regions")
folium_static(map)



# import math
# # Adding code so we can have map default to the center of the data

# df = city[['latitude','longitude','Food']]#pd.DataFrame( np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
# df['Food']=df['Food'].apply(lambda x: x*x)
# fig1=pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
# initial_view_state=pdk.ViewState(
# latitude=df['latitude'].mean(),
# longitude=df['longitude'].mean(),
# zoom=1,
# pitch=50,
# ),
# layers=[
# pdk.Layer(
# 'HexagonLayer',
# data=df,
# get_position='[longitude, latitude]',
# radius=2000,
# elevation_scale=4,
# elevation_range=[0, 1000],
# pickable=True,
# extruded=True,
# ),
# pdk.Layer('ScatterplotLayer',
# data=df,
# get_position='[longitude, latitude]',
# get_color='[200, 30, 0, 160]',
# get_radius=2000,),],)
# st.pydeck_chart(fig1)
