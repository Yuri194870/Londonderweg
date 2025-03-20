import requests
import pandas as pd
import streamlit as st
import json
import plotly.express as px
import matplotlib.pyplot as plt
import folium 
from streamlit_folium import st_folium
import geopandas as gpd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from folium.plugins import MarkerCluster, MeasureControl, MiniMap
import branca
from folium.plugins import Search
import matplotlib.colors as mcolors


############################################################
############################################################

csv_url = "https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/londenweer.csv"
londenweer = pd.read_csv(csv_url)
aantal_ritten_per_maand = pd.read_csv('https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/aantal_ritten_per_maand.csv')
gemiddelde_duur_per_maand= pd.read_csv('https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/gemiddelde_duur_per_maand.csv')
metropredict = pd.read_csv('https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/metrokaart.csv')

# Definieer de seizoenen op basis van de maand
def bepaal_seizoen(maand):
    if maand in ['March', 'April', 'May']:
        return "Lente"
    elif maand in ['June', 'July', 'August']:
        return "Zomer"
    elif maand in ['September', 'October', 'November']:
        return "Herfst"
    else:
        return "Winter"

# Voeg de seizoenen toe aan de dataset
aantal_ritten_per_maand["Seizoen"] = aantal_ritten_per_maand["Month"].apply(bepaal_seizoen)

# Definieer kleuren per seizoen
kleuren = {
    "Lente": "darkgreen",
    "Zomer": "orange",
    "Herfst": "brown",
    "Winter": "gray"
}

# Maak de kleurenlijst aan voor de plot
kleuren_per_maand = aantal_ritten_per_maand["Seizoen"].map(kleuren)

# Maak een subplot met 4 rijen en 1 kolom (één grafiek per rij)
plot = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,  # Zelfde x-as voor alle grafieken
    vertical_spacing=0.1,  
    subplot_titles=[
        "Aantal ritten & duur per maand",
        "Weerdata per maand"
    ],
    specs=[[{"secondary_y": True}],[{"secondary_y": True}]]
)

# Voeg de bar graph toe met seizoenskleuren
plot.add_trace(
    go.Bar(
        x=aantal_ritten_per_maand["Month"], 
        y=aantal_ritten_per_maand["count"], 
        name="Aantal ritten",        
        hovertemplate="Aantal ritten: %{y:.0f}<extra></extra>",
        marker=dict(color=kleuren_per_maand)
    ),row=1, col=1, secondary_y=False
)

# Voeg de gemiddelde duur lijn toe
plot.add_trace(
    go.Scatter(
        x=gemiddelde_duur_per_maand["Month"], 
        y=gemiddelde_duur_per_maand["median"], 
        name="Gemiddelde duur", 
        yaxis="y2",
        hovertemplate="Gemiddelde duur: %{y:.0f} min<extra></extra>",
    ),row=1, col=1, secondary_y=True
)


# Voeg de lijn toe voor temperatuur
plot.add_trace(
    go.Scatter(
        x=londenweer["Month"],
        y=londenweer["gem_temp"],
        name="Temperatuur (°C)",
        yaxis="y3",
        hovertemplate="Temp: %{y:.1f} °C<extra></extra>",
        line=dict(color="red", dash="dot")
    ),
    row=2, col=1
)

# Voeg de neerslag en windkracht toe op dezelfde subplot
plot.add_trace(
    go.Scatter(
        x=londenweer["Month"],
        y=londenweer["gem_neerslag"],
        name="Neerslag (mm)",
        hovertemplate="Neerslag: %{y:.1f} mm<extra></extra>",
        line=dict(color="blue", dash="dash")
    ),
    row=2, col=1
)

plot.add_trace(
    go.Scatter(
        x=londenweer["Month"],
        y=londenweer["gem_windkracht"],
        name="Windkracht (km/u)",
        hovertemplate="Wind: %{y:.1f} km/h<extra></extra>",
        line=dict(color="purple", dash="solid")
    ),
    row=2, col=1
)

# Update layout
plot.update_layout(yaxis=dict(title="Aantal ritten", side="left"),yaxis2=dict(title="Gemiddelde duur (min)", side="right",range=[0, max(gemiddelde_duur_per_maand["median"]+2)]),yaxis3=dict(title="Wind, temperatuur, neerslag", side="left"),
    height=800,  # Zorgt voor genoeg ruimte
    title_text="2021 : fietstrends en weerdata per maand",
    showlegend=True,  # Laat de legenda zien
    hovermode="x unified",plot_bgcolor='rgba(255, 255, 255, 0.2)',  # Transparante witte achtergrond voor de plot
    paper_bgcolor='rgba(255, 255, 255, 0.1)'
)

############################################################
############################################################
# Streamlit instellingen
pd.set_option('display.max_columns', None)
st.set_page_config(layout='wide')

# Achtergrondkleur instellen via CSS
st.markdown(
    """
    <style>
        body {
            background-color: #FFC72C; /* NS-geel */
        }
        .stApp {
            background-color: #FFC72C; /* NS-geel */
        /* Alle tekst in de app */
        html, body, [class*="st-"] {
            color: #003082 !important;  /* NS-blauw */
            font-family: Arial, sans-serif;
        }
        /* Titels */
            h1, h2, h3, h4 {
            color: #003082 !important;  /* NS-blauw */
        }
        .css-1d391kg, .css-2trqyj, .st-emotion-cache-1y4p8pa {
            background-color: rgba(255, 255, 255, 0) !important; /* Transparant */
            color: #002D72 !important; /* NS-blauw */
        }
        .stSelectbox, .stPlotlyChart, .stButton {
            background-color: rgba(255, 255, 255, 0) !important; /* Transparant */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Functie om afbeeldingen correct te laden
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# NS-logo rechts onderin plaatsen
ns_logo_base64 = get_image_base64("NS_logo.png")
st.markdown(
    f"""
    <div style="position: fixed; bottom: 10px; right: 10px;">
        <img src="data:image/png;base64,{ns_logo_base64}" style="width: 120px;" />
    </div>
    """,
    unsafe_allow_html=True
)

# Dataset inlezen
fietsweerdata = pd.read_csv('https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/fietsweerdata.csv')
fietsweerdata['datum'] = pd.to_datetime(fietsweerdata['datum'], format="%Y-%m-%d")
fietsweerdata['dag_van_de_week'] = fietsweerdata['datum'].dt.day_name()

dagvandeweek = fietsweerdata.groupby('dag_van_de_week').agg(
    gemiddelde_duur=('gemiddelde_duur', 'mean'),
    aantal_ritjes=('aantal_ritjes', 'mean'),
    mediane_duur=('mediane_duur','mean')
).reset_index()

#################### Tabs aanmaken ##############################3
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Data Exploration", "London Maps", "Predictions","Conclusions"])

# Homepagina met NS-verhaal
with tab1:
    st.title("NS: LondOnderweg!")
    ov_fiets_base64 = get_image_base64("OV_fiets_foto.jpg")
    st.markdown(f'''
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{ov_fiets_base64}" style="width:1000px; height:auto;" />
        </div>
    ''', unsafe_allow_html=True)
    st.header("")
    st.header("Na het domineren van de markt in Nederland...")
    st.header("is het tijd voor de volgende stap: 🇳🇱 ➡️ 🇬🇧")
    st.header("")
    st.write("NS gaat International en breidt haar OV-fietsen-netwerk uit naar Londen")
    st.write("Met een geavanceerd fietsdeelsysteem integreren we naadloos met de Londense vervoersnetwerken. ")
    st.write("Onze missie? Duurzaam en efficiënt reizen mogelijk maken – niet alleen in Nederland, maar ook internationaal!")
    st.write("\n")
    st.header("")
    st.header("🔍 Wat kun je in dit dashboard ontdekken?")
    st.write("- **Data exploration**: Hoe zit het OV gebruik van Londen in elkaar? En hoe beïnvloed het weer dit?")
    st.write("- **London Tube Map**: Een visualisatie van waar alle fiets hubs en metro stations zijn in London. Ook kun je hier zien hoeveel deze worden gebruikt.")
    st.write("- **Prediction**: We doen een voorspelling hoe de data er in de toekomst uit zal zien. Zodat we kunnen voorspellen hoeveel fietsen er waar geplaatst moeten worden.")   
    st.write("- **Conclusion**: Ons uiteindelijke advies aan NS International!")



################## TAB 2 BOUWEN #########################
with tab2:

    st.write("## Fietsdata samengevat")

    st.plotly_chart(plot)

    st.markdown('<hr style="border: 1px solid #003082;">', unsafe_allow_html=True)

    st.write("## Correlatie Matrix")

    st.write("")

    # opzoek naar corrolaties met het weer en de hoeveelheid fietsen die er per dag verhuurd worden 

   # Selecteer alleen correlaties met 'Aantal verhuurde fietsen'
    fig, ax = plt.subplots(figsize=(2, 2))
    fig.patch.set_alpha(0.3)  # Transparante achtergrond voor de figuur

    # Bereken de correlatiematrix en selecteer alleen de relevante kolom
    correlatie_matrix = fietsweerdata[['tmax_c', 'neerslag_mm', 'windsnelheid_kmh', 'aantal_ritjes']].corr()
    correlatie_matrix = correlatie_matrix[['aantal_ritjes']]  # Alleen deze kolom op x-as

    # Maak de heatmap met verbeterde opmaak en transparantie
    sns.heatmap(
        correlatie_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar=False,  # Verwijder kleurenschaal als er maar één kolom is
        annot_kws={"size": 10},  # Pas lettergrootte aan
    )

    # Maak de achtergrond van de assen transparant
    ax.set_facecolor('none')

    # Titels en labels
    # plt.title("Correlatie met Aantal Verhuurde Fietsen", loc='right', pad=20)
    ax.set_yticklabels(['Max temp. (C)', 'Neerslag (mm)', 'Windsnelheid (kmh)', 'Aantal ritten'])
    ax.set_xticklabels(['Aantal ritten'])

    col1, col2 = st.columns([1, 2])

    with col1:  
        st.pyplot(fig)
    
    with col2:
        st.write(""" Op basis van de correlatiematrix blijkt dat het aantal verhuurde fietsen sterk samenhangt met de temperatuur (correlatie: 0.76). Dit betekent dat bij hogere temperaturen meer fietsen worden gehuurd, wat logisch is omdat mensen eerder geneigd zijn om te fietsen bij aangenaam weer.""") 
        st.write("")
        st.write("""Daarentegen hebben neerslag (-0.28) en windsnelheid (-0.31) een negatieve invloed op het aantal ritten. Dit wijst erop dat regen en harde wind fietsgebruik ontmoedigen, zij het in mindere mate dan temperatuur het stimuleert. Hoewel deze factoren een invloed hebben, zijn ze niet de enige bepalende variabelen. Andere elementen, zoals seizoensgebonden trends, dag van de week en evenementen, kunnen ook bijdragen aan fluctuaties in fietsverhuur.""")
        st.write("")
        st.write(""" Toch bevestigt deze analyse dat goed weer een sterke stimulans is voor fietsgebruik, terwijl ongunstige weersomstandigheden een remmende werking hebben. Dit inzicht kan nuttig zijn voor het plannen van fietsvoorzieningen en het optimaliseren van de beschikbaarheid van deelfietsen. """)
        st.write("")
        st.write("Om hier meer inzicht in te verkrijgen is hieronder een lineaire regressie beschikbaar.")
    
    # Grafieken met drop down menu
    # Functie om een trendlijn te berekenen
    def get_trendline(x, y):
        slope, intercept, _, _, _ = linregress(x, y)
        trendline = slope * x + intercept
        return trendline

   
    # Keuzemenu voor de gebruiker
    opties = {
        'Neerslag (mm)': 'neerslag_mm',
        'Sneeuw (cm)': 'snow',
        'Max Temperatuur (°C)': 'tmax_c',
        'Windsnelheid (km/h)': 'windsnelheid_kmh',
        'Gemiddeld aantal ritten per dag': 'dag_van_de_week'
    }

    st.markdown("""
        <style>
            div[data-baseweb="select"] > div {
                background-color: rgba(0,0,0,0) !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr style="border: 1px solid #003082;">', unsafe_allow_html=True)

    st.write("## Lineaire regressie")

    keuze = st.selectbox("Selecteer een variabele voor de analyse:", list(opties.keys()))

    # Bepaal de geselecteerde datasetkolom
    x_col = opties[keuze]

    # Maak de plot afhankelijk van de selectie
    fig = go.Figure()

    # Zorgen dat de dagen van de week in de juiste volgorde staan
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dagvandeweek['dag_van_de_week'] = pd.Categorical(dagvandeweek['dag_van_de_week'], categories=day_order, ordered=True)
    dagvandeweek = dagvandeweek.sort_values('dag_van_de_week')

    if x_col == "dag_van_de_week":
        # Staafdiagram voor dag van de week
        fig.add_trace(go.Bar(
            x=dagvandeweek['dag_van_de_week'], 
            y=dagvandeweek['aantal_ritjes'],
            name="Gem. ritten per dag"
        ))
        fig.update_xaxes(title_text='Dag van de Week')
        fig.update_yaxes(title_text='Gemiddeld aantal ritten')
    else:
        # Scatterplot met smooth color transition
        fig.add_trace(go.Scatter(
            x=fietsweerdata[x_col], 
            y=fietsweerdata['aantal_ritjes'], 
            mode='markers',
            marker=dict(
                color=fietsweerdata[x_col],  # Kleur gebaseerd op de x-waarde
                colorscale='bluered',  # Smooth overgang van blauw naar rood
                showscale=True,  # Voeg een colorbar toe
                colorbar=dict(title=keuze)  # Label de kleurenbalk met de gekozen variabele
            ),
            name="Data"
        ))

        # Trendlijn berekenen en toevoegen
        trendline = get_trendline(fietsweerdata[x_col], fietsweerdata['aantal_ritjes'])
        fig.add_trace(go.Scatter(
            x=fietsweerdata[x_col], 
            y=trendline, 
            mode='lines', 
            name="Trendlijn",
            line=dict(color='black', dash='dash')
        ))

        fig.update_xaxes(title_text=keuze)
        fig.update_yaxes(title_text='Aantal ritten')

    # **Legenda naar onderkant verplaatsen & transparante achtergrond instellen & assen en teksten zwart maken**
    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontale legenda
            yanchor="top",  # Bovenkant van de legenda
            y=-0.2  # Plaatsing onder de plot
        ),
            plot_bgcolor='rgba(255, 255, 255, 0.2)',  # Transparante witte achtergrond voor de plot
            paper_bgcolor='rgba(255, 255, 255, 0.1)' ,  
        xaxis=dict(
            showline=True,
            showgrid=False,
            linecolor='black',
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            linecolor='black',
            tickfont=dict(color='black')
        )
    )

    # Toon de figuur


    # Voeg een beschrijvende tekst toe onder de plot
    beschrijvingen = {
        'neerslag_mm': "Deze grafiek toont de relatie tussen neerslag en het aantal verhuurde fietsen. "
                   "Over het algemeen zien we dat regenval een negatief effect heeft op het aantal ritten.",
        'snow': """Hier zien we hoe sneeuwval de verhuur van fietsen beïnvloedt. Het heeft in 2021 maar 3 dagen gesneeuwd, dus harde conclusies kunnen we niet trekken. 
        Toch zien wel wel dat er relatief weinig fietsen gehuurd worden als het sneeuwt. """,
        'tmax_c': "Deze grafiek laat zien hoe de maximale temperatuur invloed heeft op de fietshuur. "
              "Warme dagen stimuleren vaak fietsgebruik.",
        'windsnelheid_kmh': "Sterke wind kan een beperkende factor zijn voor fietsverhuur. "
                        "Hier zie je hoe de verhuur verandert bij verschillende windsnelheden.",
        'dag_van_de_week': "Dit staafdiagram toont het gemiddelde aantal verhuurde fietsen per dag van de week. "
                       "Weekenddagen kunnen pieken vertonen vanwege recreatief fietsen."
    }

# Toon de relevante tekst
    st.write(beschrijvingen[x_col])
    st.plotly_chart(fig)

################################ TAB 3 BOUWEN #########################################
with tab3:

    
    # %%
    london_station_locaties = gpd.read_file('https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/London%20stations.json')

    # Vervang 'Kings Cross St. Pancras' door "King's Cross St. Pancras" in de kolom 'name'
    london_station_locaties['name'] = london_station_locaties['name'].replace('Kings Cross St. Pancras', "King's Cross St. Pancras")
    london_station_locaties['name'] = london_station_locaties['name'].replace('Bank', "Bank and Monument")
    london_station_locaties['name'] = london_station_locaties['name'].replace('Highbury and Islington', "Highbury & Islington")

    # %%
    metrodata = pd.read_csv('https://raw.githubusercontent.com/Yuri194870/Londonderweg/refs/heads/main/TfL_stations.csv')
    metrodata.sort_values('En/Ex 2021', ascending=False).head(20)

    # Verwijder 'LU', 'TfL', en 'NR' uit de kolom 'Station'
    metrodata['Station'] = metrodata['Station'].str.replace(r'\s*(LU|TfL|NR)\s*', '', regex=True)

    # Verwijder overgebleven spaties
    metrodata['Station'] = metrodata['Station'].str.strip().str.replace(r'\s+', ' ', regex=True)

    
    metrodata.sort_values('En/Ex 2021', ascending=False).head(20)

    # Definieer de aggregatiefunctie voor elke kolom
    agg_dict = {
        # Voor de kolommen 'En/Ex 2017' t/m 'En/Ex 2021': tel de waarden op
        **{col: 'sum' for col in ['En/Ex 2017', 'En/Ex 2018', 'En/Ex 2019', 'En/Ex 2020', 'En/Ex 2021']},
        
        # Voor de kolom 'LINES': voeg unieke waarden samen met een komma
        'LINES': lambda x: ', '.join(x.dropna().astype(str).unique()),
        
        # Voor alle andere kolommen: behoud de eerste waarde
        **{col: 'first' for col in metrodata.columns if col not in ['Station', 'En/Ex 2017', 'En/Ex 2018', 'En/Ex 2019', 'En/Ex 2020', 'En/Ex 2021', 'LINES']}
    }

    # Voer de aggregatie uit
    metrodata = metrodata.groupby('Station', as_index=False).agg(agg_dict)

    metrodata['LINES'] = [
        line + ', Elizabeth Line' if elizabeth == 'Yes' else line 
        for line, elizabeth in zip(metrodata['LINES'], metrodata['Elizabeth Line'])
    ]

    # %%
    metrokaart = london_station_locaties.merge(metrodata,how='left', right_on='Station', left_on='name')
    metrokaart['En/Ex 2021'] = metrokaart['En/Ex 2021'].fillna(0).astype(int)
    metrokaart['En/Ex 2020'] = metrokaart['En/Ex 2020'].fillna(0).astype(int)
    metrokaart['En/Ex 2019'] = metrokaart['En/Ex 2019'].fillna(0).astype(int)
    metrokaart['En/Ex 2018'] = metrokaart['En/Ex 2018'].fillna(0).astype(int)
    metrokaart['En/Ex 2017'] = metrokaart['En/Ex 2017'].fillna(0).astype(int)


    # %%
    def maak_kleur_donkerder(kleur_naam, factor=0.7):
        """
        Maak een kleur donkerder op basis van de kleurnaam.

        Parameters:
            kleur_naam (str): De naam van de kleur, bijvoorbeeld 'orange', 'GreenYellow', 'yellow'.
            factor (float): Een getal tussen 0 en 1 dat bepaalt hoe donker de kleur wordt. 
                            Standaard is 0.7 (70% van de oorspronkelijke helderheid).

        Returns:
            str: Een hexadecimale kleurcode van de donkerdere kleur.
        """
        if not (0 <= factor <= 1):
            raise ValueError("Factor moet tussen 0 en 1 liggen.")

        # Converteer de kleurnaam naar een RGB-tuple
        rgb_kleur = mcolors.to_rgb(kleur_naam)

        # Maak de kleur donkerder
        donkerdere_rgb = tuple(min(max(channel * factor, 0), 1) for channel in rgb_kleur)

        # Converteer de donkerdere RGB-kleur naar een hexadecimale kleurcode
        donkerdere_hex = mcolors.to_hex(donkerdere_rgb)

        return donkerdere_hex

    metrokaart['donkerekleur'] = metrokaart['marker-color'].apply(maak_kleur_donkerder, factor = 0.4)

    # %%
    London_locatie = [51.50338,-0.08]
    m = folium.Map(location = London_locatie, zoom_start=11, tiles=None)

    folium.raster_layers.TileLayer('CartoDB_PositronNoLabels', name= 'Light map',opacity=0.5,overlay=True, control=False).add_to(m)
    folium.raster_layers.TileLayer('OpenRailwayMap', name= 'Train lines', overlay=True,opacity=0.2).add_to(m)

    leeg = folium.FeatureGroup(overlay= False,name="Pick a year", show=True).add_to(m)
    jaar_2021 = folium.FeatureGroup(overlay= False,name="2021", show=False).add_to(m)
    jaar_2020 = folium.FeatureGroup(overlay= False,name="2020", show=False).add_to(m)
    jaar_2019 = folium.FeatureGroup(overlay= False,name="2019", show=False).add_to(m)
    jaar_2018 = folium.FeatureGroup(overlay= False,name="2018", show=False).add_to(m)
    jaar_2017 = folium.FeatureGroup(overlay= False,name="2017", show=False).add_to(m)

    for i in metrokaart.index:
        tooltip_text = f"""<big><b>{metrokaart['name'][i]}</b></big><br><br>
                            <b>Passengers in 2021</b><br>
                            {((metrokaart['En/Ex 2021'][i])/1000000).round(2)} million<br><br>
                            <b>Lines</b><br>
                            {str(metrokaart['LINES'][i]).replace(',', '<br>') if pd.notna(metrokaart['LINES'][i]) else 'No data'}
                        """
        jaar_2021.add_child(folium.Circle(
            location=[metrokaart.geometry[i].y, metrokaart.geometry[i].x],
            tooltip=tooltip_text,
            color=(metrokaart['donkerekleur'][i]),
            fill=True,
            fill_color=(metrokaart['donkerekleur'][i]),
            radius=metrokaart['En/Ex 2021'][i]/80000,
        ).add_to(m))

    for i in metrokaart.index:
        tooltip_text = f"""<big><b>{metrokaart['name'][i]}</b></big><br><br>
                            <b>Passengers in 2020</b><br>
                            {((metrokaart['En/Ex 2020'][i])/1000000).round(2)} million<br><br>
                            <b>Lines</b><br>
                            {str(metrokaart['LINES'][i]).replace(',', '<br>') if pd.notna(metrokaart['LINES'][i]) else 'No data'}

                        """
        jaar_2020.add_child(folium.Circle(
            location=[metrokaart.geometry[i].y, metrokaart.geometry[i].x],
            tooltip=tooltip_text,
            color=(metrokaart['donkerekleur'][i]),
            fill=True,
            fill_color=(metrokaart['donkerekleur'][i]),
            radius=metrokaart['En/Ex 2020'][i]/80000
        ).add_to(m))

    for i in metrokaart.index:
        tooltip_text = f"""<big><b>{metrokaart['name'][i]}</b></big><br><br>
                            <b>Passengers in 2019</b><br>
                            {((metrokaart['En/Ex 2019'][i])/1000000).round(2)} million<br><br>
                            <b>Lines</b><br>
                            {str(metrokaart['LINES'][i]).replace(',', '<br>') if pd.notna(metrokaart['LINES'][i]) else 'No data'}

                        """
        jaar_2019.add_child(folium.Circle(
            location=[metrokaart.geometry[i].y, metrokaart.geometry[i].x],
            tooltip=tooltip_text,
            color=(metrokaart['donkerekleur'][i]),
            fill=True,
            fill_color=(metrokaart['donkerekleur'][i]),
            radius=metrokaart['En/Ex 2019'][i]/80000
        ).add_to(m))

    for i in metrokaart.index:
        tooltip_text = f"""<big><b>{metrokaart['name'][i]}</b></big><br><br>
                            <b>Passengers in 2018</b><br>
                            {((metrokaart['En/Ex 2018'][i])/1000000).round(2)} million<br><br>
                            <b>Lines</b><br>
                            {str(metrokaart['LINES'][i]).replace(',', '<br>') if pd.notna(metrokaart['LINES'][i]) else 'No data'}

                        """    
        jaar_2018.add_child(folium.Circle(
            location=[metrokaart.geometry[i].y, metrokaart.geometry[i].x],
            tooltip=tooltip_text,
            color=(metrokaart['donkerekleur'][i]),
            fill=True,
            fill_color=(metrokaart['donkerekleur'][i]),
            radius=metrokaart['En/Ex 2018'][i]/80000
        ).add_to(m))

    for i in metrokaart.index:
        tooltip_text = f"""<big><b>{metrokaart['name'][i]}</b></big><br><br>
                            <b>Passengers in 2017</b><br>
                            {((metrokaart['En/Ex 2017'][i])/1000000).round(2)} million<br><br>
                            <b>Lines</b><br>
                            {str(metrokaart['LINES'][i]).replace(',', '<br>') if pd.notna(metrokaart['LINES'][i]) else 'No data'}

                        """    
        jaar_2017.add_child(folium.Circle(
            location=[metrokaart.geometry[i].y, metrokaart.geometry[i].x],
            tooltip=tooltip_text,
            color=(metrokaart['donkerekleur'][i]),
            fill=True,
            fill_color=(metrokaart['donkerekleur'][i]),
            radius=metrokaart['En/Ex 2017'][i]/80000
        ).add_to(m))

    folium.LayerControl(position='topright', collapsed=False, draggable=False).add_to(m)

    

    # ###
    # Je ziet nu een kaart van Londen. Daaroverheen zit een kaartlaag met de trainlines geplot. Die is ook uit te zetten.
    # Kies het jaartal om de verschillende druktes per station te zien.
    # De kleuren corresponderen met de zones. Maar misschien moet dat weg...

    # %%
    # #########################################################################################################

    # FIETSEN

    # #########################################################################################################

    fietslocatiestats = pd.read_pickle('https://github.com/Yuri194870/Londonderweg/raw/refs/heads/main/flstats.pkl')

    lenteritjes = fietslocatiestats[fietslocatiestats['seizoen'] == 'Lente']
    winterritjes = fietslocatiestats[fietslocatiestats['seizoen'] == 'Winter']
    zomerritjes = fietslocatiestats[fietslocatiestats['seizoen'] == 'Zomer']
    herfstritjes = fietslocatiestats[fietslocatiestats['seizoen'] == 'Herfst']

    herfstritjes = herfstritjes.groupby(['name','long','lat']).agg(gemiddelde_duur=('gemiddelde_duur', 'mean'),
        aantal_ritjes=('aantal_ritjes', 'mean')).reset_index()

    winterritjes = winterritjes.groupby(['name','long','lat']).agg(gemiddelde_duur=('gemiddelde_duur', 'mean'),
        aantal_ritjes=('aantal_ritjes', 'mean')).reset_index()

    lenteritjes = lenteritjes.groupby(['name','long','lat']).agg(gemiddelde_duur=('gemiddelde_duur', 'mean'),
        aantal_ritjes=('aantal_ritjes', 'mean')).reset_index()

    zomerritjes = zomerritjes.groupby(['name','long','lat']).agg(gemiddelde_duur=('gemiddelde_duur', 'mean'),
        aantal_ritjes=('aantal_ritjes', 'mean')).reset_index()

    herfstritjes[['gemiddelde_duur','aantal_ritjes']] = herfstritjes[['gemiddelde_duur','aantal_ritjes']].astype('int')

    zomerritjes[['gemiddelde_duur','aantal_ritjes']] = zomerritjes[['gemiddelde_duur','aantal_ritjes']].astype('int')
    winterritjes[['gemiddelde_duur','aantal_ritjes']] = winterritjes[['gemiddelde_duur','aantal_ritjes']].astype('int')
    lenteritjes[['gemiddelde_duur','aantal_ritjes']] = lenteritjes[['gemiddelde_duur','aantal_ritjes']].astype('int')


    # %%
    # Basiskaart
    London_locatie = [51.50338, -0.12]
    fietskaart = folium.Map(location=London_locatie, zoom_start=12, tiles=None)

    folium.raster_layers.TileLayer('CartoDB.Voyager', opacity=1, overlay=True,control=False).add_to(fietskaart)

    # FeatureGroups per seizoen
    leeg = folium.FeatureGroup(overlay=False, name="Pick a season", show=True).add_to(fietskaart)
    lente = folium.FeatureGroup(overlay=False, name="Lente", show=False).add_to(fietskaart)
    zomer = folium.FeatureGroup(overlay=False, name="Zomer", show=False).add_to(fietskaart)
    herfst = folium.FeatureGroup(overlay=False, name="Herfst", show=False).add_to(fietskaart)
    winter = folium.FeatureGroup(overlay=False, name="Winter", show=False).add_to(fietskaart)

    # Functie om cirkels toe te voegen
    def voeg_cirkels_toe(feature_group, df, kleur):
        for i in df.index:
            tooltip_text = f"""<big><b>{df['name'][i]}</b></big><br><br>
                            <b>Aantal fietsers per dag:</b><br> {df['aantal_ritjes'][i]}<br><br>
                            <b>Gemiddelde duur van een rit:</b><br> {df['gemiddelde_duur'][i]} minuten
                            """    
            feature_group.add_child(folium.Circle(
                location=[df.lat[i], df.long[i]],
                tooltip=tooltip_text,
                color=kleur,
                fill=True,
                fill_color=kleur,
                radius=int(df['aantal_ritjes'][i]) *1.5  # Oplossing: int() gebruiken
            ))

    # Cirkels per seizoen toevoegen
    voeg_cirkels_toe(lente, lenteritjes, "darkgreen")
    voeg_cirkels_toe(zomer, zomerritjes, "orange")
    voeg_cirkels_toe(herfst, herfstritjes, "brown")
    voeg_cirkels_toe(winter, winterritjes, "gray")

    # Layer control toevoegen
    folium.LayerControl(position='topright', collapsed=False, draggable=False).add_to(fietskaart)

    


    from streamlit.components.v1 import html

    col1, col2 = st.columns([1, 1])
    with col1:  

        st.write("## London Tube Map")

        # Zet de kaart om naar HTML
        map_html = m._repr_html_()

        # Toon de kaart in Streamlit met een aangepaste grootte
        html(map_html, height=500, width=900)
    with col2:
        st.write("## London Bike Map")
        fietskaart = fietskaart._repr_html_()
        html(fietskaart, height = 500, width = 900)

    st.write("""De kaarten tonen het gebruik van de Londense metro en deelfietsen tussen 2017 en 2021. Een opvallend patroon is dat de locaties van deelfietsen zich grotendeels in de nabijheid van metrostations bevinden. Dit suggereert dat deelfietsen een aanvulling vormen op het openbaar vervoer, mogelijk als eerste- of laatste-mijloplossing voor reizigers.
    Wat betreft de metrodata valt op dat het gebruik in 2020 en 2021 aanzienlijk lager ligt dan in de voorgaande jaren. Dit is waarschijnlijk een gevolg van de COVID-19-pandemie, waarin lockdowns, thuiswerken en reisbeperkingen hebben geleid tot een afname in het aantal metroreizigers.
    Deze informatie kan waardevol zijn voor mobiliteitsplanning en stedelijke infrastructuur. De sterke koppeling tussen metro- en deelfietsgebruik kan duiden op kansen om fietsinfrastructuur bij stations verder te optimaliseren. Daarnaast kan de afname in metrogebruik in 2020 en 2021 aanleiding zijn voor verdere analyses: hebben reizigers blijvende alternatieven gevonden, zoals fietsen of thuiswerken, of is het gebruik na 2021 hersteld? Door deze trends te monitoren, kunnen beleidsmakers inspelen op veranderende reispatronen en het vervoersaanbod beter afstemmen op de behoeften van de stad.""")

################################## TAB 4 BOUWEN ###########################################
with tab4:


    st.write("## Metro voorspelling:")
    st.write("Nu we weten hoeveel fietsen er worden verhuurd per dag, en waar deze voornamelijk staan, kunnen we proberen een voorspelling te maken. Een voorspelling over het aantal reizigers per metro kan ons meer inzicht geven in waar wij onze fietsen moeten gaan plaatsen.")
    st.write("")
    st.write("**Let op**: de data is zeer aangetast door de pandemie in 2021!")

    # Model trainen
    X = metropredict[['Jaar']]
    y = metropredict['Passagiers']
    model = LinearRegression()
    model.fit(X, y)

    # Unieke lijst van stations
    stations = metropredict['name'].unique()

    # Kies een kleurenpalet met genoeg unieke kleuren
    colors = px.colors.qualitative.Dark2  # Andere opties: Set2, Pastel, Dark24

    fig = go.Figure()  # Maak een lege figuur

    for i, station in enumerate(stations):
        station_data = metropredict[metropredict['name'] == station]  # Data per station filteren
    
        X_train = station_data['Jaar'].values.reshape(-1, 1)  
        y_train = station_data['Passagiers'].values
    
        model.fit(X_train, y_train)  # Train model per station
        X_future = np.arange(2022, 2027).reshape(-1, 1)  # Jaren voor voorspelling
        y_pred = model.predict(X_future)  # Voorspel toekomstige waarden
    
        # **Selecteer een unieke kleur per station**
        color = colors[i % len(colors)]  

        # **Plot historische data**
        fig.add_trace(go.Scatter(
            x=station_data['Jaar'], 
            y=station_data['Passagiers'], 
            mode='lines+markers', 
            name=f"{station} - Historisch",
            line=dict(color=color)
        ))

        # **Plot voorspelling**
        fig.add_trace(go.Scatter(
            x=X_future.flatten(),  
            y=y_pred,  
            mode='lines',  
            name=f"{station} - Voorspelling",
            line=dict(color=color, dash='dash')  
        ))

    # **Layout updaten**
    fig.update_layout(
        title="Voorspelling passagiersaantallen per station",
        xaxis_title="Jaar",
        yaxis_title="Aantal passagiers",
        template="plotly_dark",
        plot_bgcolor='rgba(255, 255, 255, 0.2)',  # Transparante witte achtergrond voor de plot
    paper_bgcolor='rgba(255, 255, 255, 0.1)'  # Transparante volledige figuur
    )

    st.plotly_chart(fig)

    # uitleg over voorspelmodel
    st.markdown('<hr style="border: 1px solid #003082;">', unsafe_allow_html=True)

    st.write("## Verwachtte aantal verhuurde fietsen")
    st.write("Het aantal fietsen wat wordt gebruikt op een dag is grotendeels afhankelijk van het weer. Met de onderstaande sliders kan het 'weer' worden aangepast en is te  zien hoeveel ritten er worden verwacht onder deze omstandigheden. Dit model werkt met een randomforrest prediction die 100 beslisbomen gebruikt.")
    st.write("")
    st.write("De accuraatheid van de voorspelling is weergegeven met de MAE en r^2 score:")
    st.write("•**MAE**: De MAE-score (Mean Absolute Error) meet de gemiddelde absolute fout tussen de voorspelde waarden en de werkelijke waarden. Hoe hoger deze waarde, hoe slechter de voorspelling")
    st.write("")
    st.write("•**R^2**: De R^2-score (ook wel determinatiecoëfficiënt) is een maat voor hoe goed een regressiemodel de variabiliteit in de afhankelijke variabele (bijvoorbeeld het aantal verhuurde fietsen) verklaart op basis van de onafhankelijke variabelen (bijvoorbeeld temperatuur, neerslag, wind). Deze waarde is een score tussen de 0 en 1. Een score van 0.75 bijvoorbeeld betekend dat 75% van de variatie in het aantal ritten wordt verklaard door het weer. De overige 25% wordt veroorzaakt door andere factoren die het model niet meeneemt.")

    # --- Data voorbereiden ---
    relevante_kolommen = ["tmax_c", "neerslag_mm", "windsnelheid_kmh", "aantal_ritjes"]
    fietsweerdata = fietsweerdata[relevante_kolommen]
    # Missing values aanpakken
    fietsweerdata = fietsweerdata.dropna()

    # Train-test split, 80% van de data wordt gebruikt om te trainen, 20% om het te testen
    X = fietsweerdata.drop(columns=["aantal_ritjes"])
    y = fietsweerdata["aantal_ritjes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model trainen --- estimators is het aantal beslisbomen dat wordt gegenereerd
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # laat het model leren met de test data
    model.fit(X_train, y_train)

    # Model evalueren
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Model evaluatie:**\n🔹 MAE: {mae:.2f} 🚲\n🔹 R²-score: {r2:.2f}")

    # --- Voorspelling maken ---
    st.write("## Maak een voorspelling")
    temp = st.slider("Temperatuur (°C)", min_value=-10, max_value=35, value=15)
    regen = st.slider("Neerslag (mm)", min_value=0, max_value=50, value=5)
    wind = st.slider("Windsnelheid (km/h)", min_value=0, max_value=50, value=10)

    # Input omzetten naar dataframe
    input_data = pd.DataFrame([[temp, regen, wind]], columns=["tmax_c", "neerslag_mm", "windsnelheid_kmh"])
    
    # Voorspelling tonen
    voorspelling = model.predict(input_data)[0]
    st.write(f"**Voorspeld aantal verhuurde fietsen:** 🚴‍♂️ {voorspelling:.0f} ritjes per dag")

################################## TAB 4 BOUWEN #############################################

with tab5 :

    st.header(" Conclusie ")
    st.write("Na een uitgebreide analyse van de beschikbare datasets is het duidelijk geworden dat het fietsgebruik in Londen sterk afhankelijk is van het weer. Warme en droge dagen zorgen voor een toename in het aantal ritten, terwijl regen en harde wind juist een negatief effect hebben op het gebruik van deelfietsen. Dit patroon bevestigt dat de keuze voor de fiets grotendeels weersafhankelijk is en niet alleen afhangt van bijvoorbeeld metroverbindingen of infrastructuur.")
    st.write("")
    st.write("Daarnaast wijzen de groeitrends in metrogebruik op enkele stations die de komende jaren significant zullen groeien, namelijk **Bond Street, Canada Water** en **Canning Town**🚇. Dit zijn locaties waar het aantal reizigers zal toenemen, en waar dus ook een grotere vraag naar last-mile transportoplossingen verwacht kan worden. Om in te spelen op deze groei adviseren wij om juist hier extra deelfietsen🚲 te plaatsen, zodat reizigers gemakkelijk en snel hun laatste kilometers kunnen afleggen.")
    st.write("**Let op**: De metro data is wel beïnvloed geweest door de pandemie in 2021")
    st.write("")
    st.write("Deze inzichten kunnen bijdragen aan een strategische uitbreiding van de deelfietsinfrastructuur en een efficiëntere verdeling van middelen binnen Londen.")
# %%
