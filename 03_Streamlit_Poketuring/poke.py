import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('pokemon.csv')
df.loc[df["capture_rate"] == '30 (Meteorite)255 (Core)', "capture_rate"] = '30'
df = df.astype({'capture_rate' : 'int64'})

st.set_page_config(page_icon="poke_pika_wink.png", page_title="PokeProject")

#título da página e imagem da pokebola
a, b = st.columns([1, 6])
with a:
    st.text("")
    st.text("")
    st.image("pokeball.png", width= 100)
with b:
    st.image("pokeprojeto.png", width= 600)

#visualização do dataframe
st.dataframe(df)

#visualização de pokémons

st.image('whopoke.jpg', width = 700)
st.image('pokewho.png', width = 700)

st.markdown('#### **Gerações de Pokemóns:** ') #gerações
generation = st.selectbox(
     'Escolha uma ou mais geração',
     (df['generation'].unique()))
st.write(f'Você escolheu ver Pokemóns da geração {generation}!')
gen_df = df[df['generation'] == generation]
gen_df.set_index(['generation'], inplace = True)
st.dataframe(gen_df)

st.markdown('#### **Pokemons lendários:**') #pokemóns lendários
legendary_df = df[df['is_legendary'] == 1]
legendary_df.set_index(['name'], inplace = True)
st.dataframe(legendary_df)

st.markdown('#### **Curiosidades :**') #curiosidades

#ajusta o número da pokedex para visualizar as imagens do repositório github
def adjust_pokedex_number(pokedex_number):
    pn = int(pokedex_number)
    
    if(pn < 10):
        s_pn = '00' + str(pn)
    elif(pn < 100):
        s_pn = '0' + str(pn)
    else:
        s_pn = str(pn)
    return s_pn

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"**Maior altura:** {df['height_m'].max()}m")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/321.png")
    st.markdown("_Wailord_")
with col2:
    st.markdown(f"**Menor altura:** {df['height_m'].min()}m")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/595.png")
    st.markdown("_Joltik_")
with col3:
    st.markdown(f"**Maior peso:** {df['weight_kg'].max()}kg")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/797.png")
    st.markdown("_Celesteela_")
with col4:
    st.markdown(f"**Menor peso:** {df['weight_kg'].min()}kg")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/798.png")
    st.markdown("_Kartana_")
with col5:
    st.markdown(f"**Geração mais comum:** 5")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/496.png")
    st.markdown("_Servine_")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"**Maior velocidade:** {df['speed'].max()}")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/386.png")
    st.markdown("_Deoxys_")
with col2:
    st.markdown(f"**Menor velocidade:** {df['speed'].min()}")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/213.png")
    st.markdown("_Shuckle_")
with col3:
    st.markdown("**Tipo 1 mais comum:** Water")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/054.png")
    st.markdown("_Psyduck_")
with col4:
    st.markdown("**Tipo 2 mais comum:** Flying")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/017.png")
    st.markdown("_Pidgeotto_")
with col5:
    st.markdown("**Habilidade mais comum:** Levitate")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/329.png")
    st.markdown("_Vibrava_")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"**Tipo 1 mais forte:** Dragon") #maior ataque médio
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/373.png")
    st.markdown("_Salamence_")
with col2:
    st.markdown(f"**Tipo 1 mais fraco:** Fairy") #menor ataque médio
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/036.png")
    st.markdown("_Clefable_")
with col3:
    st.markdown("**Tipo 2 mais forte:** Fighting")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/500.png")
    st.markdown("_Emboar_")
with col4:
    st.markdown("**Tipo 2 mais fraco:** Normal")
    st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/667.png")
    st.markdown("_Litleo_")
with col5:
    st.markdown("**Pokémon do dia:**")
    st.write("")
    pkn = np.random.choice(range(1,808))
    pokedex_number = adjust_pokedex_number(pkn)
    im_dir = 'https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/'
    png = '.png'
    image = im_dir + pokedex_number + png
    st.image(image)
    name = list(df[df['pokedex_number'] == pkn]['name'])[0]
    st.markdown(f"_{name}_")

st.write("")

#visualização de gráficos
import plotly.express as px

st.image("pokegraficos.png", width = 300)

col1, col2 = st.columns(2)
with col1: 
    st.markdown('#### Tipos de Pokemón segundo Geração:')
    chosen_type = st.selectbox("Escolha um typo: ", ["type1", "type2"])
    fig = px.histogram(df, x= chosen_type, color= "generation")
    st.plotly_chart(fig, use_container_width=True)

with col2: 
    st.markdown('#### Pokemóns Lendários segundo Geração:') #gráfico 3
    fig = px.histogram(df, x= "generation", color= "is_legendary", barmode='group')
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('#### Qual a influência das base_stats no peso e altura dos Pokémons?')
chosen_base_stat = st.radio("Escolha uma base_stat", ["base_egg_steps", "base_happiness"])
fig = px.scatter_3d(df, x='weight_kg', y='height_m', z='is_legendary', color= chosen_base_stat)
st.plotly_chart(fig)

st.markdown('#### Veja as relações acima de outra forma:')

fig = px.scatter(df, x="weight_kg", y="height_m", color = 'base_egg_steps', facet_col = 'is_legendary')
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(df, x="weight_kg", y="height_m", color = 'base_egg_steps', facet_col = 'is_legendary')
st.plotly_chart(fig, use_container_width=True)

#modelo preditivo de pokemóns lendários na barra lateral esquerda
st.sidebar.image("pokevision.png", width= 300)

st.sidebar.markdown('#### Não perca seu tempo! Faça como a Wanda: crie sua PokéRealidade e preveja se seu Pokémon será lendário ou não.') #predição 1

poke_trainer = st.sidebar.text_input('Qual o seu nome, treinador?', 'Wanda Maximoff')
st.sidebar.write('O seu nome é', poke_trainer, '.')

poke_name = st.sidebar.text_input('Qual o nome do seu PokéMon?', 'Visão')
st.sidebar.write('O nome do seu PokéMon é', poke_name, '.')

sp_attack = st.sidebar.slider('Qual o ataque especial do seu Pokemón?', 10, 194, 25)
st.sidebar.write("Seu Pokemón terá um ataque especial de ", sp_attack, '.')

capture_rate = st.sidebar.slider('Qual a taxa de captura do seu Pokemón?', 3, 255, 15)
st.sidebar.write("Seu Pokemón terá uma taxa de captura de ", capture_rate, '.')

base_egg_steps = st.sidebar.slider('Qual o número de passos necessário para chocar um ovo do seu Pokemón?', 1280, 30720, 6400)
st.sidebar.write("Para chocar um ovo do seu Pokemón, serão necessários", base_egg_steps, 'passos.')

new_dict = {'base_egg_steps' : [base_egg_steps], 'capture_rate' : [capture_rate], 'sp_attack' : [sp_attack]}
new_poke = pd.DataFrame.from_dict(new_dict)

#treina o modelo preditivo de pokemóns lendários
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
knn = KNeighborsClassifier()
pipeline = Pipeline(steps = [('scaler', scaler), ('knn', knn)])
X = df[['base_egg_steps', 'capture_rate', 'sp_attack']]
y = df[['is_legendary']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
pipeline.fit(X_train, np.ravel(y_train))
prediction = int(pipeline.predict(new_poke))

import base64

#visualização do resultado do modelo
if prediction == 1:
    st.sidebar.success(f'Que demais! {poke_name} é um Pokemon lendário!')
    file_ = open("wandadancing.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.sidebar.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="wanda dancing gif">',
        unsafe_allow_html=True,
    )
    st.sidebar.write("")
else:
    st.sidebar.error(f'Puxa... {poke_name} não é um Pokemon lendário...')

    file_ = open("mondays.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.sidebar.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="wandavision gif">',
        unsafe_allow_html=True,
    )
    st.sidebar.write("")

#cálculos das informações do pokemón a ser criado

#porcentagem masculina
random_male_percentage = np.random.random_sample() 
def male_percentage(random_male_percentage):
    male_percentage = random_male_percentage*100
    male_percentage = round((male_percentage), 1)
    if male_percentage < 24.6:
        male_percentage = 0.0
    elif male_percentage > 88.1: 
        male_percentage = 100.0
    return male_percentage
percentage_male = male_percentage(random_male_percentage)

pokeframe = df.astype('str')
columns =['against_bug', 'against_electric', 'against_fairy',
       'against_fight', 'against_fire', 'against_flying', 'against_ghost',
       'against_grass', 'against_ground', 'against_ice', 'against_normal',
       'against_psychic', 'against_rock', 'against_steel']
new_pokemon = {'trainer' : poke_trainer, 'name': poke_name, 'generation': 'Turing', 'classfication' : 'Turing Pokémon', 
'percentage_male': str(percentage_male), 'pokedex_number' : '802'}

#tipos 1 e 2
df['type2'].fillna('Não tem', inplace = True)
values_type1 = list(pokeframe['type1'].unique())
chosen_type1 = np.random.choice(values_type1)
values_type2 = list(pokeframe['type2'].unique())
values_type2.remove(chosen_type1)
chosen_type2 = np.random.choice(values_type2)
dict_partial = {'type1' : chosen_type1, 'type2' : chosen_type2}
new_pokemon.update(dict_partial)

#abilidades
ab_dict = {'grass' : ['Chlorophyll', 'Effect Spore', 'Grassy Surge', 'Harvest', 'Leaf Guard', 'Overgrow'], 
'fire' : ['Blaze', 'Flame Body', 'Magma Armor', 'White Smoke'], 
'water' : ['Drizzle', 'Mega Launcher', 'Torrent', 'Water Veil'],
'bug' : ['Compound Eyes', 'Swarm', 'Shield Dust', 'Hyper Cutter', 'Moxie'],
'normal' : ['Fluffy', 'Aerilate', 'Galvanize', 'Normalize', 'Pixilate', 'Refrigerate', 'Scrappy'],
'poison' : ['Liquid Ooze', 'Stench', 'Dry Skin', 'Poison Touch'],
'electric' : ['Eletric Surge', 'Static', 'Motor Drive', 'Volt Absorb'],
'ground' : ['Arena Trap', 'Sand Rush', 'Drought', 'Immunity'],
'fairy' : ['Aroma Veil', 'Flower Veil', 'Arena Surge', 'Misty Surge', 'Pixilate', 'Cute Charm'],
'fighting' : ['Inner Focus', 'Mold Breaker', 'Guts', 'Iron Fist', 'Scrappy'],
'psychic' : ['Forewarn', 'Psychic Surge', 'Dazzling', 'Keen Eye'],
'rock' : ['Rock Head', 'Sturdy', 'Weak Armor', 'Sand Stream', 'Sand Force'],
'ghost' : ['Cursed Body', 'Infiltrator', 'Clear Body', 'Pressure', 'Frisk', 'Levitate'], 
'ice' :['Slush Rush', 'Regrigerate', 'Snow Cloack', 'Snow Warning'],
'dragon' : ['Rivalry', 'Rough Skin', 'Unnerve', 'Sheer Force'],
'dark' : ['Bad Dreams', 'Dark Aura', 'Illusion', 'Justified', 'Super Luck'], 
'steel' : ['Heavy Metal', 'Iron Barbs', 'Light Metal', 'Magnet Pull'],
'flying' : ['Aerilate', 'Big Pecks', 'Multiscale', 'Competitive']}
ab_gen = ['Berserk', 'Battle Armor', 'Magician']

def def_abs(type1, type2):
    first_type = np.random.choice(range(1,5))
    sec_type = gen_type = 0
    if type2 != 'Não tem':
        sec_type = 4 - first_type
        abs_1 = np.random.choice(ab_dict[type1], first_type, replace = False)
        if(sec_type != 0): 
            abs_2 = np.random.choice(ab_dict[type2], sec_type, replace = False)
    else:
        gen_type = 4 - first_type
        abs_1 = np.random.choice(ab_dict[type1], first_type, replace = False)
        if(gen_type != 0):
            abs_2 = np.random.choice(ab_gen, gen_type, replace = False)
    if(type2 != 'Não tem' and sec_type != 0 or gen_type != 0):
        abilities = np.concatenate((abs_1, abs_2))
    else:
        abilities = abs_1
    return abilities

abilities = def_abs(new_pokemon['type1'], new_pokemon['type2'])
abilities = ', '.join(abilities)
new_pokemon.update({'abilities' : abilities})

#escolha aleatória de outras características
for poke in columns:
    values = pokeframe[poke].unique()
    chosen = np.random.choice(values)
    dict_partial = {poke : chosen}
    new_pokemon.update(dict_partial)

#funções que definem alguns valores
def define_dark_water_values(predicted_value):
    x = predicted_value
    if x < 1 and x >= 0.25:
        x = 0.5
    elif x < 0.25:
        x = 0.25
    elif x >=1 and x < 1.5:
        x = 1.0
    elif x >= 1.5 and x < 2 or x >= 2 and x < 3:
        x = 2.0
    else:
        x = 4.0
    return x

def define_dragon_values(predicted_value):
    x = predicted_value
    if x <= 0 or x < 0.25:
        x = 0.0
    elif x >= 0.25 and x <= 0.5 or x > 0.5 and x < 0.75:
        x = 0.5
    elif x > 0.75 and x <= 1 or x > 1 and x < 1.5:
        x = 1.0
    else:
        x = 2.0
    return x

def define_poison_values(predicted_value):
    x = predicted_value
    if x <= 0 or x < 0.125:
        x = 0.0
    elif x < 1 and x >= 0.25:
        x = 0.5
    elif x >= 0.125 and x < 0.25:
        x = 0.25
    elif x >=1 and x < 1.5:
        x = 1.0
    elif x >= 1.5 and x < 2 or x >= 2 and x < 3:
        x = 2.0
    else:
        x = 4.0
    return x

#predição de algumas outras características
from sklearn.linear_model import Lasso

def lr_model(features, target, alpha_value, features_values, activation_value = 0):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    lr = Lasso(alpha = alpha_value)
    pipeline = Pipeline(steps = [('scaler', scaler), ('lr', lr)])
    pipeline.fit(X_train, y_train)
    new_dict = {}
    for feature, value in zip(features, features_values):
        new_dict.update({feature : [value]})
    new_poke = pd.DataFrame.from_dict(new_dict)
    prediction = pipeline.predict(new_poke)
    if(activation_value):
        prediction = int(prediction)
    return prediction

prediction_dark = lr_model(['against_ghost'], ['against_dark'], 0.01, [new_pokemon['against_ghost']])
prediction_dark = define_dark_water_values(prediction_dark)

prediction_dragon = lr_model(['against_fairy', 'against_ice'], ['against_dragon'], 0.01, [new_pokemon['against_fairy'], new_pokemon['against_ice']])
prediction_dragon = define_dragon_values(prediction_dragon)

prediction_poison = lr_model(['against_normal'], ['against_poison'], 0.0001, [new_pokemon['against_normal']])
prediction_poison = define_poison_values(prediction_poison)


prediction_water = lr_model(['against_ground'], ['against_water'], 0.0001, [new_pokemon['against_ground']])
prediction_water= define_dark_water_values(prediction_water)

base_total = lr_model(['sp_attack', 'base_egg_steps', 'is_legendary'], ['base_total'], 5, [sp_attack, base_egg_steps, prediction], 1)
base_happines = lr_model(['sp_attack', 'base_egg_steps', 'is_legendary', 'base_total'], ['base_happiness'], 0.01, [sp_attack, base_egg_steps, prediction, base_total], 1)

sp_defense = lr_model(['sp_attack', 'base_total'], ['sp_defense'], 1, [sp_attack, base_total], 1)
defense = lr_model(['sp_defense', 'base_total'], ['defense'], 1, [sp_defense, base_total], 1)

attack = lr_model(['defense', 'base_total'], ['attack'], 2, [defense, base_total], 1)

hp = lr_model(['attack', 'base_total'], ['hp'], 1, [attack, base_total], 1)

df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].mean())
weight_kg = lr_model(['base_egg_steps', 'base_total'], ['weight_kg'], 1, [base_egg_steps, base_total])

df['height_m'] = df['height_m'].fillna(df['height_m'].mean())
height_m = lr_model(['weight_kg', 'base_total'], ['height_m'], 0.0001, [weight_kg, base_total])

speed = lr_model(['sp_attack', 'base_total'], ['speed'], 0.001, [sp_attack, base_total], 1)

experience_growth = lr_model(['is_legendary', 'base_total'], ['experience_growth'], 1, [prediction, base_total], 1)

#reajuste de alguns valores predizidos
def impede_menor(sp_x, x):
    if (sp_x < x):
        value = sp_x
        sp_x = x
        x = value
    
    return sp_x, x

sp_defense, defense = impede_menor(sp_defense, defense)
sp_attack, attack = impede_menor(sp_attack, attack)

def impede_altura(altura, peso):
    alt = altura[0]
    pes = peso[0]
    if alt == 0.0:
        frac = np.random.random_sample() 
        alt = frac*pes
    alt = np.round(alt, 1)
    pes = np.round(pes, 1)
    return alt, pes

height_m, weight_kg = impede_altura(height_m, weight_kg)

def legendary(pred):
    if pred == 1:
        answer = 'Legendary'
    else:
        answer = 'Non legendary'
    return answer

is_legendary = legendary(prediction)

#finaliza a adição das últimas características ao perfil do pokémon criado
dict_partial = {'against_dark' : str(prediction_dark),'against_dragon' : str(prediction_dragon), 'against_poison': str(prediction_poison),
'against_water' : str(prediction_water), 'weight_kg' : weight_kg, 'height_m' : height_m, 'speed' : speed, 'hp' : hp, 
'experience_growth' : experience_growth, 'base_egg_steps' : base_egg_steps, 'base_happines' : base_happines, 'base_total' : base_total,
 'capture_rate' : capture_rate, 'attack' : attack, 'sp_attack' : sp_attack, 'defense' : defense, 'sp_defense' : sp_defense, 
 'is_legendary' : is_legendary}
new_pokemon.update(dict_partial)
keys_values = new_pokemon.items()
new_pokemon = {str(key): str(value) for key, value in keys_values}
new_pokemon = pd.Series(new_pokemon)


# Acima, foram calculadas as características do pokémon criado. Mas, elas só serão postas à visualização mais pra frente.
# Logo abaixo, vamos visualizar mais relações na forma de gráficos.



st.subheader('Quais fatores tornam um Pokemón lendário ou não?') 

chosen_column = st.selectbox("Escolha uma categoria: ", (df.columns.drop('is_legendary')))

fig = px.box(df, x="is_legendary", y= chosen_column)
st.plotly_chart(fig, use_container_width=True)

# Agora sim, vamos visualizar o Pokémon criado anteriormente.

poke_button = 0
poke_button = st.sidebar.button('Crie seu Pokemon')

col1, col2 = st.columns(2)
with col1:
    st.image("meupokemon.png", width = 300)
    if poke_button:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.image("https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/802.png")
        st.write(new_pokemon)
    else:
        st.image("ashpika.png", width = 300)
        file_ = open("pikacry.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.error("Você não criou nenhum PokéMon ainda...")
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="pikachu crying gif">',
            unsafe_allow_html=True,
        )
with col2: 
    st.image("pokedex.png", width = 200)
    pokemon = 'Squirtle'
    pokemon = st.selectbox(
        'Escolha um pokemon',
        (df['name'].unique()))
    pokeframe = df.astype('str')
    pokedex = pokeframe.set_index('name').T.to_dict('series')
    chosen_pokemon = pokedex[pokemon]
    from PIL import Image
    pokedex_number = adjust_pokedex_number(chosen_pokemon['pokedex_number'])
    im_dir = 'https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/thumbnails-compressed/'
    png = '.png'
    image = im_dir + pokedex_number + png
    st.image(image)
    st.write(chosen_pokemon)