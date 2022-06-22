import streamlit as st
import pandas as pd
from seaborn import scatterplot, lineplot
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.markdown("""#### Problème 1 : Prédire la durabilité d'une batterie à partir de ses 5 premiers cycles""")
st.write('\n')
st.write('\n')
st.write('\n')

df_raw_data_model_one = pd.read_csv('data/raw_data_test_model_one.csv').iloc[1:,2:]
df_X_model_one = pd.read_csv('data/X_test_model_one.csv')
df_y_model_one = pd.read_csv('data/y_test_model_one.csv')

col1, col2 = st.columns(2)
if col1.button('Faire une prédiction :'):
    n = np.random.randint(0,df_raw_data_model_one.shape[0]-1)

    ### proviroire : à remplacer par le call d'API
    model = joblib.load('model_one.joblib')
    prediction = model.predict(df_X_model_one.iloc[n,:].values.reshape(1, -1))[0]
    ###===================

    if prediction == 1:
        col2.info(f'Prédiction du modèle : {prediction}')
    else :
        col2.warning(f'Prédiction du modèle : {prediction}')

    fig, axs = plt.subplots(figsize=(8,2))

    scatterplot(y = df_raw_data_model_one.iloc[n,:].fillna(0), x =np.arange(0,3000,1), ax = axs, color = 'blue', alpha = 0.5)
    lineplot(y = [0.8, 1.2], x = [550,550],color = 'red')
    axs.set_ylim(0.8,1.2)
    axs.set_yticks([0.8,0.9,1.0,1.1,1.2])
    axs.set_xlim(0,2500)
    axs.set_xticks(range(0,2501,250))
    axs.set_ylabel('Capacity (in Ah)')
    axs.set_xlabel('Number of cycle')
    axs.tick_params(axis='both', which='major', labelsize=10)
    st.write('\n')
    col1, col2, col3 = st.columns(3)
    col2.write('##### Courbe réelle')
    st.pyplot(fig)
    resultat = df_y_model_one.iloc[n].values[0]
    if resultat == prediction:
        st.success('Prédiction réussie')
    else :
        st.error('Prédiction fausse')

st.write('')
st.write('')
st.write('')
col1, col2, col3 = st.columns(3)
col2.markdown("""# ***""")


st.markdown("""#### Problème 2 : Prédire le nombre de cycles de vie restant""")
st.write('\n')
st.write('\n')
st.write('\n')

df_raw_data_model_three = pd.read_csv('data/raw_data_test_model_three.csv').iloc[1:,2:]
X_test_model_three = np.genfromtxt('data/X_test_model_three.csv')
X_test_model_three.reshape(X_test_model_three.shape[0],40,4)
X_test_scaled_model_three = np.genfromtxt('data/X_test_scaled_model_three.csv')
X_test_scaled_model_three.reshape(X_test_scaled_model_three.shape[0],40,4)
y_test_model_three = pd.read_csv('data/y_test_model_three.csv')
st.write(X_test_model_three.shape)
st.write(X_test_scaled_model_three.shape)
st.write(y_test_model_three.shape)

# col1, col2 = st.columns(2)
# if col1.button('Faire une prédiction :'):
#     n_2 = np.random.randint(0,df_raw_data_model_three.shape[0]-1)

#     ### proviroire : à remplacer par le call d'API
#     model_2 = joblib.load('model_three.joblib')
#     prediction_2 = model_2.predict(df_X_test_scaled_model_three.iloc[n_2,:].values.reshape(1, -1))[0]
#     ###===================

#     col2.info(f'Prédiction du modèle : {prediction_2}')

#     fig, axs = plt.subplots(figsize=(8,2))

#     last_cycle = df_X__test_model_three[n_2,-1,-1]
#     scatterplot(y = df_raw_data_model_three.iloc[n_2,:last_cycle].fillna(0), x =np.arange(0,3000,1), ax = axs, color = 'blue', alpha = 0.5)
#     scatterplot(y = df_raw_data_model_three.iloc[n_2,last_cycle:].fillna(0), x =np.arange(0,3000,1), ax = axs, color = 'grey', alpha = 0.5)
#     lineplot(y = [0.8, 1.2], x = [last_cycle+prediction_2,last_cycle+prediction_2],color = 'red')
#     axs.set_ylim(0.8,1.2)
#     axs.set_yticks([0.8,0.9,1.0,1.1,1.2])
#     axs.set_xlim(0,2500)
#     axs.set_xticks(range(0,2501,250))
#     axs.set_ylabel('Capacity (in Ah)')
#     axs.set_xlabel('Number of cycle')
#     axs.tick_params(axis='both', which='major', labelsize=10)
#     st.write('\n')
#     col1, col2, col3 = st.columns(3)
#     col2.write('##### Courbe réelle')
#     st.pyplot(fig)
#     resultat = df_y_model_three.iloc[n_2].values[0]
#     st.write(f"Le modèle fait une erreur de {resultat - }")
#         st.success('Prédiction réussie')
#     else :
#         st.error('Prédiction fausse')
