import streamlit as st
import pandas as pd
from random import randint
from seaborn import scatterplot, lineplot
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""#### Problème 1 : Prédire la durabilité d'une batterie avec 5 cycles""")
st.write('\n')
st.write('\n')
st.write('\n')

df_raw_data_model_one = pd.read_csv('data/raw_data_test_model_one.csv').iloc[1:,2:]
df_X_model_one = pd.read_csv('data/X_test_model_one.csv')
df_y_model_one = pd.read_csv('data/y_test_model_one.csv')

n = randint(0,df_X_model_one.shape[0]-1)

col1, col2 = st.columns(2)
if col1.button('Faire une prédiction :'):
    n = randint(0,df_raw_data_model_one.shape[0]-1)
    prediction = n%2
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

# st.line_chart(df_raw_data_model_one.iloc[n,2:].fillna(0).T)
# st.write(df_raw_data_model_one.iloc[n,100:120].fillna(0))
# st.write(df_raw_data_model_one.iloc[n,2:].fillna(0).shape)
