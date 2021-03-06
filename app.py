import streamlit as st
import pandas as pd
from seaborn import scatterplot, lineplot
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests


def serialize_features(X):
    serial = X.flatten()
    return ",".join([str(i) for i in serial])

def deserialize_features_model1(X_serialized, delimiter=','):
    res = np.array([float(idx) for idx in X_serialized.split(delimiter)])
    return res.reshape(1,-1)

def deserialize_features_model2(X_serialized, n_features=5, deep=20, delimiter=','):
    res = np.array([float(idx) for idx in X_serialized.split(delimiter)])
    return res.reshape(1,deep,n_features)


st.markdown("""#### Problème 1 : Prédire la durabilité d'une batterie à partir de ses 5 premiers cycles""")
st.write('\n')
st.write('\n')
st.write('\n')

df_raw_data_model_one = pd.read_csv('data/raw_data_test_model_one.csv').iloc[1:,2:]
df_X_model_one = pd.read_csv('data/X_test_model_one.csv')
df_y_model_one = pd.read_csv('data/y_test_model_one.csv')

if st.button('Faire une prédiction : Est-ce que la batterie va durer plus de 550 cycles ?', key = 0):
    n = np.random.randint(0,df_raw_data_model_one.shape[0]-1)
    X_sample = df_X_model_one.iloc[n,:].values
    y_true = df_y_model_one.iloc[n].values[0]

    ### call d'API
    params = {
        "X_val_serialized": serialize_features(X_sample),
    }

    server_exist = True
    #url = "http://127.0.0.1:8000"
    url = "https://battery-hrfer72diq-ew.a.run.app"
    response = requests.get(url+"/predict1", params=params)

    if server_exist and response.status_code == 200:
        prediction = response.json()['predict']
        print("prediction (depuis l'api):", prediction)
        print('real:', int(y_true))
    else:
        if server_exist:
            print("API call error", response.status_code)
        model = joblib.load('model_one.joblib')
        X_val = df_X_model_one.iloc[n,:].values.reshape(1, -1)
        prediction = model.predict(X_val)[0]
        print("prediction:", prediction)
        print('real:', int(y_true))

    ### proviroire : à remplacer par le call d'API
    #model = joblib.load('model_one.joblib')
    #prediction = model.predict(df_X_model_one.iloc[n,:].values.reshape(1, -1))[0]
    ###===================

    if prediction == 1:
        st.info(f'Prédiction : Oui, la batterie va durer plus de 550 cycles !')
    else :
        st.warning(f'Prédiction : Non, la batterie ne va pas durer 550 cycles')

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


st.markdown("""#### Problème 2 : Prédire le nombre de cycles de vie restants""")
st.write('\n')
st.write('\n')
st.write('\n')

df_raw_data_model_three = pd.read_csv('data/raw_data_test_model_three.csv').iloc[:,1:]
X_test_model_three = np.genfromtxt('data/X_test_model_three.csv', delimiter = ',')
X_test_model_three = X_test_model_three.reshape(X_test_model_three.shape[0],40,4)
X_test_scaled_model_three = np.genfromtxt('data/X_test_scaled_model_three.csv',delimiter = ',')
X_test_scaled_model_three = X_test_scaled_model_three.reshape(X_test_scaled_model_three.shape[0],40,4)
y_test_model_three = pd.read_csv('data/y_test_model_three.csv')
bc_test_model_three = pd.read_csv('data/bc_test_model_three.csv')

col1, col2 = st.columns(2)
if st.button('Faire une prédiction : Combien de cycles la batterie va-t-elle encore durer ?', key = 1):
    n_features = 4
    deep = 40
    n = np.random.randint(0,X_test_scaled_model_three.shape[0])
    X_sample = X_test_scaled_model_three[n,:,:]
    y_true = int(y_test_model_three.iloc[n,0])

    params = {
        "n_features": n_features,
        "deep": deep,
        "X_val_serialized": serialize_features(X_sample),
    }
    server_exist = True
    #url = "http://127.0.0.1:8000"
    url = "https://battery-hrfer72diq-ew.a.run.app"
    response = requests.get(url+"/predict3", params=params)

    if server_exist and response.status_code == 200:
        prediction_2 = response.json()['predict']
        print("prediction (depuis l'api):", prediction_2)
        print('true:', int(y_true))
    else:
        if server_exist:
            print("API call error", response.status_code)
        #model_2 = joblib.load('model_three.joblib')
        #X_val = X_sample.reshape(1,deep,n_features)
        #prediction_2 = int(model_2.predict(X_val)[0,0])
        #print("prediction:", prediction_2)
        #print('true:', int(y_true))

    ### proviroire : à remplacer par le call d'API
    #model_2 = joblib.load('model_three.joblib')
    #prediction_2 = int(model_2.predict(X_test_scaled_model_three[n,:,:].reshape(1,40,4))[0,0])
    ###===================

    last_cycle = int(X_test_model_three[n,-1,-1])
    barcode = bc_test_model_three.iloc[n,0]

    y_true = int(y_test_model_three.iloc[n,0])
    col1, col2 = st.columns(2)
    col1.info(f'Prédiction du nombre de cycles de vie restants : {prediction_2}')
    col2.info(f'Nombre de cycles de vie restants réels : {y_true}')

    fig, axs = plt.subplots(figsize=(8,2))

    df_data = df_raw_data_model_three[df_raw_data_model_three['barcode'] == barcode]
    scatterplot(y = df_data.iloc[0,1:last_cycle+1].fillna(0), x =np.arange(0,last_cycle,1), ax = axs, color = 'blue', alpha = 0.5)
    scatterplot(y = df_data.iloc[0,1+last_cycle:].fillna(0), x =np.arange(last_cycle,3000,1), ax = axs, color = 'grey', alpha = 0.5)
    lineplot(y = [0.8, 1.2], x = [last_cycle+prediction_2,last_cycle+prediction_2],color = 'red')
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

    if y_true > prediction_2:
        st.info(f"Le modèle a sous-estimé la durée de {y_true - prediction_2} cycles")
    else :
        st.info(f"Le modèle a surestimé la durée de {prediction_2 - y_true} cycles")
