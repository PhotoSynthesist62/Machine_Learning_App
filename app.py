# streamlit run ml_app.py
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# WEB PAGE TITLE
st.title('Machine Learning - Sky Observations')

# FRONT PAGE TITLE
st.write("""
### Classifying Space Observations as Galaxies, Quasars or Stars
Source: Sloan Digital Sky Survey DR14

""")

# READ IN DATA


@st.cache
def load_data(nrows):
    data = pd.read_csv(
        './data/Skyserver_SQL2_27_2018 6_51_39 PM.csv', nrows=nrows)
    return data


# LOAD DATA
data_load_state = st.text('Loading data...')
obsrv = load_data(1000)
data_load_state.text("Data Loaded from CSV (using st.cache)")

# SPILT DATA INTO
df1 = pd.DataFrame(obsrv, columns=['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run',
                                   'camcol', 'class', 'redshift'])

df2 = pd.DataFrame(
    obsrv, columns=['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift'])

# OPTION TO VIEW ABREV. DATA SET
if st.checkbox('Show Data Set'):
    st.subheader('Data Set')
    st.write(df1)


# HISTOGRAM
st.subheader('Histogram')
df2.hist(bins=20)
plt.tight_layout()
st.pyplot()


st.sidebar.header('User Input Parameters')


def user_input_features():
    # run = st.sidebar.slider('Run', 1, 23)
    # camcol = st.sidebar.slider('Camera Column', 1, 6)
    ra = st.sidebar.slider('Right Ascension', 8.235100, 260.884382, 183.531326)
    dec = st.sidebar.slider('Declination', -5.382632, 68.542265, 183.531326)
    u = st.sidebar.slider('U', 12.988970, 18.619355, 183.531326)
    g = st.sidebar.slider('G', 12.799550, 17.371931, 183.531326)
    r = st.sidebar.slider('R', 12.431600, 16.840963, 183.531326)
    i = st.sidebar.slider('I', 11.947210, 28.179630, 183.531326)
    z = st.sidebar.slider('Z', 11.610410, 22.833060, 183.531326)
    redshift = st.sidebar.slider('Redshift', -0.004136, 5.353854, 183.531326)

    d = {
        # 'run': run,
        # 'camcol': camcol,
        'ra': ra,
        'dec': dec,
        'u': u,
        'g': g,
        'r': r,
        'i': i,
        'z': z,
        'redshift': redshift
    }
    specs = pd.DataFrame(d, index=[0])
    return specs


param = user_input_features()

st.subheader('User Input parameters')
st.write(param)

target = df2['class']
target_names = ['GALAXY', 'QSO', 'STAR']

ml_data = df2.drop("class", axis=1)
feature_names = ml_data.columns

X_train, X_test, y_train, y_test = train_test_split(ml_data, target, test_size=.3)
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

prediction = dt_model.predict(param)
prediction_proba = dt_model.predict_proba(param)

st.subheader('Class labels and their corresponding index number')
st.write(target_names)

st.subheader('Prediction')
st.write(target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
