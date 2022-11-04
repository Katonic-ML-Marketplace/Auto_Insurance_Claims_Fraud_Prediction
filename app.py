import pandas as pd 
import plotly.graph_objects  as go
import plotly.express as px
import pickle
from PIL import Image
import streamlit as st

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier 

im = Image.open('image/favicon.ico')

st.set_page_config(
    page_title='Auto Insurance Claims Fraud Prediction', 
    page_icon = im, 
    layout = 'wide', 
    initial_sidebar_state = 'auto'
)

st.sidebar.image('image/logo.png')
st.sidebar.title('Auto Insurance Claims Fraud Prediction')
st.sidebar.write('---')

st.write("""
# Auto Insurance Claims Fraud Prediction
This app predicts which **clients are highly likely to commit fraudulent activities.!**
""")
st.write('---')

# Loads Dataset

original_df = pd.read_csv('data/insurance fraud claims.csv', encoding = 'ISO-8859-1')
st.write('**Client Information**')
st.write(original_df.head(20))
st.write('---')

# showing fig1
st.subheader('Percentage of Client involved in fraudulent activities')
labels = 'Fraud', 'Not-Fraud'
sizes = [original_df.fraud_reported[original_df['fraud_reported']=='Y'].count(), original_df.fraud_reported[original_df['fraud_reported']=='N'].count()]
fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, pull=[0, 0.05, 0, 0])])
fig.update_traces(marker=dict(colors=['indigo', 'forestgreen']))
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
st.plotly_chart(fig, use_container_width=False)

# Incident_state
st.subheader('States involved ')
ds = original_df.groupby(['incident_state']).fraud_reported.count()
fig = px.bar(data_frame=ds)
fig.update_layout(margin=dict(t=50, b=50, l=20, r=50)) 
st.plotly_chart(fig, use_container_width=False)

# Incident_city
st.subheader('States involved ')
ds = original_df.groupby(['incident_city']).fraud_reported.count()
fig = px.bar(data_frame=ds)
fig.update_layout(margin=dict(t=50, b=50, l=20, r=50)) 
st.plotly_chart(fig, use_container_width=False)


# Incident_type
st.subheader('Incident Types involved')
ds = original_df.groupby(['incident_type']).fraud_reported.count()
fig = px.bar(data_frame=ds)
fig.update_layout(margin=dict(t=50, b=50, l=20, r=0)) 
st.plotly_chart(fig, use_container_width=False)


#Authorities Contacted
st.subheader('Authorities Contacted')
ds = original_df.groupby(['authorities_contacted']).fraud_reported.count()
fig = px.bar(data_frame=ds)
fig.update_layout(margin=dict(t=50, b=50, l=20, r=0)) 
st.plotly_chart(fig, use_container_width=False)

# Loads Required Dataset

client_train = pd.read_csv('data/data_req.csv', encoding = 'ISO-8859-1')
st.write('**Client Information**')
st.write(client_train.head(20))
st.write('---')


def user_input_features():
    
    st.sidebar.subheader('Client Details')
    months_as_customer_groups = st.sidebar.number_input('months_as_customer', 1, 344700, 51471)
    policy_deductable = st.sidebar.number_input('policy deductable',1, 10000, 1 )
    umbrella_limit = st.sidebar.selectbox('umbrella Limit',original_df['umbrella_limit'].unique())
    insured_hobbies_new =  st.sidebar.selectbox('Insured Hobbies', original_df['insured_hobbies'].unique())
    incident_type = st.sidebar.selectbox('incident Type', original_df['incident_type'].unique())
    collision_type_new = st.sidebar.selectbox('collision Type', original_df['collision_type'].unique())
    incident_severity = st.sidebar.selectbox('Incident Severity',original_df['incident_severity'].unique() )
    authorities_contacted = st.sidebar.selectbox('Authorities Contacted', original_df['authorities_contacted'].unique())
    number_of_vehicles_involved = st.sidebar.number_input('Vehicles Involved',1, 50, 1 )
    bodily_injuries = st.sidebar.number_input('Bodily Injuries',1, 50, 1 )
    witnesses = st.sidebar.number_input('witnesses',1, 50, 1 )
    police_report_available_new = st.sidebar.selectbox('Police Report Available',original_df['police_report_available'].unique())
    vehicle_claim_groups = st.sidebar.number_input('Vehicle Claim ',1, 155648, 1 )

    client = {
        'months_as_customer_groups':  months_as_customer_groups,
        'policy_deductable':  policy_deductable,
        'umbrella_limit': umbrella_limit,
        'insured_hobbies_new': insured_hobbies_new,
        'incident_type': incident_type,
        'collision_type_new': collision_type_new,
        'incident_severity': incident_severity,
        'authorities_contacted': authorities_contacted,
        'number_of_vehicles_involved': number_of_vehicles_involved,
        'bodily_injuries': bodily_injuries,
        'witnesses': witnesses,
        'police_report_available_new': police_report_available_new,
        'vehicle_claim_groups': vehicle_claim_groups,
    }
   
    return pd.DataFrame(client, index=[0])

p = user_input_features()

# Print specified input parameters
st.header('Specified Input parameters')
st.subheader('Client details')
st.write(p)
st.write('---')

# label endcoding for the object datatypes.

obj_features = ['insured_hobbies_new', 'collision_type_new', 'months_as_customer_groups', 'incident_severity',
'vehicle_claim_groups', 'incident_type', 'authorities_contacted' ,'police_report_available_new']

def get_fit_transform(df):
    for col in obj_features:
        if(df[col].dtype == 'object'):
            le = preprocessing.LabelEncoder()
            le = le.fit(df[col])
            df[col] = le.transform(df[col])
    return df

client_df = get_fit_transform(client_train)
input_df = get_fit_transform(p)

sc = StandardScaler()
data_scaled = sc.fit_transform(client_df)
input_data_scaled = sc.fit_transform(input_df)

# fraud_reported is our target column. We will convert it to 1 and 0 and build the target dataframe.
original_df['fraud_reported'] = original_df['fraud_reported'].str.replace('Y', '1')
original_df['fraud_reported'] = original_df['fraud_reported'].str.replace('N', '0')
original_df['fraud_reported'] = original_df['fraud_reported'].astype(int)
data_target = original_df['fraud_reported']


# Random Forest with Bagging Classifier                 

#to retrain
agree = st.checkbox('Check to retrain the model')
filename = 'model/final_model.sav'
if agree:
    # Build  Model
    model = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(),
                                 sampling_strategy = 'auto',
                                 replacement = False,
                                 random_state = 0)

    model.fit(data_scaled, data_target)
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
else:
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))

if st.sidebar.button("Prediction"):
    prediction = model.predict(input_data_scaled)[0]
    st.header('Fraud Detection Predictions')
    if prediction == 1:
        st.success('The claim reported is fraud.')
    else:
        st.success('The claim reported is genuine.')

else:
    st.warning("Please Click on Prediction")
st.write('---')
