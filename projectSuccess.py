import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

import numpy as np

# --------------------
# Load data & model
# --------------------
#@st.cache_data
def load_data():
    # Example: replace with your dataset path
    df = pd.read_excel("PPIData_082025.xlsx")  
    return df

def cLL():
    # Example: replace with your dataset path
    cl= pd.read_excel("country_latlon.xlsx")  
    return cl

@st.cache_resource
def load_model():
    model = joblib.load("DT_082025.joblib")
    return model

df = load_data()
cl=cLL()
model = load_model()

df=df[df["Project status"]!="Active"]
df["status"]=np.where(df["Project status"].isin(["Cancelled","Distressed"]),"Failed","Successful")


df=df.merge(cl,left_on="Country",right_on="country",how="left")

jitter_amount = 0.001  # ~100m
if "lat_jitter" not in st.session_state:
    jitter_amount = 0.001
    st.session_state.lat_jitter = df["lat"].values + np.random.uniform(-jitter_amount, jitter_amount, len(df))
    st.session_state.lon_jitter = df["lon"].values + np.random.uniform(-jitter_amount, jitter_amount, len(df))

df["lat_jitter"] = st.session_state.lat_jitter
df["lon_jitter"] = st.session_state.lon_jitter

df.dropna(subset=["lat","lon"])

# --------------------
# Sidebar Inputs
# --------------------
st.sidebar.header("Input Parameters")

subtype = st.sidebar.selectbox("Subtype of PPI", df["Subtype of PPI"].unique())
subsector = st.sidebar.selectbox("Subsector", ["Roads","Water Utility","ICT","Treatment/ Disposal"])
#option3 = st.sidebar.selectbox("Dropdown 3", df["col3"].unique())
#option4 = st.sidebar.selectbox("Dropdown 4", df["col4"].unique())


# Prepare input for model (example assumes order matches training features)

df1=pd.DataFrame({'Subtype of PPI_Build, own, and operate':[0],
       'Subtype of PPI_Build, rehabilitate, operate, and transfer':[0],
       'Subtype of PPI_Full':[0], 'Subtype of PPI_Lease contract':[0],
       'Subtype of PPI_Management contract':[0], 'Subtype of PPI_Merchant':[0],
       'Subtype of PPI_Partial':[0],
       'Subtype of PPI_Rehabilitate, lease or rent, and transfer':[0],
       'Subtype of PPI_Rehabilitate, operate, and transfer':[0],
       'Subtype of PPI_Rental':[0], 'Subsector_Electricity':[0],
       'Subsector_Electricity, Water Utility':[0], 'Subsector_ICT':[0],
       'Subsector_Natural Gas':[0], 'Subsector_Ports':[0], 'Subsector_Railways':[0],
       'Subsector_Roads':[0], 'Subsector_Treatment plant':[0],
       'Subsector_Treatment/ Disposal':[0], 'Subsector_Water Utility':[0]})



df1["Subtype of PPI"+"_"+subtype]=1
df1["Subsector"+"_"+subsector]=1


df2=df[(df["Subtype of PPI"]==subtype)&(df["Subsector"]==subsector)]

df2=df2.fillna("Unknown")

# --------------------
# Map Visualization
# --------------------
st.subheader("Map of Records")





#df2["lat_jitter"] = df2["lat"] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df2))
#df2["lon_jitter"] = df2["lon"] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df2))

#df2=df.copy(deep=True)



#df2=df2[~df2.isna()]
if not df2.empty:
    m = folium.Map(location=[df2["lat"].mean(), df2["lon"].mean()], zoom_start=1)

    for _, row in df2.iterrows():
        color = "green" if row["status"] == "Successful" else "red"
        folium.CircleMarker(
            location=[row["lat_jitter"], row["lon_jitter"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Project Name:{row['Project name']}<br>Status: {row['status']}<br>Subsector:{row['Subsector']}"
        ).add_to(m)

        

    # Show map in Streamlit


    st_data = st_folium(m, width=700, height=500)
else:
    st.write("No available data to show")

# --------------------
# Metric Box
# --------------------
st.metric(label="No. of Past Records", value=round(df2.shape[0]+0.0001))
st.metric(label="Failure Rate", value=round((df2[df2["status"]=="Failed"].shape[0]+0.0001)/(df2.shape[0]+0.0001),2))

# --------------------
# Prediction Section
# --------------------
st.subheader("Prediction")



# Prediction
prediction = model.predict(df1)

y=[]
if prediction==True:
    y="Success Likely"
else:
    y="Failure Likely"

st.write(f"### Predicted Outcome: **{y}**")
