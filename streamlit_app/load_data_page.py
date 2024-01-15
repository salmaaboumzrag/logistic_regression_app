import streamlit as st
import pandas as pd

def load_data(upload_file):
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        return df
    else:
        return None

def render():
    st.title("Chargement du Dataset")
    upload_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    if upload_file is not None:
        data = load_data(upload_file)
        if data is not None:
            st.write("Aperçu du Dataset:")
            st.write(data)
            st.session_state["data"] = data
