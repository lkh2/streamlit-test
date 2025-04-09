import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="CrowdInsight",
    initial_sidebar_state="collapsed"
)

st.write("CrowdInsight")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #2A5D4E 0%, #65897F 50%, #2A5D4E 100%);
        }  
        [data-testid="stHeader"] {
            background: transparent;
        }  
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    CrowdInsight is a robust web application designed with three primary objectives in mind: 
    to gather and preprocess data from crowdfunding platforms, to predict the success rates 
    of projects using advanced machine learning models, and to offer users a dynamic and 
    intuitive interface for project exploration and improvement.
    """,
    unsafe_allow_html=True
)