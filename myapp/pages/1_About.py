import streamlit as st
import pandas as pd

def about():
    st.set_page_config(
        page_title="About",
        page_icon="🐟",
    )

    st.title('About Fishcatch')
    
    st.markdown(
        '''
        This app is part of an assignment for the post graduation course ['Data Science and Digital Transformation'](https://www.ipportalegre.pt/pt/oferta-formativa/pos-graduacao-data-science-and-digital-transformation) at the Instituto Politécnico de Portalegre (IPP).

The goal is to apply machine learning techniques to the dataset and to evaluate the results. In this case, regression techniques such as linear and polynomial regression will be used to predict the width of the fish based on the other variables.

[Click here to see the code](https://github.com/joaoalexarruda/fishcatch)

'''
    
    )
    
    
about()