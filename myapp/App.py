import streamlit as st
import joblib


def app():
    st.set_page_config(page_title="Fishcatch", page_icon="🐟")

    col1, col2 = st.columns(2)

    with col1:
        st.title('Fishcatch')

    with col2:
        st.markdown(
            """
            <div style="text-align: center">
                <img src="https://imgur.com/h7ppyOC.png" width="140">
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.write('Predicting the width of a fish based on its features.')

    st.markdown(
        '''
        by [joaoalexarruda](https://github.com/joaoalexarruda)
        '''
    )

    try:
        loaded_model = joblib.load('linear_regression_model.pkl')
    except FileNotFoundError:
        loaded_model = joblib.load('../linear_regression_model.pkl')

    st.divider()

    st.image('https://imgur.com/JU41HUE.png')

    st.write('### Make a prediction:')

    length1_min = 7.5
    length1_max = 59.0
    length2_min = 8.4
    length2_max = 63.4
    length3_min = 8.8
    length3_max = 68.0
    height_min = 1.7284
    height_max = 18.957

    length1 = st.number_input(
        'Length1', min_value=length1_min, max_value=length1_max, value=length1_min)
    length2 = st.number_input(
        'Length2', min_value=length2_min, max_value=length2_max, value=length2_min)
    length3 = st.number_input(
        'Length3', min_value=length3_min, max_value=length3_max, value=length3_min)
    height = st.number_input(
        'Height', min_value=height_min, max_value=height_max, value=height_min)

    if st.button('Predict'):
        prediction = loaded_model.predict(
            [[length1, length2, length3, height]])
        st.write(f'#### Predicted Width: {prediction[0]:.2f}')


app()
