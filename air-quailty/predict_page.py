import streamlit as st
import pickle
import pandas as pd
import numpy as np
from prophet import Prophet

def load_model():
    with open('prophet_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    return forecast

def main():
    st.title('Air Quality Index Prediction')

    st.write("""## Input the number of hours for forecasting""")
    periods = st.number_input('Number of hours to forecast:', min_value=1, max_value=168, value=24)

    if st.button('Predict'):
        with st.spinner('Loading model...'):
            model = load_model()
        
        with st.spinner('Making forecast...'):
            forecast = make_forecast(model, periods)
        
        st.success('Forecast completed!')
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

        st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

if __name__ == "__main__":
    main()

