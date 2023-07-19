#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle

import sklearn


# In[2]:


import numpy as np
import pandas as pd



# In[3]:


# Load the pickled model
with open('random_forest_model.pickle', 'rb') as file:
    loaded_model = pickle.load(file)


# In[4]:


# Load the pickled model
with open('random_forest_model.pickle', 'rb') as file:
    loaded_model = pickle.load(file)

# Define a function to preprocess input data
def preprocess_input(carat, cut, color, clarity, depth, table):
    # Create a dataframe with the input data
    data = pd.DataFrame({'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity],
                         'depth': [depth], 'table': [table]})
    
    # Perform any necessary encoding or preprocessing on the data
    
    # Example: One-hot encoding for categorical features
    data_encoded = pd.get_dummies(data, columns=['cut', 'color', 'clarity'])
    
    # Get the column names of the encoded data
    feature_names = data_encoded.columns.tolist()
    
    # Convert the encoded data to a numpy array
    data_array = data_encoded.values
    
    return data_array, feature_names

# Define the Streamlit app
def main():
    st.title("Diamond Price Prediction")

    # Add the necessary input fields for diamond characteristics
    carat = st.number_input("Carat")
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = st.number_input("Depth")
    table = st.number_input("Table")

    # Preprocess the input data
    new_data, feature_names = preprocess_input(carat, cut, color, clarity, depth, table)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_data)

    # Display the predictions
    st.subheader("Predicted Price:")
    st.write(predictions[0])

if __name__ == "__main__":
    main()


# In[ ]:




