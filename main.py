import streamlit as st
import langchain_helper as lh

st.title("Restaurant Name Generator")
cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "American", "Mexican", "Italian", "Chinese", "Japanese"))

if cuisine:
    response = lh.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'])
    menu_items = response['menu_items'].split(',')

    st.write("**Menu Items**")
    for item in menu_items:
        st.write('-', item.strip())  # Use strip() to remove extra spaces