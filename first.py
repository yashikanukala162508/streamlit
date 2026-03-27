import streamlit as st
with st.form("my_form"):
    col1, col2 = st.columns(2)
    name1 = col1.text_input("First Name")
    name2 = col2.text_input("Last Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    address = st.text_area("Enter the address")
    submit = st.form_submit_button("Submit")
    if submit:
        if password != confirm_password:
            st.error("Passwords do not match!")
        else:
            st.success("Form submitted successfully!")
            st.write("Name:", name1, name2)
            st.write("Email:", email)
            st.write("Address:", address)