import streamlit as st
import time
import pandas as pd

# Titles and Text
st.title("Hello Streamlit!")
st.write("BIO")

# Input widgets
st.text_input("Enter your first name")
st.text_input("Enter your last name")
st.slider("Enter age")
st.number_input("Enter phone number")
st.text_area("Enter address")
st.checkbox("I agree")

# Dropdowns and Radio Buttons
options=st.selectbox("select an option",["option 1","option 2","option 3"])
st.write("You selected:",options)
st.radio("select your gender",["male","female","other"])

# sidebar
st.sidebar.title("Welcome")
st.sidebar.write("This is sidebar")
st.sidebar.text_input("enter your email")
st.sidebar.selectbox("select your country",["India","USA","UK","Other"])
st.sidebar.slider("select your rating",0,5)

# Image
st.image("image.jpg",caption="my image", width=300)

# Columns
col1, col2 = st.columns(2)
with col1:
    st.write("Column 1")
    col1.text_input("Enter your name in column1")

with col2:
    st.write("Column 2")
    col2.text_input("Enter your name in column2")

# Password + Buttons
st.text_input("Enter your password",type="password")
st.button("submit")

# Links and other inputs
st.number_input("enter your age",min_value=0,max_value=120,step=1)
st.write("[google](https://www.google.com)")
st.date_input("select a date")
st.time_input("select a time")
st.color_picker("select a color")
st.file_uploader("select folder")

# Data + Math
st.write([1,2,3])
st.latex(r"E = mc^2")

# Progress + Effects
progress_bar = st.progress(50)
st.header("progress bar")
time.sleep(5)
st.write("Hello World")
st.write("progress completed")
st.balloons()
st.snow()

# Spinner
st.subheader("Spinner")
with st.spinner("Waiting..."):
    time.sleep(10)
    st.write("Process completed!")

# DataFrame
df = pd.read_csv("test.csv")
st.subheader("DataFrame")
st.dataframe(df.head())

# Display Dataframe as Table
st.subheader("DataFrame as Table")
st.table(df.head(10))

# Sample code
st.subheader("Sample code")
sample_code = '''
def greet(name):
    return "Hello, "+ name + "!"

Print(greet("KGP Talkie"))
'''
st.code(sample_code, language='python')

# Sample metric
st.subheader("Sample metric")
st.metric("Accuracy", value=0.85, delta=+0.05)

# Sample data editor
st.subheader("Data Editor")
edited_data = st.data_editor(df.head())

st.write("Edited DataFrame:")
st.write("edited data")
##session##
if "count" not in st.session_state:
    st.session_state.count=0
if st.button("Increase"):
    st.session_state.count+=1
st.write("Count:", st.session_state.count)