import numpy as np
import streamlit as st


# Utility functions from utils.py
from utils import load_mhd_file, process_coordinates_and_file


def main():
    st.set_page_config(page_title="Lung Module Classification", page_icon="🫁", layout="wide")
    # Display the logo
    # Center and enlarge the logo using columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.write("")
    with col2:
        st.image("logo.PNG", width=500)
    with col3:
        st.write("")

    st.markdown('<h1 style="color: #1212a1; ">Upload MHD File</h1>', unsafe_allow_html=True)

    # Upload MHD file
    uploaded_file = st.file_uploader("Upload an MHD file", type=["mhd"])

    if uploaded_file:
        st.session_state['mhd_file'] = uploaded_file
        st.success("File uploaded successfully")

    # Input coordinates

    st.sidebar.markdown("<h2><span style='font-size: 24px;'>📍 Nodule Coordinates :</span></h2>",unsafe_allow_html=True)
    x = st.sidebar.number_input("X Coordinate", min_value=0, max_value=1024, value=0, step=1)
    y = st.sidebar.number_input("Y Coordinate", min_value=0, max_value=1024, value=0, step=1)
    z = st.sidebar.number_input("Z Coordinate", min_value=0, max_value=1024, value=0, step=1)

    st.sidebar.markdown("<h2><span style='font-size: 24px;'>💡 Launch ML Models :</span></h2>",unsafe_allow_html=True)
    if st.sidebar.button("Process"):
        if 'mhd_file' not in st.session_state:
            st.error("Please upload an MHD file first")
        else:
            # Process the input and file
            scores = process_coordinates_and_file(st.session_state['mhd_file'], x, y, z)

            # Display the results
            st.subheader("Scores")
            for i, (accuracy, precision) in enumerate(scores):
                st.write(f"Model {i + 1}: Accuracy = {accuracy:.2f}, Precision = {precision:.2f}")


if __name__ == "__main__":
    main()