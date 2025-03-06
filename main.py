import streamlit as st

def main():
    """
    Main Streamlit application entry point
    Configures the overall application layout
    """
    # Set page configuration for the entire application
    st.set_page_config(
        page_title="Intelligent System Project", 
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main application header
    st.title("Intelligent System Project")
    st.write("Exploring Machine Learning and Neural Networks")

    # Main page content
    st.markdown("""
    ## Welcome to the Intelligent System Project

    This interactive dashboard provides insights into:
    - Machine Learning Concepts
    - Neural Network Architectures
    - Model Demonstrations
    """)

    # Feature highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Machine Learning")
        st.write("""
        - Supervised and Unsupervised Learning
        - Classification and Regression Models
        - Interactive Demonstrations
        """)
    
    with col2:
        st.subheader("Neural Networks")
        st.write("""
        - Deep Learning Principles
        - Network Architectures
        - Training Visualizations
        """)

def run():
    """
    Wrapper function to run the Streamlit application
    """
    main()

if __name__ == "__main__":
    run()