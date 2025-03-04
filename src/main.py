# Streamlit Example Repository
# This repository contains multiple example Streamlit applications demonstrating various features

# Project Structure:
# streamlit_examples/
# ├── requirements.txt
# ├── README.md
# ├── app1_basic_input.py
# ├── app2_data_visualization.py
# ├── app3_machine_learning.py
# └── app4_interactive_dashboard.py

# requirements.txt
"""
streamlit==1.32.0
pandas==2.2.1
numpy==1.26.4
plotly==5.18.0
scikit-learn==1.4.1
matplotlib==3.8.3
"""

# README.md content
"""
# Streamlit Example Applications

This repository contains a collection of Streamlit applications demonstrating various features and capabilities.

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run any application:
   ```
   streamlit run app1_basic_input.py
   ```

## Applications

- `app1_basic_input.py`: Basic user input and interaction
- `app2_data_visualization.py`: Data visualization with Plotly and Matplotlib
- `app3_machine_learning.py`: Simple machine learning model deployment
- `app4_interactive_dashboard.py`: Interactive data dashboard
"""

# app1_basic_input.py
import streamlit as st


def app1_basic_input():
    st.title("Basic Input and Interaction")
    
    # Text input
    name = st.text_input("Enter your name", "")
    
    # Slider
    age = st.slider("Select your age", 0, 100, 25)
    
    # Checkbox
    is_student = st.checkbox("Are you a student?")
    
    # Multiselect
    interests = st.multiselect(
        "Select your interests", 
        ["Technology", "Sports", "Music", "Travel", "Reading"]
    )
    
    # Button
    if st.button("Submit"):
        st.write(f"Hello, {name}!")
        st.write(f"You are {age} years old.")
        st.write(f"Student status: {is_student}")
        st.write(f"Your interests: {', '.join(interests)}")

# app2_data_visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

def app2_data_visualization():
    st.title("Data Visualization with Plotly")
    
    # Generate sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': np.random.randint(10, 100, 5),
        'Secondary': np.random.randint(5, 50, 5)
    })
    
    # Bar chart
    st.subheader("Bar Chart")
    fig_bar = px.bar(df, x='Category', y='Value', 
                     title='Sample Bar Chart')
    st.plotly_chart(fig_bar)
    
    # Scatter plot
    st.subheader("Scatter Plot")
    fig_scatter = px.scatter(df, x='Value', y='Secondary', 
                              color='Category', 
                              title='Value vs Secondary')
    st.plotly_chart(fig_scatter)

# app3_machine_learning.py
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def app3_machine_learning():
    st.title("Iris Flower Classification")
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    # User inputs
    st.subheader("Predict Iris Flower Species")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
    
    # Prediction
    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = clf.predict(input_data)
        species = iris.target_names[prediction[0]]
        st.success(f"Predicted Species: {species}")
        
        # Show model accuracy
        accuracy = clf.score(X_test, y_test)
        st.info(f"Model Accuracy: {accuracy * 100:.2f}%")

# app4_interactive_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

def app4_interactive_dashboard():
    st.title("Interactive Sales Dashboard")
    
    # Generate sample sales data
    np.random.seed(42)
    data = {
        'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'Sales': np.random.randint(100, 1000, 365),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 365)
    }
    df = pd.DataFrame(data)
    
    # Sidebar for filtering
    st.sidebar.header("Filters")
    selected_category = st.sidebar.multiselect(
        "Select Category", 
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )
    
    # Filter data
    filtered_df = df[df['Category'].isin(selected_category)]
    
    # Total sales
    total_sales = filtered_df['Sales'].sum()
    st.metric(label="Total Sales", value=f"${total_sales:,}")
    
    # Line chart of sales
    fig_sales = px.line(
        filtered_df.groupby(['Date', 'Category'])['Sales'].sum().reset_index(), 
        x='Date', y='Sales', color='Category',
        title='Daily Sales by Category'
    )
    st.plotly_chart(fig_sales)
    
    # Bar chart of category sales
    category_sales = filtered_df.groupby('Category')['Sales'].sum()
    fig_category = px.bar(
        x=category_sales.index, 
        y=category_sales.values,
        title='Total Sales by Category'
    )
    st.plotly_chart(fig_category)

# Main entry point
def main():
    st.sidebar.title("Streamlit Examples")
    app_selection = st.sidebar.selectbox(
        "Choose an Application",
        [
            "Basic Input",
            "Data Visualization", 
            "Machine Learning", 
            "Interactive Dashboard"
        ]
    )
    
    if app_selection == "Basic Input":
        app1_basic_input()
    elif app_selection == "Data Visualization":
        app2_data_visualization()
    elif app_selection == "Machine Learning":
        app3_machine_learning()
    elif app_selection == "Interactive Dashboard":
        app4_interactive_dashboard()

if __name__ == "__main__":
    main()