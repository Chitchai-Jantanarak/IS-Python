import streamlit as st

st.set_page_config(page_title="Machine Learning Explanation", page_icon="📚")

st.title("Machine Learning Explanation")

st.markdown("""
## Introduction to Machine Learning

Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.

### Key Concepts:

1. **Supervised Learning**: 
   - Algorithms are trained on labeled data
   - Goal is to learn a function that maps input to output
   - Examples: Classification, Regression

2. **Unsupervised Learning**:
   - Algorithms work on unlabeled data
   - Discover hidden patterns or groupings
   - Examples: Clustering, Dimensionality Reduction

3. **Types of Machine Learning Algorithms**:
   - Decision Trees
   - Random Forests
   - Support Vector Machines
   - Neural Networks
   - K-Means Clustering

### Applications:
- Predictive Analytics
- Image and Speech Recognition
- Recommendation Systems
- Fraud Detection
- Autonomous Vehicles
""")

# Optional: Add an interactive element
st.sidebar.header("Quick ML Quiz")
quiz_question = st.sidebar.selectbox(
    "What is supervised learning?",
    [
        "Select an answer",
        "Learning from unlabeled data",
        "Learning from labeled data with known outcomes",
        "Learning without any data",
        "Learning only from images"
    ]
)

if quiz_question == "Learning from labeled data with known outcomes":
    st.sidebar.success("Correct! 🎉")
elif quiz_question != "Select an answer":
    st.sidebar.error("Not quite right. Try again!")