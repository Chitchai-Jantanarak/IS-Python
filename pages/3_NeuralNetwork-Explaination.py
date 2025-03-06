import streamlit as st

st.set_page_config(page_title="Neural Networks Explanation", page_icon="🧠")

st.title("Neural Networks Explanation")

st.markdown("""
## Understanding Neural Networks

Neural Networks are a fundamental approach in machine learning inspired by the human brain's neural structure.

### Key Concepts:

1. **Artificial Neurons**:
   - Basic computational units
   - Receive inputs, apply weights, and produce an output
   - Mimic biological neurons' signal transmission

2. **Network Architecture**:
   - Input Layer: Receives initial data
   - Hidden Layers: Process and transform data
   - Output Layer: Produces final prediction or classification

3. **Learning Mechanisms**:
   - Backpropagation
   - Gradient Descent
   - Weight Adjustment

### Types of Neural Networks:
- Feedforward Neural Networks
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Generative Adversarial Networks (GAN)

### Applications:
- Image Recognition
- Natural Language Processing
- Speech Recognition
- Autonomous Driving
- Medical Diagnosis
""")

# Interactive element
st.sidebar.header("Neural Network Quiz")
quiz_question = st.sidebar.selectbox(
    "What is the primary function of a hidden layer?",
    [
        "Select an answer",
        "To receive initial data",
        "To produce final predictions",
        "To process and transform data",
        "To store network weights"
    ]
)

if quiz_question == "To process and transform data":
    st.sidebar.success("Correct! 🎉")
elif quiz_question != "Select an answer":
    st.sidebar.error("Not quite right. Try again!")