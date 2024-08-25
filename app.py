import streamlit as st
from PIL import Image
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import os

# Setting up the page layout
st.set_page_config(page_title="AI Product Design & Development", layout="wide")

# Sidebar - User inputs for Product Specifications
st.sidebar.title("Product Specifications")
product_name = st.sidebar.text_input("Product Name", "Example Product")
material = st.sidebar.selectbox("Material", ["Plastic", "Metal", "Wood", "Composite"])
dimensions = st.sidebar.text_input("Dimensions (L x W x H in cm)", "10 x 5 x 3")
constraints = st.sidebar.text_area("Design Constraints", "E.g., Must be lightweight, eco-friendly")
budget = st.sidebar.number_input("Budget ($)", min_value=0, value=1000)

st.sidebar.subheader("Project Info")
st.sidebar.text("AI-Powered Product Design")

# Main app title and description
st.title("AI Product Design & Development Tool")
st.markdown("""
Welcome to the AI-powered product design and development tool. This app leverages generative AI to accelerate the design process, optimize products for manufacturing, and simulate product performance.
""")

# Tabs for different sections of the app
tabs = st.tabs(["Design Generation", "Simulation", "Optimization"])

# IBM WatsonX API Setup
project_id = os.getenv('WATSONX_PROJECT_ID')
api_key = os.getenv('WATSONX_API_KEY')

if api_key and project_id:
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)
    client = APIClient(credentials)
    client.set.default_project(project_id)

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.MAX_NEW_TOKENS: 200,
        GenParams.STOP_SEQUENCES: ["\n"]
    }

    model_id = ModelTypes.GRANITE_13B_CHAT_V2
    model = ModelInference(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)

    # Design Generation Tab
    with tabs[0]:
        st.header("Generate Product Designs")
        st.write("Input your product specifications in the sidebar and click below to generate design concepts.")
        
        if st.button("Generate Design Concepts"):
            prompt = f"""You are an AI specialized in product design. Generate creative product design concepts based on the following details:\n
            Product Name: {product_name}\n
            Material: {material}\n
            Dimensions: {dimensions}\n
            Constraints: {constraints}\n
            Budget: {budget} USD\n
            Provide detailed design concepts and explain their features."""
            
            try:
                response = model.generate_text(prompt=prompt, params=parameters)
                st.success("Generated Design Concepts:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Simulation and Optimization tabs will be expanded in future steps.
