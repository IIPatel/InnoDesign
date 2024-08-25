import streamlit as st

from ibm_watsonx_ai import APIClient

from ibm_watsonx_ai import Credentials

from ibm_watsonx_ai.foundation_models import ModelInference

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import os



# Set up page configuration

st.set_page_config(page_title="AI Product Design & Development", layout="wide")



# Initialize session state to keep track of queries

if 'query_count' not in st.session_state:

    st.session_state.query_count = 0

if 'generated_response' not in st.session_state:

    st.session_state.generated_response = None



# Limit the number of queries per session

MAX_QUERIES = 5



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



    model_id = ModelTypes.GRANITE_13B_CHAT_V2  # Initial model, to be evaluated further

    model = ModelInference(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)



    # Design Generation Tab

    with tabs[0]:

        st.header("Generate Product Designs")

        st.write("Input your product specifications in the sidebar and click below to generate design concepts.")



        if st.session_state.query_count < MAX_QUERIES:

            if st.button("Generate Design Concepts"):

                prompt = f"""You are an AI specialized in product design. Generate creative product design concepts based on the following details:\n

                Product Name: {product_name}\n

                Material: {material}\n

                Dimensions: {dimensions}\n

                Constraints: {constraints}\n

                Budget: {budget} USD\n

                Provide detailed design concepts, explaining how they meet the constraints and budget. Also, suggest alternatives if the current design exceeds the budget or constraints."""



                try:

                    with st.spinner("Generating design concepts..."):

                        response = model.generate_text(prompt=prompt, params=parameters)

                        st.session_state.generated_response = response

                        st.session_state.query_count += 1

                        st.success("Generated Design Concepts:")

                        st.write(response)

                except Exception as e:

                    st.error(f"An error occurred: {e}")

        else:

            st.warning(f"You have reached the query limit of {MAX_QUERIES}. Please restart the session to continue.")



        # Display the previous generated response and allow for follow-up queries

        if st.session_state.generated_response:

            st.subheader("Refine Your Design")

            if st.session_state.query_count < MAX_QUERIES:

                if st.button("Ask for a cheaper variant"):

                    follow_up_prompt = prompt + "\nPlease suggest a cheaper variant."

                    try:

                        follow_up_response = model.generate_text(prompt=follow_up_prompt, params=parameters)

                        st.session_state.query_count += 1

                        st.info("Cheaper Variant:")

                        st.write(follow_up_response)

                    except Exception as e:

                        st.error(f"An error occurred: {e}")



                if st.button("Explore alternative materials"):

                    follow_up_prompt = prompt + "\nPlease explore alternative materials that might better fit the design constraints."

                    try:

                        follow_up_response = model.generate_text(prompt=follow_up_prompt, params=parameters)

                        st.session_state.query_count += 1

                        st.info("Alternative Materials:")

                        st.write(follow_up_response)

                    except Exception as e:

                        st.error(f"An error occurred: {e}")

            else:

                st.warning("You have reached the query limit.")



# Simulation and Optimization tabs will be expanded in future steps.