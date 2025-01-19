import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


load_dotenv()


st.set_page_config(
    page_title="📊 AI Data Visualization Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0fff4;
        border: 1px solid #9ae6b4;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fffaf0;
        border: 1px solid #fbd38d;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    """Execute Python code in the E2B sandbox and return results."""
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results

def match_code_blocks(llm_response: str) -> str:
    """Extract Python code blocks from LLM response."""
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    """Interact with the LLM to analyze the dataset and return results."""
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning("Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    """Upload the dataset to the E2B sandbox and return the path."""
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error

def main():
    """Main Streamlit application."""
    st.markdown("""
        <div style='background-color: #e0f7fa; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h1 style='color: #00796b; text-align: center;'>🚀 Ultimate Data Analysis Agent</h1>
        <p style='color: #004d40; text-align: center;'>Powered by AI for comprehensive data insights!</p>
        </div>
    """, unsafe_allow_html=True)

   
    st.session_state.together_api_key = os.getenv('TOGETHER_API_KEY')
    st.session_state.e2b_api_key = os.getenv('E2B_API_KEY')
    
    if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
        st.error("Please ensure TOGETHER_API_KEY and E2B_API_KEY are set in your .env file")
        return

    with st.sidebar:
        st.header("⚙️ Settings & Info")
        
        # Dark Mode Toggle
        dark_mode = st.checkbox("🌙 Dark Mode", value=False)
        if dark_mode:
            st.markdown("""
                <style>
                    body { background-color: #1E1E1E; color: #FFFFFF; }
                    .stApp { background-color: #1E1E1E; }
                    .plot-container { background-color: #2D2D2D; }
                    .streamlit-expanderHeader { color: #FFFFFF; }
                    .css-1d391kg { background-color: #2D2D2D; }
                </style>
            """, unsafe_allow_html=True)
        
        # Usage Steps with better formatting
        st.subheader("📝 Quick Start Guide")
        st.markdown("""
    <div style='background-color: rgba(255, 255, 255, 0.1); padding: 2rem; border-radius: 0.5rem; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
        <h2 style='text-align: center; color: #333;'>📊 Data Analysis Workflow</h2>
        <ol style='font-size: 1.1rem; line-height: 1.6;'>
            <li>📂 <strong>Upload your CSV dataset</strong></li>
            <li>👀 <strong>Preview and understand your data</strong></li>
            <li>✨ <strong>Choose analysis options:</strong>
                <ul>
                    <li>📝 Custom Questions</li>
                </ul>
            </li>
            <li>🎯 <strong>Get AI-powered insights</strong></li>
            <li>💾 <strong>Download results & code</strong></li>
        </ol>
    </div>
""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced About section
        st.subheader("🎯 Features")
        st.markdown("""
            - 📊 Interactive Visualizations
            - 🔍 Deep Data Insights
            - 💡 Natural Language Queries
            - 🎯 SQL Query Generation
            - 📈 Statistical Analysis
            - 📱 Mobile-Friendly UI
            - 🌙 Dark/Light Mode
            - 💾 Export Options
        """)
        
        # Model selection with better UI
        st.markdown("---")
        st.subheader("🤖 AI Model Configuration")
        model_options = {
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0 
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    # Example queries with categories
    with st.expander("💡 Example Queries & Templates", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Analysis", "🔍 SQL"])
        
        with tab1:
            st.markdown("""
                **Visualization Queries:**
                - 📊 Show distribution of ratings across genres
                - 📈 Create a scatter plot of budget vs revenue
                - 📉 Plot the trend of ratings over years
                - 🎨 Generate a bar chart showing the number of movies per genre
                - 📊 Show me the distribution of movie genres in the dataset.
            """)
        
        with tab2:
            st.markdown("""
                **Analysis Queries:**
                - 📊Create a scatter plot of ratings vs. number of votes?
                - 📈 Find outliers in the budget distribution
                - 🔍 Analyze the relationship between rating and revenue
                - 📉 Calculate summary statistics for numeric columns
                - 🎯 Identify trends in movie releases by year
            """)
        
        with tab3:
            st.markdown("""
                **SQL-Style Queries:**
                ```sql
                -- Average rating by genre
                SELECT genre, AVG(rating) FROM movies GROUP BY genre

                -- Top grossing movies
                SELECT title, revenue FROM movies ORDER BY revenue DESC LIMIT 10

                -- Movies with high ratings
                SELECT title, rating FROM movies WHERE rating > 8.0

                -- Genre distribution
                SELECT genre, COUNT(*) FROM movies GROUP BY genre
                ```
            """)

    uploaded_file = st.file_uploader(
        "📂 Choose a CSV file",
        type="csv",
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        col1, col2 = st.columns([1, 2])
        with col1:
            show_full = st.checkbox("Show full dataset", help="Toggle to view the complete dataset")
        with col2:
            st.download_button(
                "📥 Download Dataset",
                data=df.to_csv(index=False),
                file_name="dataset.csv",
                mime="text/csv"
            )
        
        if show_full:
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.dataframe(df.head(), use_container_width=True)
        
        # Query Options
        st.markdown("### 🎯 Analysis Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            generate_sql = st.checkbox("🔍 Generate SQL", help="Generate SQL queries for your question")
        with col2:
            include_stats = st.checkbox("📊 Include Statistics", help="Include statistical analysis")
        with col3:
            interactive_plots = st.checkbox("📈 Interactive Plots", help="Make plots interactive", value=True)
        
        # Query input with better UI
        st.markdown("### 🤔 Ask Your Question")
        query = st.text_area(
            "What would you like to know?",
            placeholder="Example: Show me the distribution of ratings and their correlation with revenue",
            help="Ask any question about your data"
        )
        
        if st.button("🔍 Analyze", use_container_width=True):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("Please ensure both API keys are set in the .env file.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Enhance the query based on options
                    enhanced_query = query
                    if generate_sql:
                        enhanced_query += "\nAlso generate equivalent SQL query."
                    if include_stats:
                        enhanced_query += "\nInclude relevant statistical analysis."
                    
                    # Get analysis and results
                    with st.spinner('🤖 AI Agent is analyzing your data...'):
                        code_results, llm_response = chat_with_llm(code_interpreter, enhanced_query, dataset_path)
                    
                    # Results section
                    st.markdown("### 🎯 Analysis Results")
                    
                    # Display AI response
                    with st.expander("📝 AI Analysis", expanded=True):
                        st.markdown(llm_response)
                    
                    # Code section with download option
                    with st.expander("💻 Python Code", expanded=False):
                        code = match_code_blocks(llm_response)
                        st.code(code, language='python')
                        st.download_button(
                            "📥 Download Code",
                            data=code,
                            file_name="analysis.py",
                            mime="text/plain"
                        )
                    
                    # Visualizations with enhanced interactivity
                    if code_results:
                        st.markdown("### 📊 Visualizations")
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:
                                png_data = base64.b64decode(result.png)
                                image = Image.open(BytesIO(png_data))
                                st.image(image,  use_container_width = True)
                                
                                # Add download button for the image
                                buf = BytesIO()
                                image.save(buf, format='PNG')
                                st.download_button(
                                    "📥 Download Plot",
                                    data=buf.getvalue(),
                                    file_name="visualization.png",
                                    mime="image/png"
                                )
                                
                            elif hasattr(result, 'figure'):
                                fig = result.figure
                                if interactive_plots:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.pyplot(fig)
                                
                            elif hasattr(result, 'show'):
                                st.plotly_chart(result, use_container_width=True, config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToAdd': [
                                        'drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape',
                                        'lasso2d',
                                        'select2d',
                                        'zoom2d',
                                        'pan2d',
                                        'resetScale2d',
                                        'hoverClosestCartesian',
                                        'toggleSpikelines'
                                    ],
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'visualization',
                                        'height': None,
                                        'width': None,
                                        'scale': 2
                                    }
                                })
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                                # Add download button for dataframes
                                st.download_button(
                                    "📥 Download Data",
                                    data=result.to_csv(index=False),
                                    file_name="analysis_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.write(result)

    # Footer with version info
    st.markdown("""
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa;'>
            <p>Made with ❤️ by Sanskar | Version 2.0 | 
            <a href='https://github.com/sanskaryo' target='_blank'>
                <img src='https://img.shields.io/github/stars/sanskaryo/data-analyst-ai-agent?style=social' alt='GitHub stars'/>
            </a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()