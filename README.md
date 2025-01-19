
# AI-Powered Applications Suite

Welcome to the AI-Powered Agent Suite! This suite includes several applications designed to leverage AI for various purposes, including stock analysis, health and fitness planning, data visualization, and more. Each application is built using Streamlit for a seamless and interactive user experience.

## Table of Contents

- [Projects Overview](#projects-overview)
- [Setup Instructions](#setup-instructions)
- [Usage Guidelines](#usage-guidelines)
- [Contributing](#contributing)
- [License](#license)

## Projects Overview

### 1. Stock Analyzer AI Agent

- **Description**: An AI-powered tool for analyzing stock market data. It provides real-time stock prices, technical analysis, company fundamentals, and market insights.
- **Features**:
  - Real-time stock prices
  - Technical analysis (SMA, RSI, MACD)
  - Company fundamentals
  - Historical price trends
  - Market insights

### 2. AI Health & Fitness Planner

- **Description**: A personalized health and fitness planner that generates dietary and fitness plans based on user input.
- **Features**:
  - Personalized dietary plans
  - Customized fitness routines
  - Goal-oriented recommendations
  - Q&A section for plan-related queries

### 3. AI Data Visualization Agent

- **Description**: A data visualization tool that uses AI to provide insights and visualizations from uploaded datasets.
- **Features**:
  - Interactive visualizations
  - Deep data insights
  - Natural language queries
  - SQL query generation
  - Statistical analysis

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [YFinance](https://pypi.org/project/yfinance/)
- [TA-Lib](https://pypi.org/project/ta/)
- [Plotly](https://plotly.com/python/)
- [dotenv](https://pypi.org/project/python-dotenv/)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sanskaryo/DataLens-AI-Agent.git
   cd DataLens-AI-Agent
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Copy `.env.example` to `.env` and fill in your API keys.
   - Example:
     ```
     GROQ_API_KEY = "your_groq_api_key"
     TOGETHER_API_KEY = "your_together_api_key"
     E2B_API_KEY = "your_e2b_api_key"
     ```

## Usage Guidelines

### Running the Applications

- **Stock Analyzer AI Agent**:
  ```bash
  streamlit run streamlit_finance_app.py
  ```

- **AI Health & Fitness Planner**:
  ```bash
  streamlit run fitnessAgent.py
  ```

- **AI Data Visualization Agent**:
  ```bash
  streamlit run app.py
  ```

### Interacting with the Applications

- **Stock Analyzer**: Use the sidebar to access features like quick analysis, technical indicators, and company info. Enter queries in the chat input to get insights.
- **Health & Fitness Planner**: Fill in your profile details and generate personalized plans. Use the Q&A section for any questions about your plan.
- **Data Visualization Agent**: Upload a CSV file and use the query input to ask questions about your data. View visualizations and download results.

## Contributing

We welcome contributions to improve our applications! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your fork.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
