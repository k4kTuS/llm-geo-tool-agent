Follow these steps to deploy the Streamlit app locally:

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. **Clone the Repository**

    ```bash
    git clone git@github.com:k4kTuS/llm-geo-tool-agent.git
    cd llm-geo-tool-agent
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set API Keys**

    Open the `config.ini` file and set the correct API keys for your application. Currently, the implementation is using OpenAI api, but it can be changed for any other LLM provider that supports tool calling (see `agent.py`).

    ```ini
    [OPENAI]
    api_key=YOUR_API_KEY
    model_id=gpt-4o-mini
    ```

5. **Run the App**

    ```bash
    streamlit run app.py
    ```

6. **Access the App**

    Open your web browser and go to `http://localhost:8501`. (You should be redirected automatically)

### Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)