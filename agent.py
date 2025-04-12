import os
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Set API keys directly
os.environ["PHI_API_KEY"] = 'phi-mjsL62o9sFCR2acbECb8bycMsxufltD3jArICCe4rh0'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyC3GCuLQrQwkPN5QM-fEHtqou5so_EFVro'
# Function to initialize agents
@st.cache_resource
def create_agents():
    # Web search agent
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for the information",
        model=Gemini(model="pro"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tools_calls=True,
        markdown=True,
    )

    # Financial agent
    finance_agent = Agent(
        name="Finance AI Agent",
        model=Gemini(model="pro"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            ),
        ],
        instructions=["Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )

    # Combined multi-agent
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        model=Gemini(model="pro"),
        instructions=["Always include sources", "Use table to display the data"],
        show_tools_calls=True,
        markdown=True,
    )
    return multi_ai_agent

# Streamlit app
st.set_page_config(page_title="Finance Agent", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>🤖 Finance Agent </h1>
""", unsafe_allow_html=True)

# Initialize the multi-agent (cached)
multi_ai_agent = create_agents()

# User input for query
st.subheader("Enter your query below")
query = st.text_area("Query", value="", height=70,placeholder="e.g. Summarize analyst recommendations and share the latest news for NVDA")

if st.button("Run Query") and query.strip():
    st.info("Processing your request...")
    try:
        # Get the response object
        response = multi_ai_agent.run(query)

        # If response is dict or has a known structure, format it
        if hasattr(response, 'content'):
            st.markdown(response.content)
        elif isinstance(response, str):
            st.markdown(response)
        else:
            st.write(response)
    except Exception as e:
        st.error(f"Error running agents: {e}")
