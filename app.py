
import streamlit as st

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)

from langchain_community.callbacks import StreamlitCallbackHandler
import wikipedia

wikipedia.set_user_agent(
    "LangChainSearchEngine/1.0 (educational project)"
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LangChain Search Engine",
    page_icon="🔎",
)

st.title("🔎 LangChain - Chat with Search")

st.write(
    "Ask questions using web search, Wikipedia, or Arxiv."
)


# --------------------------------------------------
# Initialize tools
# --------------------------------------------------

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)

arxiv = ArxivQueryRun(
    api_wrapper=arxiv_wrapper
)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)

wiki = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)

search = DuckDuckGoSearchRun(
    name="search"
)

tools = [search, arxiv, wiki]


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a chatbot that can search "
                "the web, Wikipedia, and Arxiv. "
                "How can I help you?"
            ),
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(
        msg["content"]
    )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

if prompt := st.chat_input(
    placeholder="What is machine learning?"
):

    if not api_key:
        st.warning("Please enter your Groq API key.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.chat_message("user").write(prompt)

    try:
        # Initialize Groq model
        llm = ChatGroq(
            groq_api_key=api_key,
            model="openai/gpt-oss-120b",
            temperature=0,
        )

        # Create LangChain 1.x agent
        search_agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "You are a helpful search assistant. "
                "Use the available tools when needed. "
                "Answer clearly and accurately."
            ),
        )

        # Invoke agent with the current conversation
        with st.chat_message("assistant"):

            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False,
            )

            response = search_agent.invoke(
                {
                    "messages": [
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                        }
                        for msg in st.session_state.messages
                    ]
                },
                config={
                    "callbacks": [st_cb]
                },
            )

            answer = response["messages"][-1].content

            st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")