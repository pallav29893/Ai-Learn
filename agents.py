from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

import os
from dotenv import load_dotenv
# import logging
# logging.basicConfig(level=logging.INFO)
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model="liquid/lfm-2.5-1.2b-thinking:free",
    # model="meta-llama/llama-3.1-8b-instruct",
    temperature=0,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

@tool
def get_weather(city: str) -> str:
    """Get the weather at the location"""
    return f"The weather at {city}"



agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a weather agent. You are given a city name and you need to return the weather at that location.",
)
response = model.invoke("What is the today's weather in Delhi?")
# response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Delhi?"}]})
# logging.info(response)
print(response)