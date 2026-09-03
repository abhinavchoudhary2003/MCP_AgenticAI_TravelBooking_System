
import os
import asyncio  # needed for asyncio.run() used by hotel_agent's MCP call
from typing import TypedDict, Annotated

import operator

import psycopg # 'psycopg' (psycopg3) -> PostgreSQL database driver for Python. Required because we're using Postgres as the backend to store LangGraph checkpoints


from langgraph.graph import StateGraph, START, END


from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
# 'langchain_core' -> Base message classes used to represent conversation turns in a structured way (who said what).
# AnyMessage   -> generic type covering any message class (used in typing).
# HumanMessage -> represents user input.
# AIMessage    -> represents the model's response.
# SystemMessage-> give instruction or behaviour rules to the model 

from langchain_groq import ChatGroq

# from tools.tavily_tool import tavily_search
# here tavily tool API  replace with tavily mcp search 
# from Mcp_Client import tavily_mcp_search 
# from tools.flight_tool import search_flights

# Import MCP helper functions from the Mcp_Client module.

from Mcp_Client import (
    tavily_mcp_search, # Used to perform web searches through the Tavily MCP server.
    get_airports,  # Used to get the list of available airports.
    get_airlines,# Used to get the list of available airlines.
    aviation_mcp_call, # Used to dynamically call a specific aviation MCP tool.
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search
    )

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") # we need db to store our agents converstains

# Add LLM ModeL
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)
#State is the shared information storage of our workflow. Agents can read information from the state and write their results back into it.
# TravelState is the shared memory of our workflow. Every node (agent) can read information from this state  and can also add or update information in the state. LangGraph passes this state from one node to the next.


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]# messages stores all the messages created during the workflow.It can contain messages from the user, AI agents, or other instructions.operator.add means new messages are added to the existing list instead of replacing the old messages.
    user_query: str# This stores the original request given by the user.# Other agents can read this value and use it to perform their tasks.
    flight_results: str# This stores the results produced by the flight agent.# The flight agent writes the results here, and the itinerary agent later reads them.
    hotel_results: str # This stores the results produced by the hotel agent. The hotel agent writes the results here, and the itinerary agent later reads them.
    itinerary: str# This stores the travel itinerary created by the itinerary agent. The final agent reads this information to create the final response.
    llm_calls: int # This keeps track of how many times our LLM is called. Each node increases this number when it performs its work.
    weather_results: str


# # flight agent 
# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)
#     return {
#         "flight_results": flight_data,

#         "messages": [
#             AIMessage(content="Flight results fetched")
#         ],

#         "llm_calls": state.get("llm_calls", 0) + 1
#     }


# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""



# # flight Agent Using MCP local Server 
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Hotel agent  using MCP remote Server 

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
   # hotel_results = tavily_search(query)

   # repace with mcp tavily server 
    hotel_results = asyncio.run( 
        tavily_mcp_search(query)
    )
    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],

        "llm_calls": state.get("llm_calls", 0) + 1
    }
# Weather agent 
def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }


# ── Itinerary agent ─────────────────────────────────────────────────────
# This agent reads the user's query, flight results, and hotel results.

def itinerary_agent(state: TravelState):
    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
  
    Weather Information:
    {state['weather_results']}
    """
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner"),
        HumanMessage(content=prompt)
    ])
    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ── Final response agent ────────────────────────────────────────────────
# This agent reads the itinerary and turns it into the final response
# shown to the user.
#
# NOTE: We intentionally do NOT re-send flight_results/hotel_results here.
# itinerary_agent already read them and folded everything relevant into
# state["itinerary"], so passing the raw results again just duplicates
# tokens in the prompt without adding new information — and was pushing
# this call over Groq's per-minute token limit (8,000 TPM on some models).



# ── Build the graph ─
graph = StateGraph(TravelState)


# Add each agent/function to the graph.
# The name on the left is the name we give to the node.
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)




graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)

# _conn = psycopg.connect(DATABASE_URL)
_conn = psycopg.connect(DATABASE_URL, autocommit=True)


checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)

# This block runs only when we directly run this Python file, not when this file is imported into another file. this is just for testing the agents 
if __name__ == "__main__":
    # config = {
    #     "configurable": {
    #         "thread_id": "user_aarohi"
    #     }
    # }

    # every run starts fresh.
    import uuid
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    user_input = input("Enter travel request: ")
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)