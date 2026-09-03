# pip install langchain-mcp-adapters #  need this package to install MCP client in the project

import os
import asyncio  # Keeps Python working on other tasks instead of sitting idle waiting for network responses.
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Create an MCP client to connect with one or more MCP servers.
client =MultiServerMCPClient(
     { 
         # Configure connection to the remote Tavily MCP server
        "tavily": {
             "transport": "streamable_http",   # MCP client Use HTTP streaming to communicate with the remote MCP server         
             "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
        # Configure connection to the local Aviationstack MCP server
        "aviationstack": {
                    "transport": "stdio", # MCP CLient use standard input/output (stdio) to  Communicate with the local MCP Server
                    # all this for to run MCP local Server
                    "command": r"D:\MCP_AgenticAI_Project\aviationstack-mcp\.venv\Scripts\python.exe",
                    "args": [
                        "-m",
                        "aviationstack_mcp",
                        "mcp",
                        "run"
                    ],
                    "env": {
                        "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
                    }
                },
                # Configure connection to the local Custom Weather MCP server
                "weather": {
                "transport": "stdio",
                "command": r"D:\MCP_AgenticAI_Project\MCP_Venv\Scripts\python.exe",
                "args": [
                        r"D:\MCP_AgenticAI_Project\custom_weather_mcp_server.py"
                        ],
                        "env": {
                             "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
                            }
                        }

     }  



)



# this method will do the same work first MCP client  get all tools  from MCP server and then search that specific toolbut here we do not give query .
# This method is responsible for connecting to the MCP Server throught the MCP client  and discovery the tools exposed by that MCP Server.
# SO even if user perform hunderds of searches , the app does not repeatedly ask the MCP server for the list of tools
# Insted , it reuses the cached tavily_search tool that is already stored in memory
search_tool = None
aviation_tools = {}


async def initialize_mcp():

    global search_tool
    global aviation_tools

    # Skip initialization if MCP tools are already loaded.
    if search_tool is not None and aviation_tools:
        return

    # Get all available tools from the configured MCP servers.
    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    # Display the names of all tools provided by the MCP servers.
    for tool in tools:
        print(tool.name)

    # Find and store the Tavily search tool for general web searches.
    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    # Store all non-Tavily tools separately for aviation-related operations.
    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name != "tavily_search"
    }


async def tavily_mcp_search(query: str):

    # Make sure the MCP tools are initialized before using Tavily.
    await initialize_mcp()

    # Send the user's search query to the Tavily MCP tool.
    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )

    # Return Tavily's search result to the calling agent.
    return result


async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):

    # Get the latest list of tools from the connected MCP servers.
    tools = await client.get_tools()

    # Find the aviation tool requested by its name.
    tool = next(
        t for t in tools
        if t.name == tool_name
    )

    # Execute the selected aviation MCP tool with the given arguments.
    result = await tool.ainvoke(
        tool_args or {}
    )

    # Return the aviation tool result.
    return result


async def get_airports():

    # Make sure MCP tools are initialized before accessing aviation tools.
    await initialize_mcp()

    # Get the airport-listing tool from the aviation tools dictionary.
    tool = aviation_tools.get("list_airports")

    # Handle the case where the airport tool is unavailable.
    if not tool:
        return "Airport tool unavailable"

    # Execute the airport tool without additional arguments.
    result = await tool.ainvoke({})

    # Return the list of airports.
    return result


async def get_airlines():

    # Make sure MCP tools are initialized before accessing aviation tools.
    await initialize_mcp()

    # Get the airline-listing tool from the aviation tools dictionary.
    tool = aviation_tools.get("list_airlines")

    # Handle the case where the airline tool is unavailable.
    if not tool:
        return "Airline tool unavailable"

    # Execute the airline-listing tool without additional arguments.
    result = await tool.ainvoke({})

    # Return the list of airlines.
    return result

weather_tool = None
forecast_tool = None

# this function is to get custom tools of weather MCP server from custom_weather_mcp_server.py

async def initialize_weather_tools():

    global weather_tool, forecast_tool

    if weather_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(
        t for t in tools
        if t.name == "get_current_weather"
    )

    forecast_tool = next(
        t for t in tools
        if t.name == "get_forecast"
    )


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city
        }
    )


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city
        }
    )




from langchain_groq import ChatGroq

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

###################################
# Destination Extractor
###################################

def extract_destination(query: str):

    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


if __name__ == "__main__":
    asyncio.run(main())




# this method is for  tools discovery means  MCP Client asks MCP server which tool do you have 
# async def main():
#     tools = await client.get_tools() # use .get_tools()
#     print("\nAvailable MCP Tools:\n")
#     for tool in tools:
#         print(tool.name)

# if __name__ == "__main__":
#     asyncio.run(main())        

# this  method is  to run specific tool i.e. "tavily_search", first it get all tools  from MCP server and then search that specific tool and then ask a query from that specific tool 
# async def main():
#     tools = await client.get_tools() # got all tools from MCP server
#     search_tool = next(
#         tool
#         for tool in tools
#         if tool.name == "tavily_search" # find a specific tool 
#     )
#     result = await search_tool.ainvoke(
#         {
#             "query": "Best hotels in Delhi" # ask a query from that specific tool 
#         }
#     )
#     print(result)
# asyncio.run(main()) 

#search_tool = None
# this method will do the same work first MCP client  get all tools  from MCP server and then search that specific toolbut here we do not give query .
# This method is responsible for connecting to the MCP Server throught the MCP client  and discovery the tools exposed by that MCP Server.
# SO even if user perform hunderds of searches , the app does not repeatedly ask the MCP server for the list of tools
# Insted , it reuses the cached tavily_search tool that is already stored in memory
# Here we just do for travily_search
# async def initialize_mcp():
#     global search_tool
#     if search_tool is not None:
#         return

#     tools = await client.get_tools() # mcp client fetch tools from MCP Server 
#     print("\nAvailable MCP Tools:")

#     for tool in tools:
#         print(tool.name)

#     search_tool = next(
#         tool
#         for tool in tools
#         if tool.name == "tavily_search"
#     )

# async def tavily_mcp_search(query: str):
#     await initialize_mcp()
#     result = await search_tool.ainvoke(
#         {
#             "query": query
#         }
#     )
#     return result   