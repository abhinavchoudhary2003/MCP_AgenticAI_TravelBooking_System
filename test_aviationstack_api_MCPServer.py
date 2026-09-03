import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
# create a client that will talk with MCP server
client = MultiServerMCPClient(
    {
        "aviationstack": {
            "transport": "stdio", # used to communicate with MCP local server over stdio(standard input output )
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
        }
    }
)


import asyncio
# function to get tools from MCP local Server 
async def main():

    tools = await client.get_tools() # get tool from MCP Server

    print("\nAvailable Tools:\n")

    for tool in tools:
        print(tool.name) # print tools 

if __name__ == "__main__":
    asyncio.run(main())

# (MCP_Venv) PS D:\MCP_AgenticAI_Project> python .\test_aviationstack_api_MCPServer.py

# Available Tools:

# flights_with_airline
# historical_flights_by_date
# flight_arrival_departure_schedule
# future_flights_arrival_departure_schedule
# random_aircraft_type
# random_airplanes_detailed_info
# random_countries_detailed_info
# random_cities_detailed_info
# list_airports
# list_airlines
# list_routes
# list_taxes    