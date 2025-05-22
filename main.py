from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, RunConfig
import chainlit as cl
import os
import requests
from dotenv import load_dotenv
load_dotenv()
w_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
if not w_api_key:
    raise ValueError("WEATHER_API_KEY is not set. Please ensure it is defined in your .env file.")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
weatherapi_key = os.getenv("WEATHERAPI_KEY")
if not weatherapi_key:
    raise ValueError("WEATHERAPI_KEY is not set. Please ensure it is defined in your .env file.")
tomorrow_key = os.getenv("TOMORROW_KEY")
if not tomorrow_key:
    raise ValueError("TOMORROW_KEY is not set. Please ensure it is defined in your .env file.")
visualcrossing_key = os.getenv("VISUALCROSSING_KEY")
if not visualcrossing_key:
    raise ValueError("VISUALCROSSING_KEY is not set. Please ensure it is defined in your .env file.")
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(
    model=model,
    model_provider=external_client,
)

@function_tool

def get_weatherapi_forecast(city: str, days: int = 3) -> dict:
    url = f"http://api.weatherapi.com/v1/forecast.json?key={weatherapi_key}&q={city}&days={days}&aqi=no&alerts=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        forecast = [
            {
                "date": day["date"],
                "condition": day["day"]["condition"]["text"],
                "avg_temp": day["day"]["avgtemp_c"],
                "humidity": day["day"]["avghumidity"],
                "wind_kph": day["day"]["maxwind_kph"]
            }
            for day in data.get("forecast", {}).get("forecastday", [])
        ]
        return {
            "source": "WeatherAPI",
            "location": data.get("location", {}).get("name", city),
            "forecast": forecast
        }
    else:
        return {"error": f"WeatherAPI error: {response.status_code}"}

# Tool 2: Tomorrow.io
@function_tool
def get_tomorrow_forecast(city: str, hours: int = 24) -> dict:
    url = (
        f"https://api.tomorrow.io/v4/weather/forecast"
        f"?location={city}"
        f"&apikey={tomorrow_key}"
        f"&timesteps=1h"
        f"&units=metric"
        f"&limit={hours}"
    )
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Get hourly timeline data
        timelines = data.get("timelines", {}).get("hourly", [])
        
        forecast = [
            {
                "time": entry["time"],
                "temperature": entry["values"].get("temperature"),
                "wind_speed": entry["values"].get("windSpeed"),
                "humidity": entry["values"].get("humidity"),
                "precipitation": entry["values"].get("precipitationProbability"),
            }
            for entry in timelines
        ]
        
        return {"source": "Tomorrow.io", "location": city, "forecast": forecast}
    else:
        return {"error": f"Tomorrow.io error: {response.status_code}"}

# Tool 3: Visual Crossing
@function_tool
def get_visualcrossing_weather(
    city: str,
    start_date: str = None,  # format 'YYYY-MM-DD' or None for today
    end_date: str = None,    # format 'YYYY-MM-DD' or None for today
    include_hours: bool = False,
    days_limit: int = 3
) -> dict:
    """
    Fetch weather data from Visual Crossing with options for historical, forecast, and hourly data.
    
    Args:
        city (str): Location city or place name.
        start_date (str): Start date 'YYYY-MM-DD' (optional, default today).
        end_date (str): End date 'YYYY-MM-DD' (optional, default same as start_date).
        include_hours (bool): If True, include hourly data; otherwise daily data.
        days_limit (int): Number of days to return (only applicable for daily data).
        
    Returns:
        dict: weather data with source, location, and forecast info.
    """
    
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}"
    params = {
        "key": visualcrossing_key,
        "unitGroup": "metric",
        "include": "hours" if include_hours else "days"
    }
    
    if start_date:
        base_url += f"/{start_date}"
    if end_date:
        base_url += f"/{end_date}"
    elif start_date:
        # if only start_date provided, set end_date same as start_date
        base_url += f"/{start_date}"
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        location = data.get("resolvedAddress", city)
        
        if include_hours:
            # Collect hourly data for the specified date range
            forecast = []
            for day in data.get("days", []):
                for hour in day.get("hours", []):
                    forecast.append({
                        "datetime": hour.get("datetime"),
                        "temp": hour.get("temp"),
                        "description": hour.get("conditions"),
                        "humidity": hour.get("humidity"),
                        "wind_kph": hour.get("windspeed"),
                    })
        else:
            # Collect daily data, limit to days_limit
            forecast = [
                {
                    "date": day.get("datetime"),
                    "temp": day.get("temp"),
                    "description": day.get("conditions"),
                    "humidity": day.get("humidity"),
                    "wind_kph": day.get("windspeed"),
                }
                for day in data.get("days", [])[:days_limit]
            ]
        
        return {"source": "Visual Crossing", "location": location, "forecast": forecast}
    else:
        return {"error": f"Visual Crossing error: {response.status_code}"}


@function_tool  
def get_weather(city: str) -> dict:
    """
    Fetch weather information for a specific city.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather
    else:
        return {"error": f"Failed to get weather data: {response.status_code}"}



agent = Agent(
    name="WeatherAssistant",
    instructions="""You are a helpful assistant that fetches weather forecasts using different APIs.
      you give weather details in proper chart and icons which show weather conditions. also give time in 
      proper format like 21 may 12:00 PM""",
    model=model,
    tools=[
        get_weatherapi_forecast,
        get_tomorrow_forecast,
        get_visualcrossing_weather,
        get_weather
    ]
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history1",[])
    await cl.Message(content="Welcome to Weather assistant Enter your query about weather").send()

@cl.on_message
async def handle_message(message:cl.Message):
    history2 = cl.user_session.get("history1")
    history2.append({"role":"user", "content":message.content})
    result = await Runner.run(
        agent,
        input = history2,
        run_config=config
    )
    history2.append({"role":"assistant", "content":result.final_output})
    cl.user_session.set("history1", history2)  # Update the session with the new history
    await cl.Message(content=result.final_output).send()