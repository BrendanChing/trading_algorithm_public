import requests
import json

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_API_KEY = "J7E8IP0IRthaJebTxv2MODaeT1uFB6nP"
symbol = "PLYM"

start_time = "2024-12-01"  # Adjust as needed
end_time = "2025-01-27"

url = f"{FMP_BASE_URL}/historical-chart/15min/{symbol}"
params = {
    'from': start_time,
    'to': end_time,
    'apikey': FMP_API_KEY
}

response = requests.get(url, params=params)
data = response.json()

print(json.dumps(data, indent=2))
print(f"Total data points fetched: {len(data)}")
