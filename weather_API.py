import requests
import pandas as pd
url = "https://meteostat.p.rapidapi.com/stations/daily"

# parameters gwn als voorbeeld 
querystring = {"station":"06447","start":"2017-01-01","end":"2021-12-31"}

headers = {
	"x-rapidapi-key": "36a3dc31e5msh88f23068f6e8389p1f439djsnbd2bdd68d59f",
	"x-rapidapi-host": "meteostat.p.rapidapi.com",
	"Content-Type": "application/json"
}

response = requests.get(url, headers=headers, params=querystring)

if response.status_code == 200:
    data = response.json()
    
    # Meteostat usually wraps the data in a 'data' key
    # lijst van dictionaries omzetten in een beter leesbaar dataframe 	
    full_df = pd.DataFrame(data['data'])
    
	# voor taak enkel weten hoeveel regen er elke dag is gevallen (prcp = precipitation)
    df = full_df[['date', 'prcp', 'tsun']]
    
    # tabel laten zien in terminal 
    print(df.to_string(index=False))
    
else:
    print(f"Error: {response.status_code}")
df.to_csv("weather.csv", index=False)
print("Weather data opgeslagen als weather.csv")