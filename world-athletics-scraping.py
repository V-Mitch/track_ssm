from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

# Links to data
MEN_LINK = "https://www.worldathletics.org/records/all-time-toplists/throws/hammer-throw/outdoor/men/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-31&lastDay=2021-02-09"
WOMEN_LINK = "https://www.worldathletics.org/records/all-time-toplists/throws/hammer-throw/outdoor/women/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08"
year = 2024
first_day = "2024-01-01"
last_day = str(date.today())

# Simple web scraping with BeautifulSoup

def get_and_df_data(url):
    page = urlopen(url)
    soup = BeautifulSoup(page.read(), 'html.parser')
    ranks = []
    for tag in soup.find_all('td', {"data-th": "Rank"}):
        ranks.append(int(tag.getText().strip()))
    marks = []
    for tag in soup.find_all('td', {"data-th": "Mark"}):
        marks.append(float(tag.getText().strip()))
    wind = []
    for tag in soup.find_all('td', {"data-th": "WIND"}):
        wind.append(float(tag.getText().strip()))
    competitors = []
    for tag in soup.find_all('td', {"data-th": "Competitor"}):
        competitors.append(tag.getText().strip())
    dobs = []
    for tag in soup.find_all('td', {"data-th": "DOB"}):
        dobs.append(tag.getText().strip())
    nationalities = []
    for tag in soup.find_all('td', {"data-th": "Nat"}):
        nationalities.append(tag.getText().strip())
    positions = []
    for tag in soup.find_all('td', {"data-th": "Pos"}):
        positions.append(tag.getText().strip())
    venues = []
    for tag in soup.find_all('td', {"data-th": "Venue"}):
        venues.append(tag.getText().strip())
    dates = []
    for tag in soup.find_all('td', {"data-th": "Date"}):
        dates.append(tag.getText().strip())
    result_scores = []
    for tag in soup.find_all('td', {"data-th": "ResultScore"}):
        result_scores.append(int(tag.getText().strip()))
    df_data = {
        'Rank': ranks,
        'Mark': marks,
        'Competitor': competitors,
        'DOB': dobs,
        'Nationality': nationalities,
        'Position': positions,
        'Venue': venues,
        'Date': dates,
        'Result Score': result_scores    
    }
    df = pd.DataFrame(df_data)
    return(df)

list_df = []
for i in range(1,19):
  page = str(i)
  men_link = "https://worldathletics.org/records/all-time-toplists/sprints/100-metres/all/men/senior?regionType=world&timing=electronic&windReading=regular&page="+page+"&bestResultsOnly=false&firstDay="+first_day+"&lastDay="+last_day+"&maxResultsByCountry=all&eventId=10229630&ageCategory=senior"
  df = get_and_df_data(men_link)
  list_df.append(df)
  print('url '+ page + ' saved')
  
stacked_df = pd.concat(list_df, axis = 0)
name_file = "men 100m "+"from_"+first_day+" to_"+last_day+".csv"
stacked_df.to_csv(name_file)
