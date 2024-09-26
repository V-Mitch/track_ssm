from urllib.request import urlopen

# Links to data
MEN_LINK = "https://www.worldathletics.org/records/all-time-toplists/throws/hammer-throw/outdoor/men/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-31&lastDay=2021-02-09"
WOMEN_LINK = "https://www.worldathletics.org/records/all-time-toplists/throws/hammer-throw/outdoor/women/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08"

# Simple web scraping with BeautifulSoup

# Web scraping is a very useful technique to get data from websites, it is basically a way to parse the html code of a website in order to retrieve the information we want. For this, as you can imagine, you need to do the following:  
# 
# * Establish connection with the website
# * Get the html code from the website
# * Analyze the sructure of the html code and locate the information we want to get
# 
# For this example, I'll get the hammer throw top 100 scores for men and women to then save them in a csv file for later analysis. The data will be coming from [World Athletics](https://www.worldathletics.org/), take a look at the page, it has very interesting information.  
# 
# Let's start with the code, we'll follow the steps I listed above.
# 
# ## Establish connection with the website
# 
# This is really easy, there are a lot of python packages that allow you to do this, we'll use **urllib**, you can check the documentation [here](https://docs.python.org/3/library/urllib.html).


# For opening and readning urls
page = urlopen(MEN_LINK)
page
page.read()[:1000]

from bs4 import BeautifulSoup

# When you use page.read() 2 times, it sometimes works incorrectly, so let's try to avoid that
page = urlopen(MEN_LINK)
soup = BeautifulSoup(page.read(), 'html.parser')

soup

soup.find_all('td', {"data-th": "Rank"})

ranks = []

for tag in soup.find_all('td', {"data-th": "Rank"}):
    ranks.append(int(tag.getText().strip()))

ranks

marks = []

for tag in soup.find_all('td', {"data-th": "Mark"}):
    marks.append(float(tag.getText().strip()))


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
    positions.append(int(tag.getText().strip()))

venues = []

for tag in soup.find_all('td', {"data-th": "Venue"}):
    venues.append(tag.getText().strip())

dates = []

for tag in soup.find_all('td', {"data-th": "Date"}):
    dates.append(tag.getText().strip())

result_scores = []

for tag in soup.find_all('td', {"data-th": "ResultScore"}):
    result_scores.append(int(tag.getText().strip()))
    
len(ranks) == len(marks) == len(competitors) == len(dobs) == len(nationalities) == len(positions) == len(venues) == len(dates) == len(result_scores)
    
import pandas as pd

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
df

df.to_csv('Hammer Throw Men.csv')

def get_and_save_data(url, destfile_name):
    page = urlopen(url)
    soup = BeautifulSoup(page.read(), 'html.parser')
    
    ranks = []

    for tag in soup.find_all('td', {"data-th": "Rank"}):
        ranks.append(int(tag.getText().strip()))
    
    marks = []

    for tag in soup.find_all('td', {"data-th": "Mark"}):
        marks.append(float(tag.getText().strip()))


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
    
    df.to_csv(destfile_name)
    
get_and_save_data(WOMEN_LINK, 'Hammer Throw Women.csv')

WOMEN = 'https://www.worldathletics.org/records/all-time-toplists/throws/javelin-throw/outdoor/women/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08'
MEN = 'https://www.worldathletics.org/records/all-time-toplists/throws/javelin-throw/outdoor/men/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08'

get_and_save_data(WOMEN, 'Javelin Throw Women.csv')
get_and_save_data(MEN, 'Javelin Throw Men.csv')

WOMEN = 'https://www.worldathletics.org/records/all-time-toplists/throws/discus-throw/outdoor/women/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08'
MEN = 'https://www.worldathletics.org/records/all-time-toplists/throws/discus-throw/outdoor/men/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08'

get_and_save_data(WOMEN, 'Discus Trow Women.csv')
get_and_save_data(MEN, 'Discus Throw Men.csv')

WOMEN = 'https://www.worldathletics.org/records/all-time-toplists/throws/shot-put/outdoor/women/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08'
MEN = 'https://www.worldathletics.org/records/all-time-toplists/throws/shot-put/outdoor/men/senior?regionType=world&page=1&bestResultsOnly=false&firstDay=1899-12-30&lastDay=2021-02-08'

get_and_save_data(WOMEN, 'Shot Put Women.csv')
get_and_save_data(MEN, 'Shot Put Men.csv')
