# Tripadvisor Web Scraper

Tripadvisor Web Scraper is a part of the group 16 project to analyze the tourism flow and patterns in
Munich amid the COVID-19 Crisis. 

This sub-project will help us
scrape information found in the travel-related website “TripAdvisor”.
Key factors such as Reviews, Ratings, Satisfaction  levels,
Visit period, Origin of the visitor and Trip type (solo, couple, family, business and friends)
of Munich’  tourist attractions and turning them into a database,
helping us later to cluster these attractions and unveil the tourism patterns changes during the corona crisis.

## Installation

You need to download and install google chrome. The according 
Chrome driver should also be downloaded and placed into the same directory as the project file.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the Tripadvisor Web Scraper required packages.

```bash
pip3 install -r requirements.txt
```

## Usage
In order to scrape information related a specific attraction/activity from 
TripAdvisor, change the variable url accordingly. 
 
```python
url = "https://www.tripadvisor.de/Attraction_Review-g187309-d3590402-Reviews-FC_Bayern_Museum-Munich_Upper_Bavaria_Bavaria.html"
```
A csv file with the following name will be created into the same directory 

```python
fileName = "tripadvisor_Munich_Activities_dataset" + datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
```
In order to change the number of the harvested reviews, change the variable totalNumPages accordingly. 

```python
totalNumPages = 30
```
Finally run the get_attraction.py file

```bash
python3 get_attraction.py
```


## Usage