# Co-Tour : Applied Machine Intelligence Project SS20

A tourism analysis, clustering and recommendation system for the post-COVID period in the city of Munich contributing to the management of the tourism flow of the city and the policies concerning COVID-19 in the region.


 * For a full description of the project, please read the project documentation included
 in the repository:

   ```https://gitlab.ldv.ei.tum.de/ami2020/group16/-/tree/master/docs```

 * To submit bug reports and feature suggestions, or track changes:

     ```https://gitlab.ldv.ei.tum.de/ami2020/group16/-/issues```

Getting Started
-------------
These instructions will get you a copy of the project up and running on your local machine
for development and testing purposes.

Clone this repo to your local machine using:

```
git clone https://gitlab.ldv.ei.tum.de/ami2020/group16.git
```
TripAdvisor Web Scraping App
-------------

* This project contains a web scraping app for TripAdvisor. This sub-project will help us
scrape information found in the travel-related website “TripAdvisor”.
Key factors such as Reviews, Ratings, Satisfaction  levels, Visit period, Origin of the visitor and Trip type (solo, couple, family, business and friends) of Munich’  tourist attractions and turning them into a database, helping us later to cluster these attractions and unveil the tourism patterns changes during the corona crisis. For further information about this sub-project, the Readme file can be found under :

    ```https://gitlab.ldv.ei.tum.de/ami2020/group16/-/tree/master/Tripadvisor_web_scraper```


Prerequisites
-------------

This project requires the following software:

 * Python stable release 3.8.0        (https://www.python.org/downloads/release/python-380/)
 * TensorFlow stable release 2.3.0    (https://www.tensorflow.org/install)


Configuration
-------------

 * If the project is directly cloned from Gitlab, the database paths are already contained in the ./data directory and implemented in the code. In case of any changes, you can find the requested database in the according sub-directory ./data/...

 * All the required packages and modules that don’t come as part of the python standard library are to be found in the requirements.txt file.



Deployment
-------------

After installing the prerequisites you should set up a Python virtual environment using the command window:
```
pip install virtualenv
```
```
virtualenv venv

```
```
source venv/bin/activate venv

```

You can install the required packages and modules that don’t come as part of the python standard library using the command window:

```
$ conda create --name <env> --file <this file>
```

This command can be used to create an environment and install all the required packages.

To access the Web App, first install Docker on your local machine then with the command window run:

```
$ cd TO ROOT
$ docker-compose build
$ docker-compose up -d
```
The Web App should be then accessible from your web browser using the address http://localhost:8000/ which would land on the home page of Co-Tour


Additional Features
-------------

You can use the web scraping app to create your own database. To do so please follow the instructions under:

```
https://gitlab.ldv.ei.tum.de/ami2020/group16/-/tree/master/Tripadvisor_web_scraper
```
## Run from python files
**Tourist flow analysis**
```
1- K-means_clustring.py
2- tourism_flow_data.py
3- visual K means.py
4- visualise.py
```
**Hotsport forcast**
```
1- hotspot_forecast_data.py
2- hotspot_forecast_train.py
3- hotspot_forecast_prediction.py
```
**Recommendation system**
```
1- K-means_recommendation_system.py
2- scoring_system_data.py
3- recommendation_system.py
```
## Versioning

We use [Gitlab](https://gitlab.ldv.ei.tum.de/) for versioning. For the versions available, see the [tags on this repository](https://gitlab.ldv.ei.tum.de/ami2020/group16/-/commits/master).

## Authors

* **Alaeddine Yacoub** - *alaeddine.yacoub@tum.de* -
* **Kheireddine Achour** - *kheireddine.achour@tum.de* -
* **Stephan Rappenpserger** - *stephan.rappenpserger@tum.de* -
* **Yosra Bahri** - *yosra.bahri@tum.de* -
* **Mohamed Mezghanni** - *mohamed.mezghanni@tum.de* -
* **Oumaima Zneidi** - *oumaima.zneidi@tum.de* -
* **Salem Sfaxi** - *salem.sfaxi@tum.de* -

## License

This project is licensed under the Chair For Data Processing
Technical University of Munich
