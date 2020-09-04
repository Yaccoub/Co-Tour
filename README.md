# Co-Tour : Applied Machine Learning Project SS20

A tourism analyzing, clustering and recommendation system for the post-COVID period in the city of Munich contributing to the management of the tourism flow of the city and the policies concerning the COVID-19 of the region.


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

can deploy the program using the command window:
```
start_gui
```

This command will open the GUI where you can generate a masked output
stream and an extra video with a virtual background

Additional Features
-------------

Alternatively to a virtual background picture, the user have the
the possibility to use a virtual background video. To use this feature please enter a
video as background and check the box
``background video``


## Versioning

We use [Git](https://github.com/) for versioning. For the versions available, see the [tags on this repository](https://github.com/Yaccoub/Computer_Vision_Challenge).

## Authors

* **Alaeddine Yacoub** - *alaeddine.yacoub@tum.de* -
* **Kheireddine Achour** - *kheireddine.achour@tum.de* -
* **Mohamed Mezghanni** - *mohamed.mezghanni@tum.de* -
* **Oumaima Zneidi** - *oumaima.zneidi@tum.de* -
* **Salem Sfaxi** - *salem.sfaxi@tum.de* -

## License

This project is licensed under the Chair For Data Processing
Technical University of Munich
