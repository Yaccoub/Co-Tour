# Computer Vision SS 2020 Challenge

Distinction  of foreground and background of a given scenery, the ability to eliminate unwanted
elements of this sequence and optionally creating a background video.


 * For a full description of the project, please read the project documentation included
 in the repository:

   ```https://github.com/Yaccoub/Computer_Vision_Challenge/tree/master/doc```

 * To submit bug reports and feature suggestions, or track changes:

     ```https://github.com/Yaccoub/Computer_Vision_Challenge/issues```

Getting Started
-------------
These instructions will get you a copy of the project up and running on your local machine
for development and testing purposes.

Clone this repo to your local machine using:

```
git clone https://github.com/Yaccoub/Computer_Vision_Challenge.git
```

* The ChokePoint dataset used for this project can be found under

    ```http://arma.sourceforge.net/chokepoint/#download```

* The folder naming in this dataset follows the pattern {P#*_S#_C#} :

    ```<Portal.no.type_Sequence.no_Camera.no>```


* Each folder contains:

    ```all_file.txt        --> list of all images```

    ```bg_img.txt          --> list of background images```

Prerequisites
-------------

This project requires the following modules:

 * Matlab Release 2020a     (https://de.mathworks.com/products/matlab.html)
 * Image Processing Toolbox (https://www.mathworks.com/products/image.html)


Configuration
-------------

 * The user can configure the source paths and the program parameters (Left, Right, Start and N) using:

   - **Graphical User Interface:** *start_gui*

     Created using the Matlab built-in app developemnt framework App Designer, this user friendly GUI
      provides standard components such as buttons, check boxes, trees, and drop-down lists
      to configure and customize the program intuitivly.

   - *config.m*

     Users can manually configure and customize the program parameters by changing the variables
     values in the ``config.m`` file.


Deployment
-------------

After installing the prerequisites and setting up the enviroment you
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
