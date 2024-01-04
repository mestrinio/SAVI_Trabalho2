<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]




<!-- PROJECT LOGO -->
<br />

  <a href="https://github.com/mestrinio/SAVI_Trabalho2">
    <img src="images/logo.png" alt="Logo" width="550" height="350">
  </a>

<h3 align="center">SAVI - Trabalho Prático 2</h3>
<h3 align="center">MuG-21 Fishbed</h3>

<h2><b> Repository Owner: Pedro Martins 103800
<br>Collaborators: Emanuel Ramos 103838 & José Silva 103268 </b></h2>

  <p align="center">
    This repository was created for evaluation @ Advanced Systems of Industrial Vision "SAVI 23-24 Trabalho prático 2".
    <br />
    <!-- <a href="https://github.com/mestrinio/SAVI_Trabalho2"><strong>Explore the Wiki »</strong></a> -->
    <br >
    <a href="https://github.com/mestrinio/SAVI_Trabalho2/issues"> <u>Make Suggestion</u> </a>
  </p>
</div>
<br>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-assignment">About the assignment</a>
    </li>
     <li>
      <a href="#Objectives">Objectives</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
<br>



<!-- ABOUT THE ASSIGNMENT -->
## About the Assignment
<div align="center">
<img  src="images/drawing1.png" alt="colorsegmenter" height="400">
</div>
<br>
This assignment was developed for Advanced Systems of Industrial Vision. The program is defined as an advanced perception system that processes data incoming from 3D sensors and RGB normal cameras. The objective is to detect objects presented in a scene, as well as estimate it's properties
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- Objectives -->
## Objectives
### Training of a classifier in deep learning
The program is capable of detecting objects in a chosen scene, and then calculate various properties of the objects.
The object dataset is fully utilized to develop a deep learning network that is capable of classifying the objects. It is divided in training 80% and testing 20%. After being trained, it is able to calculate precision (global and per class).


### 3D Pre-processing
The systems processes the pointcloud of a scene and isolates the objects. Then calculates the following properties:
- Color
- Height
- Width
- Other relevant properties...


### Classification of the objects in the scene
The segmentation of objects in the pointclouds is used to discover the zone where the object is in the RGB image. From here a sub-image that contains only the object is obtained and given to the previous network for new training.


### Audio description of the scene
A speech synthesizer is used to verbally describe what objects with what properties are found in the scene.


### Perfomance metrics
All the algorithms made are classified by using perfomance metrics. For the object detectors, the following perfomance metrics are defined:
- Precision
- Recall
- F1 Score


### Realtime test
The program can be ran using a RGB-D camera on the real world, and trying the detection of objects on top of a table.




### MuG-21 Fishbed

The actual painting part of the program should accomplish the following requirements:

#### SETUP


#### CONTINUOUS


<!-- GETTING STARTED -->
## Getting Started

This is a Python file, so it should be ran in a dedicated terminal running color_segmenter.py, which is the file that runs the first part of the program, to select the desired values for detections. The second and main part of the program, ar_paint.py, runs the painting part and all it's features, but it has some arguments which will be explained later but can be seen using -h argument.

```
./color_segmenter.py
./ar_paint.py -h
```



## Setup
<h3><b>Libraries</b></h3>

To run the program, the following libraries should be installed:

```
sudo apt install python3 python3-tk
pip install opencv-python
pip install numpy
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### How it works

Run color_segmenter.py:
- Select desired values for color detection using the trackbars;
- Use the masked image to understand if the values are as desired;
- Hit 'W' to save the values to the JSON file.

<br>
<br>
<div align="center">
<img  src="images/colorsegmenter.png" alt="colorsegmenter" height="">
</div>
<div align="center">Choose your detection values then hit 'W' to save them</div>
<br>
<br>

***

Arguments when running ar_paint.py:
- -h (Calls for help argument, that explains the other arguments used);
- -usp (Activates the usage of shake protection).
- -j (Insert the full path to the JSON file created in color_segmenter.py)

<br>
<div align="center">
<img  src="images/runarpaint.png" alt="runarpaint" height="">
</div>
<div align="center">Specify the full JSON file's path after -j and write -usp if you want to activate shake protection</div>
<br>
<br>
<br>

Run ar_paint.py:
- A rememberal for the hotkeys to use during the program should pop-up;

<br>
<div align="center">
<img  src="images/tkinter.png" alt="tkinter" height="300">
</div>
<div align="center">Keybindings window pop-up</div>
<br>
<br>

- Drawing in the blank canvas should start happening when detecting the color on the camera;
- Use the hotkeys to change the brush characteristics, switch to drawing on the webcam capture, draw using the mouse and clean or save the current canvas;

<br>
<div align="center">
<img  src="images/drawings.png" alt="drawings" height="300">
</div>
<div align="center">Start drawing!</div>

##### Keybindings:
- 'R' to change brush color to <p style="color: rgb(255,0,0)">RED</p>
- 'G' to change brush color to <p style="color: rgb(0,255,0)">GREEN</p>
- 'B' to change brush color to <p style="color: rgb(0,0,255)">BLUE</p>
- 'P' to change brush color to <p style="color: rgb(0,0,0)">BLACK</p>
- '+' to increase brush size
- '-' to decrease brush size
- 'X' to use rubber
- 'C' to clear the canvas
- 'W' to save the current canvas to an image file
- 'J' to switch between the white canvas and the webcam
- 'M' start using the mouse to draw
- 'I' stop using the mouse to draw
- 'Q' shutdown the program

***
<br>

<!-- CONTACT -->
## Contact
Emanuel Ramos - eramos@ua.pt


José Silva - josesilva8@ua.pt


Pedro Martins - pedro.mestre@ua.pt


Project Link: [Trabalho Prático 2](https://github.com/mestrinio/SAVI_Trabalho2)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Professor Miguel Oliveira - mriem@ua.pt

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/RobutlerAlberto/RobutlerAlberto.svg?style=for-the-badge
[contributors-url]: https://github.com/mestrinio/SAVI_Trabalho2/graphs/contributors
[product-screenshot]: docs/logo.png
