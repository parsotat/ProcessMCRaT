<!-- # ProcessMCRaT
A collection of python scripts are being developed by Tyler Parsotan in order to aid the general comunity in using the MCRaT code and processing the results of the radiative transfer simulations. These scripts are both tools and examples that can be used or modified to better fit the specific needs of the community.

These scripts and the documentation are under development and are still adapting to the developments that are occuring with MCRaT. 


*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** parsotat, ProcessMCRaT, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]  
[![Google Scholar Badge](https://img.shields.io/badge/Google-Scholar-lightgrey)](https://scholar.google.com/citations?user=cIxaj3MAAAAJ&hl=en)
[![ResearchGate Badge](https://img.shields.io/badge/Research-Gate-9cf)](https://www.researchgate.net/profile/Tyler-Parsotan)
<!-- <a href="https://ascl.net/2005.019"><img src="https://img.shields.io/badge/ascl-2005.019-blue.svg?colorB=262255" alt="ascl:2005.019" /></a> -->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/parsotat/ProcessMCRaT">
    <img src="Doc/ProcessMCRaT_Logo.jpg" alt="Logo">
  </a>

  <h3 align="center">The ProcessMCRaT Library</h3>

  <p align="center">
    The ProcessMCRaT library is a collection of scripts that can be used to process the output of the <a href="https://github.com/lazzati-astro/MCRaT">MCRaT</a> code.
    <br />
    <a href="https://github.com/parsotat/ProcessMCRaT/tree/master/Doc"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/parsotat/ProcessMCRaT">View Demo</a>
    · -->
    <a href="https://github.com/parsotat/ProcessMCRaT/issues">Report Bug</a>
    ·
    <a href="https://github.com/parsotat/ProcessMCRaT/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

A collection of python scripts are being developed  in order to aid the general comunity in using the MCRaT code and processing the results of the radiative transfer simulations. These scripts are both tools and examples that can be used or modified to better fit the specific needs of the community. Some portions of the library are specifically compatible with FLASH hydrodynamics files, such as `mcrat_movie.py`, however the main utility fuctions that are meant to process the output of the MCRaT code are complete.

These scripts and the documentation are under development and are still adapting to the developments that are occuring with MCRaT.  While the documentation is being developed, with example jupyter notebooks, the <a href="#usage">Usage</a> section introduces how to use the main utility functions to produce spectra and light curves from the MCRaT outputs.

### Built With

* [Python](https://www.python.org/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

The following are necessary to use the ProcessMCRaT code :
* The output of a  [MCRaT](https://github.com/lazzati-astro/MCRaT) simulation
* [Python](https://www.python.org/)  -- we recommend installing Python 3 using [Anaconda](https://docs.anaconda.com/anaconda/install/)
    * Necessary packages under Python are numpy, scipy, matplotlib, h5py, pickle, tables and random -- many of these will be included in the base installation if using Anaconda and those that are not can easily be installed using the `conda` command line interface


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/parsotat/ProcessMCRaT.git
   ```
2. Install the ProcessMCRaT library in order to use it on your system by running
   ```sh
   python setup.py install 
   ```
   within the `ProcessMCRaT` folder
3. Test the installation by running `python -c "import mclib"`


<!-- _These steps are provided in greater detail in the  [Documentation](https://github.com/parsotat/ProcessMCRaT/tree/master/Doc)_ -->


<!-- USAGE EXAMPLES -->
## Usage

The typical steps in using the ProcessMCRaT library is as follows:

1. Import ProcessMCRaT's mclib in Python
```sh
import mclib as m
```
2. Create an event file for the MCRaT simulation
```sh
m.event_h5(...)
```
where the arguments that can be passed to the `event_h5` function can be found in the partially completed  [Documentation](https://github.com/parsotat/ProcessMCRaT/tree/master/Doc)

3. Calculate the light curves and spectra from the event file using the `lcur` or `spex` functions


<!-- _For more details, please refer to the [Documentation](https://github.com/parsotat/ProcessMCRaT/tree/master/Doc)_ -->



<!-- ROADMAP -->
## Roadmap

1. In the near future there will be extensive documentation of the code including jupyter notebooks that will help guide users on using the library. 
2. Additionally, the library will be able to be easily installed via the `conda` command line interface
3. The code will include the means to include various instrument response functions in the creation of mock observed light curves, spectra, and polarizations

See the [open issues](https://github.com/parsotat/ProcessMCRaT/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Tyler Parsotan - [Personal Website](https://http://sites.science.oregonstate.edu/~parsotat/) - parsotat@oregonstate.edu

Project Link: [https://github.com/parsotat/ProcessMCRaT](https://github.com/parsotat/ProcessMCRaT)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* In using ProcessMCRaT and the MCRaT codes, we ask that you cite the following papers: 
    * [Lazzati (2016)](https://doi.org/10.3847/0004-637X/829/2/76)
    * [Parsotan & Lazzati (2018)](https://doi.org/10.3847/1538-4357/aaa087)
    * [Parsotan et al. (2018)](https://doi.org/10.3847/1538-4357/aaeed1)
    * [Parsotan et. al. (2020)](https://doi.org/10.3847/1538-4357/ab910f)
* [README Template from: othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/parsotat/ProcessMCRaT.svg?style=for-the-badge
[contributors-url]: https://github.com/parsotat/ProcessMCRaT/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/parsotat/ProcessMCRaT.svg?style=for-the-badge
[forks-url]: https://github.com/parsotat/ProcessMCRaT/network/members
[stars-shield]: https://img.shields.io/github/stars/parsotat/ProcessMCRaT.svg?style=for-the-badge
[stars-url]: https://github.com/parsotat/ProcessMCRaT/stargazers
[issues-shield]: https://img.shields.io/github/issues/parsotat/ProcessMCRaT.svg?style=for-the-badge
[issues-url]: https://github.com/parsotat/ProcessMCRaT/issues
[license-shield]: https://img.shields.io/github/license/parsotat/ProcessMCRaT.svg?style=for-the-badge
 [license-url]: https://github.com/parsotat/ProcessMCRaT/blob/master/LICENSE
<!-- [linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555 
[linkedin-url]: https://linkedin.com/in/parsotat -->
