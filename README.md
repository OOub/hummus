
# Adonis Quick Start Guide

Adonis is a spiking neural network simulator coded using C++. There are currently two versions:
* Adonis_t : a clock-based version of the simulator which includes current dynamics
* Adonis_e : an event-based version of the simulator without current dynamics

## Dependencies

#### On macOS

###### Homebrew
Homebrew is used to easily install macOS dependencies. Open a terminal and run ``/usr/bin/ruby -e “$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/master](https://raw.githubusercontent.com/Homebrew/install/master) install)”``.

###### Premake 4
Premake 4 is used to build the project. Open a terminal and run ``brew install premake``.

###### Qt (optional if no GUI is needed)
The Qt framework is needed when using the GUI to visualise the output of a neural network. The following has been tested with **Qt 5.11.1** and support cannot be guaranteed for other versions.
 
**first option:**  Open a terminal and run ``brew install qt5``

**second option:** 

1. Download directly from https://www.qt.io/download/
2. Select the correct version of Qt
3. Make sure the Qt Charts add-on is selected
4. Open the premake4.lua file and modify the moc, include and library paths depending on where Qt was installed 

### On Linux (Debian and Ubuntu)

###### Premake 4
Premake 4 is used to build the project. Open a terminal and run ``sudo apt-get install premake4``.

###### Qt (optional if no GUI is needed)
The Qt framework is needed when using the GUI to visualise the output of a neural network. The following has been tested with **Qt 5.11.1** and support cannot be guaranteed for other versions.

1. Download directly from https://www.qt.io/download/
2. Select the correct version of Qt
3. Make sure the Qt Charts add-on is selected
4. Open the premake4.lua file and modify the moc, include and library paths depending on where Qt was installed 
5. Open the .bashrc file in your home directory and add these lines:
```
LD\_LIBRARY\_PATH=[path to the Qt dynamic lib]
export LD\_LIBRARY\_PATH 
```

## Testing

1. Go to the Adonis directory and run ``premake4 gmake && cd build &&.
    make``
2. execute ``./release/testNetwork`` to run the spiking neural network.

**_Disclaimer: some of the applications bundled in with the simulator use a path relative to the executable to use one of the files present in the data folder. For such cases, execute ``cd release && ./testNetwork`` instead of ``./release/testNetwork``_**

#### Premake Actions and Options
To use Xcode as an IDE on macOS, go the Adonis base directory and run ``premake4 xcode4``.

In case you do not want a GUI, you can build Adonis without any Qt dependencies by running ``premake4 --without-qt gmake`` instead of ``premake4 gmake``

Run ``premake4 --help`` for more information

## Using the simulator

The Adonis simulator is a header-only C++ library with 10 classes

![flowChart](resources/flowchart.png)

