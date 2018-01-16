# Baal: clock-based spiking neural network simulator

## Dependencies
==========================

### Homebrew is used to install the dependencies on macOS:

in the terminal run */usr/bin/ruby -e \"\$(curl -fsSL
https://raw.githubusercontent.com/Homebrew/install/master install)\"*
to install homebrew

### Premake4 is used to build this project

##### On macOS
in the terminal run *brew install premake*

##### On Linux Debian 
in the terminal run *sudo apt-get install premake4*

### If the Display Class is being used:

Install Qt 5.7+ and make sure the Qt Charts add-on is installed

##### On macOS

**option 1**  
in the terminal run *brew install qt5*

**option 2**  
download directly from: *https://www.qt.io/download/*

##### On Linux Debian

**option 1**

1. download directly from: *https://www.qt.io/download/*

2. permanently add the Qt dynamic lib path to the *LD\_LIBRARY\_PATH* by
opening the .bashrc file in your home directory and adding at the end
the lines:

*LD\_LIBRARY\_PATH=\[path the Qt dynamic lib path\]*  
*export LD\_LIBRARY\_PATH*  

3. modify the include and library paths of Qt in the premake4 file

**option 2**

1. in the terminal run *sudo apt-get install qtbase5-dev qtdeclarative5-dev*

2. follow instructions from this website to build the Qt Charts add-on
from source: *https://github.com/qt/qtcharts*

3. make sure QT Charts libraries are in the same folder as the rest of
Qt libraries

## Installation
==========================

##### Installation 
in the terminal run *premake4 install*

##### Uninstallation  
in the terminal run *premake4 uninstall*

## Testing
==========================

1. Go to the baal directory and run *premake4 gmake && cd build &&
    make*

2. Run the executable testNetwork

## Using the simulator
==========================

##### Applications

**so far there are three applications build using the simulator:**

1. a test neural network made to understand how the simulator works
2. a neural network that uses unsupervised learning to learn basic patterns
3. a neural network that uses supervised learning to learn basic patterns

##### Classes

the simulator is a header-only library with 8 classes. However, only the **Neuron.hpp** class contains the learning algorithms. 