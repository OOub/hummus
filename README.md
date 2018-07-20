Baal: clock-based spiking neural network simulator

Dependencies
==========================

# Homebrew is used to install the dependencies on macOS:

To install homebrew, open the terminal and run: /usr/bin/ruby -e \"\$(curl -fsSL
https://raw.githubusercontent.com/Homebrew/install/master install)\"

# Premake4 is used to build this project

##### On macOS #####
in the terminal run: brew install premake

##### On Linux Debian #####
in the terminal run: sudo apt-get install premake4

# If the Display Class is being used:

Install Qt 5.10.1 and make sure the Qt Charts add-on is installed (cannot guarantee support for other versions of Qt)

##### On macOS #####

option 1 

in the terminal run: brew install qt5

option 2  

download directly from: https://www.qt.io/download/

##### On Linux Debian #####

option 1

1. download directly from: https://www.qt.io/download/

2. permanently add the Qt dynamic lib path to the LD\_LIBRARY\_PATH by
opening the .bashrc file in your home directory and adding at the end
the lines:

LD\_LIBRARY\_PATH=\[path the Qt dynamic lib path\]  
export LD\_LIBRARY\_PATH 

3. modify the include and library paths of Qt in the premake4 file

option 2

1. in the terminal run: sudo apt-get install qtbase5-dev qtdeclarative5-dev

2. follow instructions from this website to build the Qt Charts add-on
from source: https://github.com/qt/qtcharts

3. make sure QT Charts libraries are in the same folder as the rest of
Qt libraries

Installation
==========================

##### Installation
in the terminal run: premake4 install

##### Uninstallation #####
in the terminal run: premake4 uninstall

Testing
==========================

1. Go to the baal directory and run: premake4 gmake && cd build &&
    make

2. Run the executable testNetwork

##### Optional: only for macOS #####
if Xcode is being used we could convert the application into an Xcode project by running: premake4 xcode3 

Using the simulator
==========================

##### Applications #####

so far there are four applications build using the simulator:

1. testNetwork.hpp: a test neural network made to understand how the simulator works
2. unsupervisedNetwork.hpp: a neural network that uses unsupervised learning to learn toy patterns
3. supervisedNetwork.hpp: a neural network that uses supervised learning to learn toy patterns
4. receptiveFieldsTest.hpp: a 2D neural network using receptive fields to learn poker card pips


##### Classes #####

the simulator is a header-only library with 10 classes:

##### network 
the network class acts as a spike manager

##### network delegate
the networkDelegate class is polymorphic class to handle add-ons

##### neuron
the neuron class defines a neuron and the learning rules dictating its behavior. ##### Any modifications to add new learning rules or neuron types are to be done at this stage.

##### dataParser
The DataParser class is used to input data from files into a vector: 1D data (timestamp, Index) or 2D data (timestamp, X, Y)

##### learningLogger
Add-on to the Network class, used to write the learning rule's output into a log binary file; In other words, which neurons are being modified at each learning epoch. This can be read using the snnReader.m matlab function

##### spikeLogger
Add-on to the Network class, used to write the spiking neural network output into a log binary file. This can be read using the snnReader.m matlab function

##### display
Add-on to the Network class, used to display a GUI of the spiking neural network output. Depends on Qt5

##### inputViewer
The InputViewer class is used by the Display class to show the input neurons. Depends on Qt5

##### outputViewer
The OutputViewer class is used by the Display class to show the output neurons. Depends on Qt5

##### potentialViewer
The PotentialViewer class is used by the Display class to show a specified neuron's potential. Depends on Qt5