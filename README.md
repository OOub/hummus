![Logo](resources/hummus_logo.png)

Lemon Juice, chickpeas, tahini, and a dash of olive oil, spice things up and you're good to go!

## Why Hummus
* Hummus is a spiking neural network simulator coded using C++, built first and foremost for neuromorphic computer vision and pattern recognition tasks.
* Hummus is interfaced with the sepia IO library for event-based devices and it can handle data in the eventstream format for reading and propagating events quickly throughout the network (https://github.com/neuromorphic-paris/sepia for more information on Sepia)
* The simulator can run both **synchronously** (with a clock) and **asynchronously** (event-based) via a simple parameter

Event-based | Clock-based
------------|------------------
performance | easier algorithms
made for neuromorphic platforms | neurons updated at every timestep

#### **Ok but seriously, why?**
Hummus was born because of the inflexibility of other simulators to adapt according the needs of the neuromorphic engineering field. We wanted tp **easily explore original learning rules**, work with **neurons that include non-linear event-based current dynamics**, and create biologically-unrealistic networks tailored specifically for certain event-based computer-vision tasks. This simulator allows us to easily implement new ideas without having to delve into endless lines of code, and it was developed with two goals in mind: flexibility and simplicity

#### **What's so flexible about it?**
* Polymorphic classes with virtual methods were implemented: we can create a new type of add-on, neuron, or synapse in a completely separate header file without looking at the main code
* Hummus has axons that are characterised by both **weights** and **delays**

#### **What about analysis and such?**

###### Matlab Toolbox
A matlab toolbox called Hummus Utilities is bundled, in order to easily generate data from popular databases to feed into a network, or to read and perform graphical and statistical analysis on the network output.

to install:

1. go to the hummus directory
2. go to ``utilities/matlab/``
3. double click on **Hummus Utilities.mltbx**
4. read the bundled quick start guide in Matlab for examples on how to use the toolbox

###### Python modules
* log_reader.py to read binary files output by Hummus (spikeLogger and PotentialLogger implemented so far)
* events_io.py to read from different type of event files (td events etc...)
* dataset_converter.py converts the most popular neuromorphic datasets into the .es format for compatibility with Sepia (POKER-DVS, N-MNIST, GESTURE-DVS)

## Dependencies
* Homebrew    - **mac only**
* CMake 3.12+
* Qt 5.9+     - **optional**
* intel TBB   - **optional**
* libtorch    - **optional**

#### **Supported OS**
Compilation requires a C++17 compiler and a recent version of CMake. Due to the dependency on the C++17 filesystem library the only supported Operating Systems are:

* macOS Catalina (10.15+)
* Ubuntu Disco (19.04+)
* Debian Buster (10+)

## Installing the dependencies

#### **On macOS**

1. install homebrew
~~~~
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
~~~~

2. proceed to the CMake installation
~~~~
brew install cmake
~~~~

3. Install Qt5 (optional: enables usage of the GUI for visualisation)
~~~~
brew install qt5
~~~~

4. Install TBB (optional: enables the parallelisation tbb library. Unused within Hummus. safe to ignore)
~~~~
brew install tbb
~~~~

5. Install libtorch (optional: for classification purposes for example. Unused within Hummus. safe to ignore)
~~~~
cd
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip
unzip libtorch-shared-with-deps-latest.zip
~~~~

disclaimer: we're saving it on the home directory for simplicity.

#### **On Linux**

1. install cmake
~~~~
sudo apt-get install cmake
~~~~

2. Install Qt5 (optional: enables usage of the GUI for visualisation)
~~~~
sudo apt-get install qt5-default libqt5charts5 libqt5charts5-dev libqt5qml5 qtdeclarative5-dev qml-module-qtcharts qml-module-qtquick2 qml-module-qtquick-controls2
~~~~

3. Install TBB (optional: enables the parallelisation tbb library for use to run networks multiple times in parallel)
~~~~
sudo apt install libtbb-dev
~~~~

4. Install libtorch (optional: for classification purposes for example. Unused within Hummus. safe to ignore)
~~~~
cd
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
~~~~

disclaimer: we're saving it on the home directory for simplicity.

## Building Hummus

From the hummus directory run CMake to build Hummus
~~~~
cmake -S . -B build
cmake --build build
~~~~

#### **Building for an IDE**
an alternative way to build Hummus with an IDE. we'll take the example of XCode on macOS:
~~~~
cmake -S . -B build -GXcode
cmake --open build
~~~~

#### **Available compiler flags**
* -DQT (ON by default) - Compiles with Qt5
* -DTBB (OFF by default) - Compiles with TBB
* -DTORCH (OFF by default) - Compiles with libtorch (C++ frontend of pytorch)

add the compiler flags right after cmake. For example, if we want to build Hummus without the Qt5 dependency:
~~~~
cmake -DQT=OFF -S . -B build
cmake --build build
~~~~

## Testing
from the base directory of hummus we can run the basic_test application to check if everything is running correctly
~~~~
cd build/release && ./basic_test
~~~~
