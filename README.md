![Logo](resources/hummus_logo.png)

* Hummus is a spiking neural network simulator coded using C++, built first and foremost for neuromorphic computer vision and pattern recognition tasks.

* Hummus is interfaced with the sepia IO library for event-based devices and it can handle data in the eventstream format for reading and propagating events quickly throughout the network (https://github.com/neuromorphic-paris/sepia for more information on Sepia)

* The simulator can run both synchronously and asynchronously

Event-based | Clock-based
------------|------------------
performance | easier algorithms
made for neuromorphic platforms | neurons updated at every timestep

* My simulator was made on the fly according to the needs of my phd. Not very optimised but quite convenient to use for pattern recognition.

* if you have libtorch (C++ frontend of PyTorch) installed on your system, you can even use PyTorch models directly in your spiking neural network. (eg. a logistic regression online classifier is implemented)

* A matlab toolbox called Hummus Utilities is bundled, in order to easily generate data from popular databases to feed into a network, or to read and perform graphical and statistical analysis on the network output.

* I've also provided some python scripts to read network output and such

#### **Supported OS**
Compilation requires a C++17 compiler and a recent version of CMake. Due to the dependency on the C++17 filesystem library the only supported Operating Systems are:

* macOS Catalina (10.15+)
* Ubuntu Disco (19.04+)
* Debian Buster (10+)

## Installation

Some of the following package are purely optional. If these optional packages are found on your system they will be automatically linked to the hummus library.

###### Install Blaze vectorization library
```
git clone git@bitbucket.org:blaze-lib/blaze.git
```

###### OPTIONAL: install libtorch (easy compatibility with torch models)
~~~~
cd && wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
~~~~

#### macOS specific

###### Install Homebrew
~~~~
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
~~~~

###### Install CMake 3.12+
~~~~
brew install cmake
~~~~

###### OPTIONAL: Install Qt5 (GUI for visualisation)
~~~~
brew install qt5
~~~~

###### OPTIONAL: Install TBB (parallelisation)
~~~~
cd && brew install tbb
~~~~

#### Linux specific

###### Install CMake 3.12+
~~~~
sudo apt-get install cmake
~~~~

###### Install Qt5 (GUI for visualisation)
~~~~
sudo apt-get install qt5-default libqt5charts5 libqt5charts5-dev libqt5qml5 qtdeclarative5-dev qml-module-qtcharts qml-module-qtquick2 qml-module-qtquick-controls2
~~~~

###### Install TBB (parallelisation)
~~~~
sudo apt install libtbb-dev
~~~~

## Building Hummus

From the hummus directory run CMake to build Hummus
~~~~
cmake -S . -B build
cmake --build build
~~~~

#### Available build options
* TORCH: building with libtorch for logistic regression classifier. The location of the TorchConfig.cmake file needs to be specified
~~~~
cmake -DTORCH=ON -DTorch_DIR=/absolute/path/to/share/cmake/Torch/ -S . -B build
cmake --build build
~~~~

* QT: building with Qt for visualisation
~~~~
cmake -DQT=ON -S . -B build
cmake --build build
~~~~

* TBB: building with Intel TBB for parallelisation
~~~~
cmake -DTBB=ON -S . -B build
cmake --build build
~~~~

#### Building for an IDE
an alternative way to build Hummus with an IDE. we'll take the example of XCode on macOS:
~~~~
cmake -S . -B build -GXcode
cmake --open build
~~~~

For example, if you're planning on using an IDE such as XCode with libtorch:

~~~~
cmake -DTORCH=ON -DTorch_DIR=/absolute/path/to/share/cmake/Torch/ -S . -B build -GXcode
cmake --open build
~~~~
