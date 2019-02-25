![Logo](resources/adonis_logo.png)

# Quick Start Guide

## Introduction

Adonis is a header-only hybrid spiking neural network simulator coded using C++ and built first and foremost for computer vision and pattern recognition tasks. The simulator has the ability to run in both **clock-based** and **event-based** modes. In the clock-based mode, neurons are updated at a certain time interval. In the event-based mode, neurons are only updated in response to a spike.

##### event-based or clock-based?
![events](resources/events.png)

We will see later in the guide how to select a mode.

#### Main Goals
Adonis was born from the frustratingly complicated endeavour of using the standard simulators to create and work with novel concepts and learning rules that are "outside the box". One of the strong points of this simulator is the ease of implementing new ideas without having to delve into endless lines of code. As such, Adonis was developed with two goals in mind: flexibility and simplicity.

##### Flexibility and Simplicity
In order keep things simple, polymorphic classes with virtual methods were implemented. This basically means we can create a new type of add-on, neuron, or learning rule in a completely separate file by simply inheriting from  a polymorphic class and overriding the available virtual methods. We can focus on the scientific part of our work without worrying about making any changes to the main code.

To easily remember and work with these polymorphic classes, the virtual methods available in each class act like messages that occur in different scenarios. We will break down the structure of each in a diagram further down.

Furthermore, Adonis allows full usage of both **weights** and **axonal conduction delays** in the learning rules

#### What's provided
A matlab toolbox called AdonisUtilities is bundled, in order to easily generate data from popular databases to feed into a network, or to read and perform graphical and statistical analysis on the network output.

## Dependencies

#### On macOS

###### Homebrew
Homebrew is used to easily install macOS dependencies. Open a terminal and run ``/usr/bin/ruby -e “$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/master](https://raw.githubusercontent.com/Homebrew/install/master) install)”``

###### Premake 4
Premake 4 is used to build the project. Open a terminal and run ``brew install premake``

###### Qt (optional if no GUI is needed)
The Qt framework is needed when using the GUI to visualise the output of a neural network. The following has been tested with **Qt 5.12.1** and support cannot be guaranteed for other versions

**first option**  
Open a terminal and run ``brew install qt5``

**second option**

1. Download directly from https://www.qt.io/download/
2. Select the correct version of Qt
3. Make sure the Qt Charts add-on is selected
4. Open the premake4.lua file and modify the moc, include and library paths depending on where Qt was installed

#### On Linux

###### Premake 4
Premake 4 is used to build the project. Open a terminal and run ``sudo apt-get install premake4``

###### Qt (optional if no GUI is needed)
The Qt framework version 5.9 or newer is needed when using the GUI to visualise the output of a neural network. To install qt5 on Debian Buster or Ubuntu 18.04, type the following:
``sudo apt-get install qt5-default libqt5charts5 libqt5charts5-dev libqt5qml5 qtdeclarative5-dev qml-module-qtcharts qml-module-qtquick-controls``

This should get you going in terms of dependencies. If your distribution does not support that version (Debian Stretch bundles 5.7), consider downloading the latest Qt manually.

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

1. Go to the Adonis directory and run ``premake4 gmake && cd build && make`` or ``premake4 --without-qt gmake && cd build && make`` to build adonis without any Qt dependencies (you will lose the GUI in the process!)

2. execute ``cd release && ./testNetwork`` to run the spiking neural network

**_Disclaimer: some of the applications bundled with the simulator use a path relative to the executable to use one of the files present in the data folder. As such, executing ``./release/testNetwork`` instead of ``cd release && ./testNetwork`` could lead to an error message when the relative path is set incorrectly_**

#### Premake Actions and Options

###### Using xCode on macOS
To use Xcode as an IDE on macOS, go the Adonis base directory and run ``premake4 xcode4``

###### Building Without Qt
In case you do not want to use the Qt GUI, you can build Adonis without any Qt dependencies by running ``premake4 --without-qt gmake`` instead of ``premake4 gmake``

###### Premake Help
Run ``premake4 --help`` for more information

## Using The Simulator

#### Adonis UML Diagram

![chart](resources/flowchart.png)

**Create a new class in a new file and override any of the pure virtual methods outlined in the diagram to create your own add-on, neuron or learning rule**

#### Namespace
all the classes are declared within the ``adonis`` namespace. Check out testNetwork.cpp for an example on how to build and run a spiking neural network.

###### Important Includes
* base framework: ``#include "../source/core.hpp"``
* Qt GUI: ``#include "../source/GUI/qtDisplay.hpp"``
* neurons: ``#include "../source/neurons/[filename].hpp"``. Choose the neuron headers to include
* learning rules: ``#include "../source/learningRules/[filename].hpp"`` Choose the learning rule headers to include
* add-ons: ``#include "../source/addOns/[filename].hpp"`` Choose the add-on headers to include

###### Reading Spike Data
the DataParser class is used to **parse spike data** from a text file **into a vector of input** via the **readData()** method which take in a string for the location of the input data file

* input is a struct with 5 fields:
  * timestamp
  * neuronID
  * x
  * y
  * sublayerID


* The text files can be formatted as such:
  * 1D input data: _timestamp, index_
  * 2D input data:  _timestamp, X, Y_
  * 2D input data with sublayers (feature maps):  _timestamp, X, Y, sublayerID_

<u>Example<u>

```
#include "../source/dataParser

adonis::DataParser parser;

auto trainingData = parser.readData([path to training file]);
auto testData = parser.readData([path to test file]);
```

the trainingData and testData vectors can then be used to inject spikes into the network either through the **injectSpikeFromData()** method or the appropriate **run()** method which takes care of that for you. Please see below for more details on injecting spikes and running the network

###### Initialisation

* Initialising the optional Add-ons

<!-- * the QtDisplay is initialised as such: ``adonis::QtDisplay qtDisplay;``
* the SpikeLogger and the LearningLogger both take in an std::string as a parameter, to define the name of their corresponding output file. They are initialised as such:
```
adonis::SpikeLogger spikeLogger(std::string("spikeLog"));
adonis::LearningLogger learningLogger(std::string("learningLog"));
``` -->
* Initialising the network

<!-- * if no add-ons are used we can directly initialise the network as such: ``adonis::Network network``

* the Network class can take in a vector of references for the standard delegates:
``adonis::Network network({&spikeLogger, &learningLogger});``

* the Network class can also take in a reference to a main thread delegate (only 1 main thread add-on can be used):
``adonis::Network network(&qtDisplay);``

* if both types of add-ons are being used then we initialise as such:
``adonis::Network network({&spikeLogger, &learningLogger}, &qtDisplay);`` -->

###### Turning Off Learning
we can manually stop learning at any time by calling the network method: **turnOffLearning(double timestamp)**

###### Qt Display Settings
The QtDisplay class has 4 methods to control the settings:

* **useHardwareAcceleration()** : a bool to control whether to use openGL for faster rendering of the plots
* **trackLayer()** : an int to track a specific layer on the OutputViewer
* **trackInputSublayer()** : an int to track a specific sublayer on the InputViewer
* **trackOutputSublayer()** : an int to track a specific sublayer on the OutputViewer
* **trackNeuron()** : an int to track the membrane potential of a neuron via its ID
* **setTimeWindow()** : a double that defines the time window of the display

###### Creating The Network

<!-- * to create neurons defined in a 1D space, we use the Network method **addNeurons()** method

* to create neurons defined in a 2D square grid (for computer vision tasks), we use the Network method **addReceptiveFields()** method which creates a grid where each square in that grid contains a separate neuron population. This method allows the network to retain spatial information

* an important parameter in the methods to create neurons is the selection of a learning rule. Each learning rule available inherits from the polymorphic class LearningRuleHandler:

1. The correct learning rule needs to be in the includes.
2. A LearningRuleHandler object needs to be created, and passed as a reference to the neuron creation methods, and if we don't want a learning rule, we simply pass a **nullptr**

currently, 3 learning rules are implemented: MyelinPlasticity, MyelinPlasticityReinforcement, and Stdp. -->

###### Connecting The Network

<!-- * the network getter **getNeuronPopulations()** returns a vector of neuron populations that we just created. This getter returns a struct with 3 fields: **rfNeurons** a vector of neurons belonging to a population, **rfID** the ID of a receptive field in case the **addReceptivefields()** method was used, and **layerID** the ID of the layer a population belongs to.

* the network method **allToAllConnectivity()** connects all neurons of a presynaptic population with all neurons from a postsynaptic population. It has 7 parameters:

1. a reference to a presynaptic neuron population
2. a reference to a postsynaptic neuron population
3. a bool to randomise the projection weights around a value
4. the weight value in question
5. a bool to randomise the projection delays around a value
6. the delay value in question
7. a bool to allow redundant connectivity (more than one projection between a set of neurons)

**it is important to note that the weight in question is scaled according to the input resistance R. So when weight w=1 the actual weight inside the projections is w/R. Additionally, by default the externalCurrent is set to 100. You can play with these parameters to control the shape of the membrane potential when a spike occurs**

the following is an example of connectivity between 2 layers, with fixed weights, random delays with a maximum value of 20, and no redundant connectivity:
```
network.allToAllConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, true, 20, false);
``` -->

###### Injecting Spikes

* To manually inject a spike into the network, use the **injectSpike(neuronID, timestamp)** method:
```
network.injectSpike(0, 10);
```
here we inject a spike at timestamp 10ms for the first neuron in the first neuron population created.

* If we are working with input data files (eg: trainingData and testData from the Reading Spike Data section) we have two options:

    1. using the **injectSpikeFromData()** method with one argument: a reference (&) to the output of the DataParser **readData()** method
    ```
    network.injectSpikeFromData(&trainingData);
    ```

    2. using ``network.run(trainingData, timestep, timestep, testData, shift);`` which automatically calls **injectSpikeFromData()**.

  **PLEASE SEE THE NEXT SECTION - RUNNING THE NETWORK - FOR MORE DETAILS**


if we are using an input data file we can use the **network.injectSpikeFromData()** method which takes in a reference (&) to the output of either the **readTrainingData()** or **readTestData()** method.

###### Running The Network
There are two ways to run a network with the same method **run()**:

1. If spikes were manually injected via the **injectSpike()** method, or through an input data file via the **injectSpikeFromData()** then we can run the network for a specific time, with a _runtime_ and _timestep_ parameter

```
network.run(runtime, timestep);
```

2. We can also run the network with _trainingData_ vector, a _timestep_, an optional _testData_ vector, and an optional _shift_ parameter that adds time to the overall runtime (to allow enough time to pass in case we are working with delayed spikes. This value shoudl be equivalent to the time window you are working with):
  * inject spikes from training and test data
  * run the network on the training data
  * stop all learning and reset network time
  * run the network on the test data

```
network.run(trainingData, timestep, timestep, testData, shift);
```

###### Event-based and Clock-based Mode Selection
* running the network with a **timestep = 0** will select the **asynchronous**, or **event-based** mode.
* running the network with a **timestep > 0** will select the **clock-based** mode.
