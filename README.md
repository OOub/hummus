![Logo](resources/adonis_logo.png)

# Quick Start Guide

## Introduction

Adonis is a header-only hybrid spiking neural network simulator coded using C++ and built first and foremost for computer vision and pattern recognition tasks. The simulator has the ability to run in both **clock-based** and **event-based** modes. In the clock-based mode, neurons are updated at a certain time interval. In the event-based mode, neurons are only updated in response to a spike.

Event-based | Clock-based
------------|------------------
performance | easier algorithms
compatible with neuromorphic platforms | membrane potentials at every timestep
Leaky-Integrate-and-Fire (LIF) neuron with constant current dynamics only | Leaky-Integrate-and-Fire (LIF) neuron with a choice between constant and time-varying current dynamics

We will see later in the guide how to select a mode.

#### **Main Goals**
Adonis was born from the frustratingly complicated endeavour of using the standard simulators to create and work with novel concepts and learning rules that are "outside the box". One of the strong points of this simulator is the ease of implementing new ideas without having to delve into endless lines of code. As such, Adonis was developed with two goals in mind: flexibility and simplicity.

###### Flexibility and Simplicity
In order keep things simple, polymorphic classes with virtual methods were implemented. This basically means we can create a new type of add-on, neuron, or learning rule in a completely separate file by simply inheriting from  a polymorphic class and overriding the available virtual methods. We can focus on the scientific part of our work without worrying about making any changes to the main code.

To easily remember and work with these polymorphic classes, the virtual methods available in each class act like messages that occur in different scenarios. We will break down the structure of each in a diagram further down.

Furthermore, Adonis allows full usage of both **weights** and **axonal conduction delays** in the learning rules

#### **What's provided**
A matlab toolbox called AdonisUtilities is bundled, in order to easily generate data from popular databases to feed into a network, or to read and perform graphical and statistical analysis on the network output.

----------------------

## Dependencies

#### **On macOS**

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

#### **On Linux**

###### Premake 4
Premake 4 is used to build the project. Open a terminal and run ``sudo apt-get install premake4``

###### Qt (optional if no GUI is needed)
The Qt framework version 5.9 or newer is needed when using the GUI to visualise the output of a neural network. To install qt5 on Debian Buster or Ubuntu 18.04, type the following:
``sudo apt-get install qt5-default libqt5charts5 libqt5charts5-dev libqt5qml5 qtdeclarative5-dev qml-module-qtcharts qml-module-qtquick2 qml-module-qtquick-controls qml-module-qtquick-controls2``

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
----------------------

## Testing

1. Go to the Adonis directory and run ``premake4 gmake && cd build && make`` or ``premake4 --without-qt gmake && cd build && make`` to build adonis without any Qt dependencies (you will lose the GUI in the process!)

2. execute ``cd release && ./testNetwork`` to run the spiking neural network

**_Disclaimer: some of the applications bundled with the simulator use a path relative to the executable to use one of the files present in the data folder. As such, executing ``./release/testNetwork`` instead of ``cd release && ./testNetwork`` could lead to an error message when the relative path is set incorrectly_**

#### **Premake Actions and Options**

###### Using xCode on macOS
To use Xcode as an IDE on macOS, go the Adonis base directory and run ``premake4 xcode4``

###### Building Without Qt
In case you do not want to use the Qt GUI, you can build Adonis without any Qt dependencies by running ``premake4 --without-qt gmake`` instead of ``premake4 gmake``

###### Premake Help
Run ``premake4 --help`` for more information

----------------------

## Using The Simulator

#### **Adonis UML Diagram**

![chart](resources/flowchart.png)

**Create a new class in a new file and override any of the pure virtual methods outlined in the diagram to create your own add-on, neuron or learning rule**

#### **Namespace**
all the classes are declared within the ``adonis`` namespace. Check out testNetwork.cpp for an example on how to build and run a spiking neural network.

#### **Important Includes**
* base framework: ``#include "../source/core.hpp"``
* Qt GUI: ``#include "../source/GUI/qtDisplay.hpp"``
* neurons: ``#include "../source/neurons/[filename].hpp"``. Choose the neuron headers to include
* learning rules: ``#include "../source/learningRules/[filename].hpp"`` Choose the learning rule headers to include
* add-ons: ``#include "../source/addOns/[filename].hpp"`` Choose the add-on headers to include

#### **Reading Spike Data**
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

**Example**

```
#include "../source/dataParser.hpp

adonis::DataParser parser;

auto trainingData = parser.readData([path to training file]);
auto testData = parser.readData([path to test file]);
```

the trainingData and testData vectors can then be used to inject spikes into the network either through the **injectSpikeFromData()** method or the appropriate **run()** method which takes care of that for you. Please see below for more details on injecting spikes and running the network

#### **Initialisation**

_** I- Initialising the optional Add-ons**_

  * **Qt display :** display useful information on runtime
  ```
  adonis::QtDisplay qtDisplay;
  ```
  * **Spike logger :** write the network output into a binary file.
  ```
  adonis::SpikeLogger spikeLog(std::string filename);
  ```

**The SpikeLogger binary file starts with an 8 bytes header**

SpikeLogger Binary File Specs |
------------| --------
timestamp | byte 0 to 8
delay | byte 8 to 12
weight | byte 12 to 16
potential | byte 16 to 20
presynaptic neuron ID | byte 20 to 22
postsynaptic neuron ID | byte 22 to 24
layer ID | byte 24 to 26
receptive field row index | byte 26 to 28
receptive field column index | byte 28 to 30
x coordinate | byte 30 to 32
y coordinate | byte 32 to 34


  * **Classification logger :** write into a binary file the spikes from the output layer when the learning is off.
  ```
  adonis::ClassificationLogger classificationLog(std::string filename);
  ```

ClassificationLogger Binary File Specs |
------------| --------
timestamp | byte 0 to 8
presynaptic neuron ID | byte 8 to 10
postsynaptic neuron ID | byte 10 to 12

  * **Potential logger :** write into a binary file the potential of specified neurons or layers of neurons at every timestep when learning is off (only logs potentials at spike times in the event-based mode). **Initialising the potential logger is a two-step process:**

    1. initialising the constructor
    ```
    adonis::PotentialLogger potentialLog(std::string filename)
    ```

    2. Calling the **neuronSelection()** PotentialLogger class method in order to chose which neurons to plot . This method should be called **after** defining all the layers of our network **(PLEASE SEE THE NETWORK CREATION SECTION FOR MORE DETAILS ON HOW TO BUILD NEURON LAYERS)**.
    ```
    // 1. Choosing to log only one neuron via its ID
    potentialLog.neuronSelection(int _neuronID);
    ```
    ```
    // 2. Choosing to log all neurons of a specific layer
    potentialLog.neuronSelection(layer _layerToLog);
    ```

    In order to pass a layer to the method we can use normal indexing on the network getLayers() method, as it outputs a vector of all the layers we built. This looks like this:
    ```
    potentialLog.neuronSelection(network.getLayers()[0]); // logging neurons in the first layer we built
    ```


PotentialLogger Binary File Specs |
------------| --------
timestamp | byte 0 to 8
potential | byte 8 to 12
postsynaptic neuron ID | byte 12 to 14

  * **Myelin plasticity logger :** write the learning rule's output into a binary file; In other words, which neurons are being modified (plastic neurons)at each learning epoch.
  ```
  adonis::MyelinPlasticityLogger mpLog(std::string filename);
  ```

MyelinPlasticityLogger Binary File Specs |
------------| --------
bit size (number of plastic neurons varies) | byte 0 to 8
timestamp | byte 8 to 16
postsynaptic neuron ID | byte 16 to 18
layer ID | byte 18 to 20
receptive field row index | byte 20 to 22
receptive field column index | byte 22 to 24

The next set of bytes depends on the number of plastic neurons (bit size). These are the specs for one such neuron:

Plastic Neurons |
--------------- |
time differences | byte 24 to 32
x coordinate | byte 32 to 34
y coordinate | byte 34 to 36
receptive field row index | byte 36 to 38
receptive field column index | byte 38 to 40

  * **Analysis :** print the classification accuracy of the network
  ```
  adonis::Analysis analysis(std::string test_data_labels)
  ```

Please note, the first layer we build does not have any presynaptic neurons. The presynaptic neuron ID will appear as -1 in such cases. The same strategy is used for neurons without any cartesian coordinates defined (spike logger)

Additionally, all the logger files can be read using the LogReader class in the AdonisUtilities matlab toolbox (more details in the quick start guide provided upon installing the toolbox in matlab).

_**II- Initialising the network**_

To initialise the network we can either initialise it without any add-ons:
```
adonis::Network network;
```

Or we can initialise it with add-ons. In that case, the Network constructor has two arguments: a vector of references for the addon constructors, and a reference for **one** main thread addon.
```
// Constructor to initialise normal add-ons
adonis::Network network({&spikeLogger, &learningLogger});

// Constructor to initialise only a MainThreadAddOn
adonis::Network network(&qtDisplay);

// Constructor to initialise both normal add-ons and a MainThreadAddOn
adonis::Network network({&spikeLogger, &learningLogger}, &qtDisplay);
```

_**III- Initialising the learning rules**_

To initialise the learning rules all we have to do is build a constructor for the learning rule and pass it as a reference to a layer of neurons.

Each neuron has a vector of pointers to learning rule constructors, making it capable of having multiple rules. Look at the stdpTest.cpp in the applications folder for an example of a network with a learning rule

```
// initialising the STDP rule
adonis::STDP stdp(1, 1, 20, 20);

// Creating a layer of 10 neurons with the STDP rule passed as a reference (more details in the network creation section)
network.addLayer<adonis::InputNeuron>(10, 1, 2, {&stdp});
```

Available Learning Rules | Arguments
----------------------- | ------------
STDP | A+, A-, tau+, tau-
RewardModulatedSTDP | Ar+, Ar-, Ap+, Ap-
TimeInvariantSTDP | alpha+, alpha-, beta+, beta-
MyelinPlasticity | delay_alpha, delay_lambda, weight_alpha, weight_lambda

#### **Turning Off Learning**
we can manually stop learning at any time by calling the network method: **turnOffLearning(double timestamp)**

#### **Qt Display Settings**
The QtDisplay class has 4 methods to control the settings:

* **useHardwareAcceleration()** : a bool to control whether to use openGL for faster rendering of the plots
* **trackLayer()** : an int to track a specific layer on the OutputViewer
* **trackInputSublayer()** : an int to track a specific sublayer on the InputViewer
* **trackOutputSublayer()** : an int to track a specific sublayer on the OutputViewer
* **trackNeuron()** : an int to track the membrane potential of a neuron via its ID
* **setTimeWindow()** : a double that defines the time window of the display

#### **Creating The Network**

To create a network we have to add layers of neurons.

Available Neuron Models | Use Case
----------------------- | ------------
InputNeuron | initial layer that is fed external spikes. This neuron fires at every external spike  
LIF | Leaky-Integrate-and-Fire (LIF) with two different synaptic kernels for current dynamics: **constant current** or **time-varying current**
IF | Integrate-and-Fire model. similar to the LIF but without any decay in the membrane potential
DecisionMakingNeuron | LIF neurons with the ability to be labelled at the start of the network, or after the training phase

Available Layer Methods | Use Case | Arguments
----------------------- | --------- | --------
addLayer | 1D neurons | number of neurons, number of receptive fields, number of sublayers, vector of pointers to learning rules, neuron arguments (variadic templating)
add2dLayer | grid of 2D neurons | number of neurons, window size, grid width, grid height, number of sublayers, bool for overlapping receptive fields (1 pixel stride), vector of pointers to learning rules, neuron arguments
addReservoir | randomly-connected reservoir of non-learning 1D neurons for liquid state machines | number of neurons, mean weight, weight standard deviation, feedforward connection probability, feedback connection probability, self-excitation probability, neuron arguments
addDecisionMakingLayer | 1D layer of decision-making neurons for classification | training label filename, bool to assign labels at the start of the network (after training if false), vector of pointers to learning rules, refractoryPeriod, bool for timeDependentCurrent (constant current if false), bool homeostasis, current decay, potential decay, eligibility decay, weight decay, homeostasis decay, homeostasis beta, threshold, resting potential, membrane resistance, externalCurrent

Each of these methods is a template; when calling them a class identifier needs to be specified. Creating a 1D layer of input neurons would look like this:
```
// Creating 10 1D input neurons with 1 receptive field, 2 sublayers and no learning rule (empty vector)
network.addLayer<adonis::InputNeuron>(10, 1, 2, {});

// Creating 10 1D LIF neurons with 1 receptive field, 2 sublayer and an learning rule
network.addLayer<adonis::InputNeuron>(10, 1, 2, {&stdp});
```

The **addReservoir()** method already connects the neurons within the reservoir so there is no need to use any of the available connection methods from the next section to interconnect the reservoir neurons

#### **Connecting Neurons In The Network**

There are currently 4 ways to connect layers of neurons:

Available Connection Methods | Use Case | Arguments
----------------------- | --------- | -------------
allToAll | fully connect all neurons in two layers | presynaptic layer - postsynaptic layer - mean weight - weight standard deviation - mean delay - delay standard deviation - connection probability
lateralInhibition | interconnects neurons in a layer with negative weights | layer - mean weight - weight standard deviation - connection
convolution | connecting two layers according to their receptive fields | presynaptic layer - postsynaptic layer - mean weight - weight standard deviation - mean delay - delay standard deviation - connection probability
pooling | subsampling the receptive fields (translation invariance) | presynaptic layer - postsynaptic layer - mean weight - weight standard deviation - mean delay - delay standard deviation - connection probability

the weight in question is scaled according to the input resistance R. So when weight w=1 the actual weight inside the projections is w/R. Additionally, by default the externalCurrent is set to 100. You can play with these parameters to control the shape of the membrane potential when a spike occurs

```
// Connecting two layers in an all-to-all fashion with a mean weight of 1 +- 0.1,  a 5ms delay +- 2ms and 50% chance of making a connection
network.allToAll(network.getLayers()[0], network.getLayers()[1], 1, 0.1, 5, 2, 50);
```

#### **Injecting Spikes**

* To manually inject a spike into the network, use the **injectSpike(neuronID, timestamp)** method:
```
network.injectSpike(0, 10);
```
here we inject a spike at timestamp 10ms for the first neuron in the first neuron population created.

* If we are working with input data files (eg: trainingData and testData from the Reading Spike Data section) we have two options:

    1. using the **injectSpikeFromData()** method with one argument: a reference (&) to the output of the DataParser **readData()** method. This will look like this ``network.injectSpikeFromData(&trainingData);``

    2. using ``network.run(&trainingData, timestep, &testData, shift);`` which automatically calls **injectSpikeFromData()**. (**PLEASE SEE THE NEXT SECTION - RUNNING THE NETWORK - FOR MORE DETAILS**)

#### **Running The Network**
There are two ways to run a network with the same method **run()**:

* If spikes were manually injected via the **injectSpike()** method, or through an input data file via the **injectSpikeFromData()** then we can run the network for a specific time, with a _runtime_ and _timestep_ parameter

```
network.run(runtime, timestep);
```

* We can also run the network with a reference to a _trainingData_ vector, a _timestep_, an optional reference to a _testData_ vector, and an optional _shift_ parameter that adds time to the overall runtime (to allow enough time to pass in case we are working with delayed spikes. This value shoudl be equivalent to the time window you are working with):
  * inject spikes from training and test data
  * run the network on the training data
  * stop all learning and reset network time
  * run the network on the test data

```
network.run(&trainingData, timestep, &testData, shift);
```

#### **Event-based and Clock-based Mode Selection**
* running the network with a **timestep = 0** will select the **asynchronous**, or **event-based** mode.
* running the network with a **timestep > 0** will select the **clock-based** mode.
