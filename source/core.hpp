/* 
 * core.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: Last Version: 24/01/2019
 *
 * Information: the core.hpp contains both the network and the polymorphic Neuron class:
 *  - The Network class acts as a spike manager
 *  - The Neuron class defines a neuron and its parameters. It can take in a pointer to a LearningRuleHandler object to define which learning rule it follows. The weight is automatically scaled depending on the input resistance used.
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>
#include <mutex>
#include <deque>

#include "dataParser.hpp"
#include "addOn.hpp"
#include "mainThreadAddOn.hpp"
#include "learningRuleHandler.hpp"

namespace hummus {
    
	class Neuron;
	
    // to be used as feature maps
	struct sublayer {
		std::vector<std::size_t>    neurons;
		int16_t                     ID;
	};
	
    // structure containing a population of neurons
	struct layer {
		std::vector<sublayer>       sublayers;
        std::vector<std::size_t>    neurons;
		int16_t                     ID;
		int                         width;
		int                         height;
	};
	
    struct axon {
        Neuron*  preNeuron;
        Neuron*  postNeuron;
        float    weight;
        float    delay;
        double   previousInputTime;
    };
    
    // used for the event-based mode only in order to predict spike times with dynamic currents
    enum class spikeType {
        normal,
        endOfIntegration,
        prediction,
        none
    };
    
    struct spike {
        double    timestamp;
        axon*     propagationAxon;
        spikeType type;
    };
    
	class Network;
	
	class Neuron {
        
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
    	Neuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9) :
                neuronID(_neuronID),
                rfRow(_rfRow),
                rfCol(_rfCol),
                sublayerID(_sublayerID),
                layerID(_layerID),
                xCoordinate(_xCoordinate),
                yCoordinate(_yCoordinate),
                threshold(_threshold),
                potential(_restingPotential),
                restingPotential(_restingPotential),
                initialAxon{nullptr, nullptr, 100/_membraneResistance, 0, 0},
                learningRuleHandler(_learningRuleHandler),
                eligibilityTrace(0),
                eligibilityDecay(_eligibilityDecay),
                membraneResistance(_membraneResistance),
                previousSpikeTime(0),
                previousInputTime(0),
                synapticEfficacy(1) {}
    	
		virtual ~Neuron(){}
		
		// ----- PUBLIC METHODS -----
		
		// ability to do things inside a neuron, outside the constructor before the network actually runs
		virtual void initialisation(Network* network){}
		
		// asynchronous update method
		virtual void update(double timestamp, axon* a, Network* network, spikeType type) = 0;
        
		// synchronous update method
		virtual void updateSync(double timestamp, axon* a, Network* network, double timestep) {
			update(timestamp, a, network, spikeType::none);
		}
        
        // reset a neuron to its initial status
        virtual void resetNeuron(Network* network) {
            previousInputTime = 0;
            previousSpikeTime = 0;
            potential = restingPotential;
            eligibilityTrace = 0;
        }
        
        // adds an axon that connects two Neurons together - used to propagate spikes
        void addAxon(Neuron* postNeuron, float weight, float delay, int probability=100, bool redundantConnections=true) {
            if (postNeuron) {
                if (connectionProbability(probability)) {
                    if (redundantConnections == false) {
                        int16_t ID = postNeuron->getNeuronID();
                        auto result = std::find_if(postAxons.begin(), postAxons.end(), [&](std::unique_ptr<axon>& a){return a->postNeuron->getNeuronID() == ID;});
                        
                        if (result == postAxons.end()) {
                            postAxons.emplace_back(new axon{this, postNeuron, weight*(1/postNeuron->getMembraneResistance()), delay, 0});
                            postNeuron->getPreAxons().emplace_back(postAxons.back().get());
                        } else {
#ifndef NDEBUG
                            std::cout << "axon " << neuronID << "->" << postNeuron->getNeuronID() << " already exists" << std::endl;
#endif
                        }
                    }
                    else {
                        postAxons.emplace_back(new axon{this, postNeuron, weight*(1/postNeuron->getMembraneResistance()), delay, 0});
                        postNeuron->getPreAxons().emplace_back(postAxons.back().get());
                    }
                }
            } else {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
        // initialise the initial axon when a neuron receives an input event
        spike prepareInitialSpike(double timestamp) {
            if (!initialAxon.postNeuron) {
                initialAxon.postNeuron = this;
            }
            return spike{timestamp, &initialAxon, spikeType::normal};
        }
        
		// utility function that returns true or false depending on a probability percentage
		static bool connectionProbability(int probability) {
			std::random_device device;
			std::mt19937 randomEngine(device());
			std::bernoulli_distribution dist(probability/100.);
			return dist(randomEngine);
		}
		
		// ----- SETTERS AND GETTERS -----        
		int16_t getNeuronID() const {
            return neuronID;
        }
		
		int16_t getRfRow() const {
			return rfRow;
		}
		
		int16_t getRfCol() const {
			return rfCol;
		}
		
		int16_t getSublayerID() const {
			return sublayerID;
		}
		
		int16_t getLayerID() const {
			return layerID;
		}
		
		int16_t getX() const {
		    return xCoordinate;
		}
		
		int16_t getY() const {
		    return yCoordinate;
		}
        
        std::vector<axon*>& getPreAxons() {
            return preAxons;
        }
        
        std::vector<std::unique_ptr<axon>>& getPostAxons() {
            return postAxons;
        }
        
        axon& getInitialAxon() {
            return initialAxon;
        }
        
        float setPotential(float newPotential) {
            return potential = newPotential;
        }
        
        float getPotential() const {
            return potential;
        }
        
        float getThreshold() const {
            return threshold;
        }
        
        float setThreshold(float _threshold) {
            return threshold = _threshold;
        }
        
        std::vector<LearningRuleHandler*> getLearningRuleHandler() const {
            return learningRuleHandler;
        }
        
        void addLearningRule(LearningRuleHandler* newRule) {
            learningRuleHandler.emplace_back(newRule);
        }
        
        float getMembraneResistance() const {
            return membraneResistance;
        }
        
        float getEligibilityTrace() const {
            return eligibilityTrace;
        }
        
        float getEligibilityDecay() const {
            return eligibilityDecay;
        }
        
        void setEligibilityTrace(float newtrace) {
            eligibilityTrace = newtrace;
        }
        
        double getPreviousSpikeTime() const {
            return previousSpikeTime;
        }
        
        double getPreviousInputTime() const {
            return previousInputTime;
        }
        
        float getSynapticEfficacy() const {
            return synapticEfficacy;
        }
        
        float setSynapticEfficacy(float newEfficacy) {
            return synapticEfficacy = newEfficacy;
        }
        
    protected:
        
        // winner-take-all algorithm
        virtual void WTA(double timestamp, Network* network) {}
        
        // loops through any learning rules and activates them
        virtual void requestLearning(double timestamp, axon* a, Network* network){}
        
		// ----- NEURON PARAMETERS -----
        int16_t                            neuronID;
		int16_t                            rfRow;
		int16_t                            rfCol;
		int16_t                            sublayerID;
		int16_t                            layerID;
		int16_t                            xCoordinate;
		int16_t                            yCoordinate;
		std::vector<axon*>                 preAxons;
        std::vector<std::unique_ptr<axon>> postAxons;
        axon                               initialAxon;
        float                              threshold;
        float                              potential;
        float                              restingPotential;
        std::vector<LearningRuleHandler*>  learningRuleHandler;
        float                              eligibilityTrace;
        float                              eligibilityDecay;
        float                              membraneResistance;
        double                             previousSpikeTime;
        double                             previousInputTime;
        float                              synapticEfficacy;
    };
	
    class Network {
        
    public:
		
		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network(std::vector<AddOn*> _addOns = {}, MainThreadAddOn* _thAddOn = nullptr) :
                addOns(_addOns),
                thAddOn(_thAddOn),
                learningStatus(true),
                asynchronous(false),
                learningOffSignal(-1),
                maxDelay(0) {}
		
		Network(MainThreadAddOn* _thAddOn) : Network({}, _thAddOn) {}
		
		// ----- NEURON CREATION METHODS -----
		
        // adds one dimensional neurons
        template <typename T, typename... Args>
        void addLayer(int _numberOfNeurons, std::vector<LearningRuleHandler*> _learningRuleHandler, Args&&... args) {
            
            if (_numberOfNeurons < 0) {
                throw std::logic_error("the number of neurons selected is wrong");
            }
            
            unsigned long shift = 0;
            
            // find the layer ID
            int16_t layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }

            // building a layer of one dimensional sublayers
            std::vector<std::size_t> neuronsInLayer;
            for (int16_t k=0+shift; k<_numberOfNeurons+shift; k++) {
                neurons.emplace_back(std::unique_ptr<T>(new T(k, 0, 0, 0, layerID, -1, -1, _learningRuleHandler, std::forward<Args>(args)...)));
                neuronsInLayer.emplace_back(neurons.size()-1);
            }
            
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1});
        }
		
        // adds a 2 dimensional grid of neurons
        template <typename T, typename... Args>
        void add2dLayer(int gridW, int gridH, int _sublayerNumber, std::vector<LearningRuleHandler*> _learningRuleHandler, Args&&... args) {
            
            // find number of neurons to build
            int numberOfNeurons = gridW * gridH;
            
            unsigned long shift = 0;
            
            // find the layer ID
            int16_t layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // building a layer of two dimensional sublayers
            int16_t counter = 0;
            std::vector<sublayer> sublayers;
            std::vector<std::size_t> neuronsInLayer;
            for (int16_t i=0; i<_sublayerNumber; i++) {
                std::vector<std::size_t> neuronsInSublayer;
                int x = 0; int y = 0;
                for (int16_t k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(std::unique_ptr<T>(new T(k+counter, 0, 0, i, layerID, x, y, _learningRuleHandler, std::forward<Args>(args)...)));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);
                    
                    y += 1;
                    if (y == gridW) {
                        x += 1;
                        y = 0;
                    }
                }
                sublayers.emplace_back(sublayer{neuronsInSublayer, i});
                
                // to shift the neuron IDs with the sublayers
                counter += numberOfNeurons;
            }
            layers.emplace_back(layer{sublayers, neuronsInLayer, layerID, gridW, gridH});
        }
		
        // creates a layer that is a convolution of the previous layer, depending on the kernel size and the stride
        void addConvolutionalLayer(layer presynapticLayer, int kernelSize, int stride=1, float _weightMean=1, float _weightstdev=0, int _delayMean=0, int _delaystdev=0, int probability=100) {
            maxDelay = std::max(maxDelay, _delayMean);
            
            std::cout << "convoluting 1D layer" << std::endl;
        }
        
        // creates a layer that is a subsampled version of the previous layer, to the nearest divisible grid size
        void addPoolingLayer(layer presynapticLayer, float _weightMean=1, float _weightstdev=0, int _delayMean=0, int _delaystdev=0, int probability=100) {
            maxDelay = std::max(maxDelay, _delayMean);
        }
        
        // add a one dimensional layer of decision-making neurons that are labelled according to the provided labels - must be on the last layer
        template <typename T>
        void addDecisionMakingLayer(std::string trainingLabelFilename, bool _preTrainingLabelAssignment=true ,std::vector<LearningRuleHandler*> _learningRuleHandler={}, int _refractoryPeriod=1000, bool _timeDependentCurrent=false, bool _homeostasis=false, float _decayCurrent=10, float _decayPotential=20, float _eligibilityDecay=20, float _decayWeight=0, float _decayHomeostasis=10, float _homeostasisBeta=1, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9, float _externalCurrent=100) {
            DataParser dataParser;
            trainingLabels = dataParser.readLabels(trainingLabelFilename);
            preTrainingLabelAssignment = _preTrainingLabelAssignment;
            
            // find number of classes
            for (auto& label: trainingLabels) {
                if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label.name) == uniqueLabels.end()) {
                    uniqueLabels.emplace_back(label.name);
                }
            }
            
            unsigned long shift = 0;
            
            // find the layer ID
            int16_t layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // add decision-making neurons
            std::vector<std::size_t> neuronsInLayer;
            if (preTrainingLabelAssignment) {
                for (int16_t k=0+shift; k<static_cast<int>(uniqueLabels.size())+shift; k++) {
                    neurons.emplace_back(std::unique_ptr<T>(new T(k, 0, 0, 0, layerID, -1, -1, _learningRuleHandler, _timeDependentCurrent, _homeostasis, _decayCurrent, _decayPotential, _refractoryPeriod, _eligibilityDecay, _decayWeight, _decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _membraneResistance, _externalCurrent, uniqueLabels[k-shift])));
                    
                    neuronsInLayer.emplace_back(neurons.size()-1);
                }
            } else {
                for (int16_t k=0+shift; k<static_cast<int>(uniqueLabels.size())+shift; k++) {
                    neurons.emplace_back(std::unique_ptr<T>(new T(k, 0, 0, 0, layerID, -1, -1, _learningRuleHandler, _timeDependentCurrent, _homeostasis, _decayCurrent, _decayPotential, _refractoryPeriod, _eligibilityDecay, _decayWeight, _decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _membraneResistance, _externalCurrent, "")));
                    
                    neuronsInLayer.emplace_back(neurons.size()-1);
                }
            }
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1});
        }
        
        // add a one-dimensional reservoir of randomly interconnected neurons without any learning rule (feedforward, feedback and self-excitation) with randomised weights and no delays.
        template <typename T, typename... Args>
        void addReservoir(int _numberOfNeurons, float _weightMean=1, float _weightstdev=0, int feedforwardProbability=100, int feedbackProbability=0, int selfExcitationProbability=0, Args&&... args) {
            
            if (_numberOfNeurons < 0) {
                throw std::logic_error("the number of neurons selected is wrong");
            }
            
            unsigned long shift = 0;
            
            // find the layer ID
            int16_t layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // creating the reservoir of neurons
            std::vector<std::size_t> neuronsInLayer;
            for (int16_t k=0+shift; k<_numberOfNeurons+shift; k++) {
                neurons.emplace_back(std::unique_ptr<T>(new T(k, 1, 0, 0, layerID, -1, -1, {}, std::forward<Args>(args)...)));
                neuronsInLayer.emplace_back(neurons.size()-1);
            }
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1});
            
            // connecting the reservoir
            for (auto pre: neuronsInLayer) {
                for (auto post: neuronsInLayer) {
                    // random engine initialisation
                    std::random_device device;
                    std::mt19937 randomEngine(device());
                    std::normal_distribution<> weightRandom(_weightMean, _weightstdev);
                    
                    // self-excitation probability
                    if (pre == post) {
                        neurons[pre].get()->addAxon(neurons[post].get(), weightRandom(randomEngine), 0, selfExcitationProbability, true);
                    } else {
                        // feedforward probability
                        neurons[pre].get()->addAxon(neurons[post].get(), weightRandom(randomEngine), 0, feedforwardProbability, true);
                        
                        // feedback probability
                        neurons[post].get()->addAxon(neurons[pre].get(), weightRandom(randomEngine), 0, feedbackProbability, true);
                    }
                }
            }
        }
        
		// ----- LAYER CONNECTION METHODS -----
		
        // all to all purely excitatory or purely inhibitory connections (if weightMean < 0 all connections will have a negative weight)
//        template <typename F>
        void allToAll(layer presynapticLayer, layer postsynapticLayer, float _weightMean=1, float _weightstdev=0, int _delayMean=0, int _delaystdev=0, int probability=100) {
            maxDelay = std::max(maxDelay, _delayMean);

            for (auto& preNeurons: presynapticLayer.neurons) {
                for (auto& postNeurons: postsynapticLayer.neurons) {
                    // randomising weights and delays
                    std::random_device device;
                    std::mt19937 randomEngine(device());
                    std::normal_distribution<> delayRandom(_delayMean, _delaystdev);
                    std::normal_distribution<> weightRandom(_weightMean, _weightstdev);
                    
                    // all weights positive if mean weight is positive and vice-versa
                    int sign = _weightMean<0?-1:_weightMean>=0;
                    
                    neurons[preNeurons].get()->addAxon(neurons[postNeurons].get(), sign*std::abs(weightRandom(randomEngine)), std::abs(delayRandom(randomEngine)), probability, true);
                }
            }
        }
		
        // interconnecting a layer with soft winner-takes-all axons, using negative weights
        void lateralInhibition(layer l, float _weightMean=-1, float _weightstdev=0, int probability=100) {
            if (_weightMean != 0) {
                if (_weightMean > 0) {
                    std::cout << "lateral inhibition axons must have negative weights. The input weight was automatically converted to its negative counterpart" << std::endl;
                }
                
                for (auto& sub: l.sublayers) {
                    // intra-sublayer soft WTA
                    for (auto& preNeurons: sub.neurons) {
                        for (auto& postNeurons: sub.neurons) {
                            if (preNeurons != postNeurons) {
                                std::random_device device;
                                std::mt19937 randomEngine(device());
                                std::normal_distribution<> weightRandom(_weightMean, _weightstdev);
                                neurons[preNeurons].get()->addAxon(neurons[postNeurons].get(), -1*std::abs(weightRandom(randomEngine)), 0, probability, true);
                            }
                        }
                    }
                    
                    // inter-sublayer soft WTA
                    for (auto& subToInhibit: l.sublayers) {
                        if (sub.ID != subToInhibit.ID) {
                            for (auto& preNeurons: sub.neurons) {
                                for (auto& postNeurons: subToInhibit.neurons) {
                                    if (neurons[preNeurons]->getRfRow() == neurons[postNeurons]->getRfRow() && neurons[preNeurons]->getRfCol() == neurons[preNeurons]->getRfCol()) {
                                        std::random_device device;
                                        std::mt19937 randomEngine(device());
                                        std::normal_distribution<> weightRandom(_weightMean, _weightstdev);
                                        neurons[preNeurons].get()->addAxon(neurons[postNeurons].get(), -1*std::abs(weightRandom(randomEngine)), 0, probability, true);
                                    }
                                }
                            }
                        }
                    }
                }
                
            } else {
                throw std::logic_error("lateral inhibition axons cannot have a null weight");
            }
        }

        // ----- PUBLIC NETWORK METHODS -----
        
        // add spike to the network - user-friendly version that wraps a neuron's prepareInitialSpike method
        void injectSpike(int16_t neuronIndex, double timestamp) {
            initialSpikes.push_back(neurons[neuronIndex].get()->prepareInitialSpike(timestamp));
        }
        
        // adding spikes generated by the network
        void injectGeneratedSpike(spike s) {
            generatedSpikes.insert(
                std::upper_bound(generatedSpikes.begin(), generatedSpikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }

        // adding spikes predicted by the asynchronous network (timestep = 0) for synaptic integration
        void injectPredictedSpike(spike s, spikeType stype) {
            // if spike doesn't already exist insert it in the list. if it does, just update the timestamp
            auto it = std::find_if(predictedSpikes.begin(), predictedSpikes.end(),[&](spike oldSpike) {
                return oldSpike.propagationAxon == s.propagationAxon;
            });
            
            if (it != predictedSpikes.end()) {
                auto idx = std::distance(predictedSpikes.begin(), it);
                predictedSpikes[idx].type = stype;
                predictedSpikes[idx].timestamp = s.timestamp;
            } else {
                predictedSpikes.insert(it, s);
            }
            
            // sort timestamps
            std::sort(predictedSpikes.begin(), predictedSpikes.end(), [](spike a, spike b) {
                return a.timestamp < b.timestamp;
            });
        }
        
        // add spikes from file to the network
        void injectSpikeFromData(std::vector<input>* data) {
            // error handling
            if (neurons.empty()) {
                throw std::logic_error("add neurons before injecting spikes");
            }
            
            for (auto& event: *data) {
                for (auto& n: layers[0].neurons) {
                    // 1D or 2D data not split into sublayers
                    if (event.sublayerID == neurons[n]->getSublayerID()) {
                        // one dimensional data
                        if (event.x == -1) {
                            if (neurons[n]->getNeuronID() == event.neuronID) {
                                injectSpike(n, event.timestamp);
                                break;
                            }
                        // two dimensional data
                        } else {
                            if (neurons[n]->getX() == event.x && neurons[n]->getY() == event.y) {
                                injectSpike(n, event.timestamp);
                                break;
                            }
                        }
                        
                    // 2D data split into sublayers
                    } else if (event.sublayerID == -1) {
                        
                    }
                }
            }
        }

        // turn off all learning rules (for cross-validation or test data)
        void turnOffLearning(double timestamp) {
            learningOffSignal = timestamp;
        }

        // running through the network asynchronously if timestep = 0 and synchronously otherwise
        void run(double _runtime, double _timestep=0) {
            // error handling
            if (_timestep < 0) {
                throw std::logic_error("the timestep cannot be negative");
            } else if (_timestep == 0) {
                std::cout << "Running the network asynchronously" << std::endl;
            } else {
                std::cout << "Running the network synchronously" << std::endl;
            }

            if (_timestep == 0) {
                asynchronous = true;
            }
            
            for (auto& n: neurons) {
                n->initialisation(this);
            }

            for (auto addon: addOns) {
                addon->onStart(this);
            }

            std::mutex sync;
            if (thAddOn) {
                sync.lock();
            }

                std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

                runHelper(_runtime, _timestep, false);

                std::cout << "Done." << std::endl;

                std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
                std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;

                for (auto addon: addOns) {
                    addon->onCompleted(this);
                }
            });

            if (thAddOn) {
                thAddOn->begin(this, &sync);
            }

            spikeManager.join();
        }

        // running through the network asynchronously if timestep = 0 and synchronously otherwise. This overloaded method takes in a training and an optional testing dataset instead of a runtime
        void run(std::vector<input>* trainingData, float _timestep=0, std::vector<input>* testData=nullptr, int shift=20) {
            if (_timestep == 0) {
                asynchronous = true;
            }
            
            for (auto& n: neurons) {
                n->initialisation(this);
            }
            
            if (learningOffSignal == -1) {
                learningOffSignal = trainingData->back().timestamp+maxDelay+shift;
            }
            
            for (auto addon: addOns) {
                addon->onStart(this);
            }
            
            std::mutex sync;
            if (thAddOn) {
                sync.lock();
            }
            
            std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();

                // importing training data and running the network through the data
                train(_timestep, trainingData, shift);

                // importing test data and running the network through the data
                if (testData) {
                    predict(_timestep, testData, shift);
                }

                for (auto addon: addOns) {
                    addon->onCompleted(this);
                }
            });

            if (thAddOn) {
                thAddOn->begin(this, &sync);
            }
            spikeManager.join();
        }

        // reset the network back to the initial conditions without changing the network build
        void reset() {
            initialSpikes.clear();
            generatedSpikes.clear();
            learningStatus = true;
            learningOffSignal = -1;
        }

        // ----- SETTERS AND GETTERS -----
        std::vector<std::unique_ptr<Neuron>>& getNeurons() {
            return neurons;
        }

        std::vector<layer>& getLayers() {
            return layers;
        }

        std::vector<AddOn*>& getAddOns() {
            return addOns;
        }

        MainThreadAddOn* getMainThreadAddOn() {
            return thAddOn;
        }

        std::deque<spike>& getGeneratedSpikes() {
            return generatedSpikes;
        }

        bool getLearningStatus() const {
            return learningStatus;
        }
        
        double getLearningOffSignal() const {
            return learningOffSignal;
        }
        
        std::string getCurrentLabel() const {
            return currentLabel;
        }
        
        std::vector<std::string>& getUniqueLabels() {
            return uniqueLabels;
        }

        bool getPreTrainingLabelAssignment() const {
            return preTrainingLabelAssignment;
        }
        
        bool getNetworkType() const {
            return asynchronous;
        }
        
    protected:

        // -----PROTECTED NETWORK METHODS -----

        // importing training data and running the network through the data
        void train(double timestep, std::vector<input>* trainingData, int shift) {
            injectSpikeFromData(trainingData);
            
            std::cout << "Training the network..." << std::endl;
            std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
        
            runHelper(trainingData->back().timestamp+maxDelay+shift, timestep, false);

            std::cout << "Done." << std::endl;
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
            std::cout << "it took " << elapsed_seconds.count() << "s for the training phase." << std::endl;
        }

        // importing test data and running it through the network for classification
        void predict(double timestep, std::vector<input>* testData, int shift) {
            learningStatus = false;
            initialSpikes.clear();
            generatedSpikes.clear();
            for (auto& n: neurons) {
                n->resetNeuron(this);
            }

            injectSpikeFromData(testData);

            std::cout << "Running classification based on a trained network..." << std::endl;

            runHelper(testData->back().timestamp+maxDelay+shift, timestep, true);

            std::cout << "Done." << std::endl;
        }

        void runHelper(double runtime, double timestep, bool classification=false) {

            if (!neurons.empty()) {
                if (timestep == 0) {
                    while (!initialSpikes.empty() || !generatedSpikes.empty() || !predictedSpikes.empty()) {
                        // if only one list is filled
                        if (predictedSpikes.empty() && generatedSpikes.empty() && !initialSpikes.empty()) {
                            update(initialSpikes.front(), classification);
                            initialSpikes.pop_front();
                        } else if (predictedSpikes.empty() && initialSpikes.empty() && !generatedSpikes.empty()) {
                            update(generatedSpikes.front(), classification);
                            generatedSpikes.pop_front();
                        } else if (generatedSpikes.empty() && initialSpikes.empty() && !predictedSpikes.empty()) {
                            update(predictedSpikes.front(), classification, true);
                            predictedSpikes.pop_front();
                        // if two lists are filled
                        } else if (predictedSpikes.empty() && !generatedSpikes.empty() && !initialSpikes.empty()){
                            if (initialSpikes.front().timestamp < generatedSpikes.front().timestamp) {
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            } else if (generatedSpikes.front().timestamp < initialSpikes.front().timestamp) {
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                            } else {
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                                
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            }
                        } else if (generatedSpikes.empty() && !predictedSpikes.empty() && !initialSpikes.empty()){
                            if (predictedSpikes.front().timestamp < initialSpikes.front().timestamp) {
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                            } else if (initialSpikes.front().timestamp < predictedSpikes.front().timestamp) {
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            } else {
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                                
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            }
                        } else if (initialSpikes.empty() && !predictedSpikes.empty() && !generatedSpikes.empty()){
                            if (predictedSpikes.front().timestamp < generatedSpikes.front().timestamp) {
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                            } else if (generatedSpikes.front().timestamp < predictedSpikes.front().timestamp) {
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                            } else {
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                                
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                            }
                        // if all lists are filled
                        } else {
                            // if one list spikes before the others
                            if (initialSpikes.front().timestamp < generatedSpikes.front().timestamp && initialSpikes.front().timestamp < predictedSpikes.front().timestamp) {
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            } else if (generatedSpikes.front().timestamp < initialSpikes.front().timestamp && generatedSpikes.front().timestamp < predictedSpikes.front().timestamp) {
                                
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                            } else if (predictedSpikes.front().timestamp < generatedSpikes.front().timestamp && predictedSpikes.front().timestamp < initialSpikes.front().timestamp) {
                                
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                            // if two lists spike at the same time, define order of spike
                            } else if (generatedSpikes.front().timestamp == initialSpikes.front().timestamp){
                                
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                                
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            } else if (generatedSpikes.front().timestamp == predictedSpikes.front().timestamp){
                                
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                                
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                            
                            } else if (initialSpikes.front().timestamp == predictedSpikes.front().timestamp){
                                
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                                
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                                // if they all spike at the same time
                            } else {
                                
                                update(generatedSpikes.front(), classification);
                                generatedSpikes.pop_front();
                                
                                update(predictedSpikes.front(), classification, true);
                                predictedSpikes.pop_front();
                                
                                update(initialSpikes.front(), classification);
                                initialSpikes.pop_front();
                            }
                        }
                    }
                } else {
                   
                    for (double i=0; i<runtime; i+=timestep) {
                        
                        if (!classification) {
                            if (!trainingLabels.empty()) {
                                if (trainingLabels.front().onset <= i) {
                                    currentLabel = trainingLabels.front().name;
                                    trainingLabels.pop_front();
                                }
                            }

                            if (learningOffSignal != -1) {
                                if (learningStatus==true && i >= learningOffSignal) {
                                    std::cout << "learning turned off at t=" << i << std::endl;
                                    learningStatus = false;
                                }
                            }
                        }

                        std::vector<spike> currentSpikes;
                        if (generatedSpikes.empty() && !initialSpikes.empty()) {

                            while (!initialSpikes.empty() && std::round(initialSpikes.front().timestamp) <= i) {
                                currentSpikes.emplace_back(initialSpikes.front());
                                initialSpikes.pop_front();
                            }
                        }
                        else if (initialSpikes.empty() && !generatedSpikes.empty()) {
                            while (!generatedSpikes.empty() && std::round(generatedSpikes.front().timestamp) <= i) {
                                currentSpikes.emplace_back(generatedSpikes.front());
                                generatedSpikes.pop_front();
                            }
                        }
                        else {
                            while (!generatedSpikes.empty() && std::round(generatedSpikes.front().timestamp) <= i) {
                                currentSpikes.emplace_back(generatedSpikes.front());
                                generatedSpikes.pop_front();
                            }
                            
                            while (!initialSpikes.empty() && std::round(initialSpikes.front().timestamp) <= i) {
                                currentSpikes.emplace_back(initialSpikes.front());
                                initialSpikes.pop_front();
                            }
                        }
                        
                        for (auto& n: neurons) {
                            std::vector<spike> local_currentSpikes(currentSpikes.size());
                            const auto it = std::copy_if(currentSpikes.begin(), currentSpikes.end(), local_currentSpikes.begin(), [&](const spike s) {
                                if (s.propagationAxon) {
                                    return s.propagationAxon->postNeuron->getNeuronID() == n->getNeuronID();
                                }
                                else {
                                    return false;
                                }
                            });

                            local_currentSpikes.resize(std::distance(local_currentSpikes.begin(), it));

                            if (it != currentSpikes.end()) {
                                for (auto& currentSpike: local_currentSpikes) {
                                    n->updateSync(i, currentSpike.propagationAxon, this, timestep);
                                }
                            } else {
                                n->updateSync(i, nullptr, this, timestep);
                            }
                        }

                        // checking for new spikes
                        std::vector<spike> newSpikes;
                        if (!generatedSpikes.empty()) {
                            while (!generatedSpikes.empty() && generatedSpikes.front().timestamp <= i) {
                                newSpikes.emplace_back(generatedSpikes.front());
                                generatedSpikes.pop_front();
                            }
                        }

                        for (auto& spike: newSpikes) {
                            spike.propagationAxon->postNeuron->updateSync(i, spike.propagationAxon, this, timestep);
                        }
                    }
                }
            }
            else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }
        
        // update neuron status asynchronously
        void update(spike s, bool classification=false, bool prediction=false) {
            if (!classification) {
                if (!trainingLabels.empty()) {
                    if (trainingLabels.front().onset <= s.timestamp) {
                        currentLabel = trainingLabels.front().name;
                        trainingLabels.pop_front();
                    }
                }

                if (learningOffSignal != -1) {
                    if (learningStatus==true && s.timestamp >= learningOffSignal) {
                        std::cout << "learning turned off at t=" << s.timestamp << std::endl;
                        learningStatus = false;
                    }
                }
            }
            s.propagationAxon->postNeuron->update(s.timestamp, s.propagationAxon, this, s.type);
        }
		
		// ----- IMPLEMENTATION VARIABLES -----
		std::deque<spike>                      initialSpikes;
        std::deque<spike>                      generatedSpikes;
        std::deque<spike>                      predictedSpikes;
        std::vector<AddOn*>                    addOns;
        MainThreadAddOn*                       thAddOn;
        std::vector<layer>                     layers;
		std::vector<std::unique_ptr<Neuron>>   neurons;
		std::deque<label>                      trainingLabels;
        std::vector<std::string>               uniqueLabels;
		bool                                   learningStatus;
		double                                 learningOffSignal;
        int                                    maxDelay;
        std::string                            currentLabel;
        bool                                   preTrainingLabelAssignment;
        bool                                   asynchronous;
    };
}
