/* 
 * core.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: Last Version: 28/02/2019
 *
 * Information: the core.hpp contains:
 *  - The Network class - spike manager
 *  - The Neuron class - neuron model
 */

#pragma once

#define _USE_MATH_DEFINES

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>
#include <mutex>
#include <deque>
#include <queue>

// external Dependencies
#include "../third_party/json.hpp"
#include "../third_party/sepia.hpp"

// random distributions
#include "randomDistributions/lognormal.hpp"
#include "randomDistributions/uniform.hpp"
#include "randomDistributions/normal.hpp"
#include "randomDistributions/cauchy.hpp"

// data parser
#include "dataParser.hpp"

// addons
#include "addon.hpp"
#include "mainThreadAddon.hpp"

// synapse models
#include "synapse.hpp"
#include "synapses/exponential.hpp"
#include "synapses/dirac.hpp"
#include "synapses/square.hpp"

namespace hummus {
    // make_unique creates a unique_ptr.
    template <typename T, typename... Args>
    inline std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    };
    
    // used for the event-based mode only in order to predict spike times with dynamic currents
    enum class spikeType {
        initial,
        generated,
        endOfIntegration,
        prediction,
        decision,
        none
    };
    
    // parameters for the decision-making layer
    struct decisionHeuristics {
        int                         layer_number;
        int                         spike_history_size;
        float                       rejection_threshold; // percentage (0.6 = 60% for a maximum of 1)
        double                      timer;
    };
    
    // the equivalent of feature maps
	struct sublayer {
		std::vector<std::size_t>    neurons;
		int                         ID;
	};
	
    // structure organising neurons into layers and sublayers for easier access
	struct layer {
		std::vector<sublayer>       sublayers;
        std::vector<std::size_t>    neurons;
		int                         ID;
		int                         width;
		int                         height;
        int                         kernelSize;
        int                         stride;
        bool                        do_not_propagate;
	};

    // spike - propagated between synapses
    struct spike {
        double        timestamp;
        Synapse*      propagationSynapse;
        spikeType     type;
        // provides the logic for the priority queue
        bool operator<(const spike& s) const {
            return timestamp > s.timestamp;
        }
    };
    
    // forward declaration of the Network class
	class Network;
    
    // polymorphic neuron class - the different implementations extending this class are available in the neurons folder
	class Neuron {
        
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Neuron(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, int _refractoryPeriod=3, float _conductance=200,
               float _leakageConductance=10, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70, std::string _classLabel="") :
                neuronID(_neuronID),
                layerID(_layerID),
                sublayerID(_sublayerID),
                rfCoordinates(_rfCoordinates),
                xyCoordinates(_xyCoordinates),
                threshold(_threshold), //mV
                potential(_restingPotential), //mV
                conductance(_conductance), // pF
                leakageConductance(_leakageConductance), // nS
                membraneTimeConstant(_conductance/_leakageConductance), // ms
                current(0), // pA
                refractoryPeriod(_refractoryPeriod), //ms
                restingPotential(_restingPotential), //mV
                trace(0),
                traceTimeConstant(_traceTimeConstant),
                previousSpikeTime(0),
                previousInputTime(0),
                classLabel(_classLabel),
                neuronType(0) {
                    // error handling
                    if (membraneTimeConstant <= 0) {
                        throw std::logic_error("The potential decay cannot less than or equal to 0");
                    }
                }
    	
		virtual ~Neuron(){}
		
		// ----- PUBLIC METHODS -----
		// ability to do things inside a neuron, outside the constructor before the network actually runs
		virtual void initialisation(Network* network) {}
		
		// asynchronous update method
		virtual void update(double timestamp, Synapse* s, Network* network, spikeType type) = 0;
        
		// synchronous update method
		virtual void updateSync(double timestamp, Synapse* s, Network* network, double timestep, spikeType type) {
			update(timestamp, s, network, type);
		}
        
        // reset a neuron to its initial status
        virtual void resetNeuron(Network* network, bool clearAddons=true) {
            previousSpikeTime = 0;
            potential = restingPotential;
            trace = 0;
            if (clearAddons) {
                relevantAddons.clear();
            }
        }
        
        // write neuron parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
        
        // adds a synapse that connects two Neurons together
        template <typename T, typename... Args>
        Synapse* makeSynapse(Neuron* postNeuron, int probability, float weight, float delay, Args&&... args) {
            if (postNeuron) {
                if (connectionProbability(probability)) {
                    axonTerminals.emplace_back(new T{postNeuron->neuronID, neuronID, weight, delay, static_cast<float>(std::forward<Args>(args))...});
                    postNeuron->getDendriticTree().emplace_back(axonTerminals.back().get());
                    return axonTerminals.back().get();
                } else {
                    return nullptr;
                }
            } else {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
        // initialise the initial synapse when a neuron receives an input event
        template <typename T, typename... Args>
        spike receiveExternalInput(double timestamp, Args&&... args) {
            if (!initialSynapse) {
                initialSynapse.reset(new T(std::forward<Args>(args)...));
            }
            return spike{timestamp, initialSynapse.get(), spikeType::initial};
        }
		
        // utility function that returns true or false depending on a probability percentage
        static bool connectionProbability(int probability) {
            std::random_device device;
            std::mt19937 randomEngine(device());
            std::bernoulli_distribution dist(probability/100.);
            return dist(randomEngine);
        }
        
		// ----- SETTERS AND GETTERS -----        
		int getNeuronID() const {
            return neuronID;
        }
		
        int getLayerID() const {
            return layerID;
        }
        
        int getSublayerID() const {
            return sublayerID;
        }
        
        std::pair<int, int> getRfCoordinates() const {
            return rfCoordinates;
        }
        
        void setRfCoordinates(int row, int col) {
            rfCoordinates.first = row;
            rfCoordinates.second = col;
        }
        
        std::pair<float, float> getXYCoordinates() const {
            return xyCoordinates;
        }
        
        void setXYCoordinates(float X, float Y) {
            xyCoordinates.first = X;
            xyCoordinates.second = Y;
        }
        
        std::vector<Synapse*>& getDendriticTree() {
            return dendriticTree;
        }
        
        std::vector<std::unique_ptr<Synapse>>& getAxonTerminals() {
            return axonTerminals;
        }
        
        std::unique_ptr<Synapse>& getInitialSynapse() {
            return initialSynapse;
        }
        
        float setPotential(float newPotential) {
            return potential = newPotential;
        }
        
        float getPotential() const {
            return potential;
        }
        
        float getRestingPotential() const {
            return restingPotential;
        }
        
        void setRestingPotential(float newE_l) {
            restingPotential = newE_l;
        }
        
        float getThreshold() const {
            return threshold;
        }
        
        float setThreshold(float _threshold) {
            return threshold = _threshold;
        }
        
        float getCurrent() const {
            return current;
        }
        void setCurrent(float newCurrent) {
            current = newCurrent;
        }
        
        float getTrace() const {
            return trace;
        }
        
        void setTrace(float newtrace) {
            trace = newtrace;
        }
        
        float getTraceTimeConstant() const {
            return traceTimeConstant;
        }
        
        void setTraceTimeConstant(float newConstant) {
            traceTimeConstant = newConstant;
        }
        
        double getPreviousSpikeTime() const {
            return previousSpikeTime;
        }
        
        double getPreviousInputTime() const {
            return previousInputTime;
        }
        
        int getType() const {
            return neuronType;
        }
        
        std::vector<Addon*>& getRelevantAddons() {
            return relevantAddons;
        }
        
        void addRelevantAddon(Addon* newAddon) {
            relevantAddons.emplace_back(newAddon);
        }
		
        float getConductance() const {
            return conductance;
        }
        
        void setConductance(float k) {
            conductance = k;
        }
        
        void setLeakageConductance(float k) {
            leakageConductance = k;
        }
        
        float getMembraneTimeConstant() const {
            return membraneTimeConstant;
        }
        
        void setMembraneTimeConstant(float newConstant) {
            membraneTimeConstant = newConstant;
        }
        
        void setRefractoryPeriod(float newRefractoryPeriod) {
            refractoryPeriod = newRefractoryPeriod;
        }
        
        std::deque<std::string>& getDecisionQueue() {
            return decision_queue;
        }
        
        std::string getClassLabel() const {
            return classLabel;
        }
        
        void setClassLabel(std::string newLabel) {
            classLabel = newLabel;
        }
        
    protected:
        
        // loops through any learning rules and activates them
        virtual void requestLearning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network){}
        
        // ----- NEURON SPATIAL PARAMETERS -----
        int                                        neuronID;
        int                                        layerID;
        int                                        sublayerID;
        std::pair<int, int>                        rfCoordinates;
        std::pair<float, float>                    xyCoordinates;
        
        // ----- SYNAPSES OF THE NEURON -----
        std::vector<Synapse*>                      dendriticTree;
        std::vector<std::unique_ptr<Synapse>>      axonTerminals;
        std::unique_ptr<Synapse>                   initialSynapse;
        
        // ----- DYNAMIC VARIABLES -----
        float                                      current;
        float                                      potential;
        float                                      trace;
        
        // ----- FIXED PARAMETERS -----
        float                                      threshold;
        float                                      restingPotential;
        float                                      traceTimeConstant;
        float                                      conductance;
        float                                      leakageConductance;
        float                                      membraneTimeConstant;
        float                                      refractoryPeriod;
        
        // ----- IMPLEMENTATION PARAMETERS -----
        std::vector<Addon*>                        relevantAddons;
        double                                     previousSpikeTime;
        double                                     previousInputTime;
        int                                        neuronType;
        std::deque<std::string>                    decision_queue;
        std::string                                classLabel;
    };
	
    class Network {
        
    public:
		
		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network() :
                learningStatus(true),
                asynchronous(false),
                decision_making(false),
                learningOffSignal(-1),
                verbose(0),
                decision_pre_ts(0),
                maxDelay(0) {
                    // seeding and initialising random engine with a Mersenne Twister pseudo-random generator
                    std::random_device device;
                    randomEngine = std::mt19937(device());
                }
		
        // ----- NETWORK IMPORT EXPORT METHODS -----
        
        // exporting the network into a JSON file
        void save(std::string filename) {
            nlohmann::json output = nlohmann::json::array();
            
            // initialising a JSON list
            output.push_back({{"layers", nlohmann::json::array()}, {"neurons", nlohmann::json::array()}});
            auto& jsonNetwork = output.back();
            
            // saving the important information needed from the layers
            for (auto l: layers) {
                jsonNetwork["layers"].push_back({
                    {"width", l.width},
                    {"height",l.height},
                    {"sublayerNumber",l.sublayers.size()},
                    {"neuronNumber",l.neurons.size()},
                    {"neuronType",neurons[l.neurons[0]]->getType()},
                });
            }
			
            // saving the important information needed from the neurons
            for (auto& n: neurons) {
                n->toJson(jsonNetwork["neurons"]);
            }
            
            std::ofstream output_file(filename.append(".json"));
            output_file << output.dump(4);
        }
        
		// ----- NEURON CREATION METHODS -----
		
        // adds one dimensional neurons
        template <typename T, typename... Args>
        layer makeLayer(int _numberOfNeurons, std::vector<Addon*> _addons, Args&&... args) {
            
            if (_numberOfNeurons < 0) {
                throw std::logic_error("the number of neurons selected is wrong");
            }
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layerID = 0;
            if (!layers.empty()) {
				
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
				
                layerID = layers.back().ID+1;
            }

            // building a layer of one dimensional sublayers
            std::vector<std::size_t> neuronsInLayer;
            for (auto k=0+shift; k<_numberOfNeurons+shift; k++) {
                neurons.emplace_back(make_unique<T>(static_cast<int>(k), layerID, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), std::forward<Args>(args)...));
                neuronsInLayer.emplace_back(neurons.size()-1);
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // building layer structure
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1, -1, -1, false});
            return layers.back();
        }
        
        // takes in training labels and creates DecisionMaking neurons according to the number of classes present - Decision layer should be the last layer
        template <typename T, typename... Args>
        layer makeDecision(std::string trainingLabelFilename, int _spike_history_size, float _rejection_threshold, double _timer, std::vector<Addon*> _addons, Args&&... args) {
            DataParser dataParser;
            trainingLabels = dataParser.readLabels(trainingLabelFilename);
            decision_making = true;
            // do not let the last layer propagate to the decision-making layer during the training phase
            layers.back().do_not_propagate = true;
            
            // add the unique classes to the classes_map
            for (auto& label: trainingLabels) {
                classes_map.insert({label.name, 0});
            }
            
            int shift = 0;
            
            // find the layer ID
            int layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += static_cast<int>(l.neurons.size());
                }
                layerID = layers.back().ID+1;
            } else {
                throw std::logic_error("the decision layer can only be on the last layer");
            }
            
            // add decision-making neurons
            std::vector<std::size_t> neuronsInLayer;
            
            int i=0;
            for (auto it: classes_map) {
                neurons.emplace_back(make_unique<T>(i+shift, layerID, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), it.first, std::forward<Args>(args)...));
                neuronsInLayer.emplace_back(neurons.size()-1);
                ++i;
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // saving the decision parameters
            decision.layer_number = layerID;
            decision.spike_history_size = _spike_history_size;
            decision.rejection_threshold = _rejection_threshold;
            decision.timer = _timer;
            
            // building layer structure
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1, -1, -1, false});
            return layers.back();
        }
        
        // adds neurons arranged in circles of various radii
        template <typename T, typename... Args>
        layer makeCircle(int _numberOfNeurons, std::vector<float> _radii, std::vector<Addon*> _addons, Args&&... args) {
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // building a layer of two dimensional sublayers
            int counter = 0;
            std::vector<sublayer> sublayers;
            std::vector<std::size_t> neuronsInLayer;
            for (int i=0; i<_radii.size(); i++) {
                std::vector<std::size_t> neuronsInSublayer;
                for (auto k=0+shift; k<_numberOfNeurons+shift; k++) {
                    float u = _radii[i] * std::cos(2*M_PI*(k-shift)/_numberOfNeurons);
                    float v = _radii[i] * std::sin(2*M_PI*(k-shift)/_numberOfNeurons);
                    neurons.emplace_back(make_unique<T>(static_cast<int>(k)+counter, layerID, i, std::pair<int, int>(0, 0), std::pair<float, float>(u, v), std::forward<Args>(args)...));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);
                }
                sublayers.emplace_back(sublayer{neuronsInSublayer, i});
                
                // to shift the neuron IDs with the sublayers
                counter += _numberOfNeurons;
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // building layer structure
            layers.emplace_back(layer{sublayers, neuronsInLayer, layerID, -1, -1, -1, -1, false});
            return layers.back();
        }
        
        // adds a 2 dimensional grid of neurons
        template <typename T, typename... Args>
        layer makeGrid(int gridW, int gridH, int _sublayerNumber, std::vector<Addon*> _addons, Args&&... args) {
            
            // find number of neurons to build
            int numberOfNeurons = gridW * gridH;
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // building a layer of two dimensional sublayers
            int counter = 0;
            std::vector<sublayer> sublayers;
            std::vector<std::size_t> neuronsInLayer;
            for (int i=0; i<_sublayerNumber; i++) {
                std::vector<std::size_t> neuronsInSublayer;
                int x = 0; int y = 0;
                for (auto k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(make_unique<T>(static_cast<int>(k)+counter, layerID, i, std::pair<int, int>(0, 0), std::pair<int, int>(x, y), std::forward<Args>(args)...));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);
                    
                    x += 1;
                    if (x == gridW) {
                        y += 1;
                        x = 0;
                    }
                }
                sublayers.emplace_back(sublayer{neuronsInSublayer, i});
                
                // to shift the neuron IDs with the sublayers
                counter += numberOfNeurons;
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // building layer structure
            layers.emplace_back(layer{sublayers, neuronsInLayer, layerID, gridW, gridH, -1, -1, false});
            return layers.back();
        }
        
        // overloading the makeGrid function to automatically generate a 2D layer according to the previous layer size
        template <typename T, typename... Args>
        layer makeGrid(layer presynapticLayer, int _sublayerNumber, int _kernelSize, int _stride, std::vector<Addon*> _addons, Args&&... args) {
            // finding the number of receptive fields
            int newWidth = std::ceil((presynapticLayer.width - _kernelSize + 1) / static_cast<float>(_stride));
            int newHeight = std::ceil((presynapticLayer.height - _kernelSize + 1) / static_cast<float>(_stride));
            
            int trimmedColumns = std::abs(newWidth - std::ceil((presynapticLayer.width - _stride + 1) / static_cast<float>(_stride)));
            int trimmedRows = std::abs(newHeight - std::ceil((presynapticLayer.height - _stride + 1) / static_cast<float>(_stride)));
            
            // warning message that some rows and columns of neurons might be ignored and left unconnected depending on the stride value
            if (verbose != 0) {
                if (trimmedColumns > 0 && trimmedRows == 0) {
                    std::cout << "The new layer did not take into consideration the last " << trimmedColumns << " column(s) of presynaptic neurons because the stride brings the sliding window outside the presynaptic layer dimensions" << std::endl;
                } else if (trimmedRows > 0 && trimmedColumns == 0) {
                    std::cout << "The new layer did not take into consideration the last " << trimmedRows << " row(s) of presynaptic neurons because the stride brings the sliding window outside the presynaptic layer dimensions" << std::endl;
                } else if (trimmedRows > 0 && trimmedColumns > 0){
                    std::cout << "The new layer did not take into consideration the last " << trimmedColumns << " column(s) and the last " << trimmedRows << " row(s) of presynaptic neurons because the stride brings the sliding window outside the presynaptic layer dimensions" << std::endl;
                }
            }
            
            // find number of neurons to build
            int numberOfNeurons = newWidth * newHeight;
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // building a layer of two dimensional sublayers
            int counter = 0;
            std::vector<sublayer> sublayers;
            std::vector<std::size_t> neuronsInLayer;
            for (int i=0; i<_sublayerNumber; i++) {
                std::vector<std::size_t> neuronsInSublayer;
                int x = 0; int y = 0;
                for (auto k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(make_unique<T>(static_cast<int>(k)+counter, layerID, i, std::pair<int, int>(0, 0), std::pair<int, int>(x, y), std::forward<Args>(args)...));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);
                    
                    x += 1;
                    if (x == newWidth) {
                        y += 1;
                        x = 0;
                    }
                }
                sublayers.emplace_back(sublayer{neuronsInSublayer, i});
                
                // to shift the neuron IDs with the sublayers
                counter += numberOfNeurons;
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // building layer structure
            layers.emplace_back(layer{sublayers, neuronsInLayer, layerID, newWidth, newHeight, _kernelSize, _stride, false});
            return layers.back();
        }
        
        // creates a layer that is a subsampled version of the previous layer, to the nearest divisible grid size
        template <typename T, typename... Args>
        layer makeSubsampledGrid(layer presynapticLayer, std::vector<Addon*> _addons, Args&&... args) {
            // find lowest common divisor
            int lcd = 1;
            for (auto i = 2; i <= presynapticLayer.width && i <= presynapticLayer.height; i++) {
                if (presynapticLayer.width % i == 0 && presynapticLayer.height % i == 0) {
                    lcd = i;
                    break;
                }
            }
            
            if (lcd == 1) {
                throw std::logic_error("The pooling cannot find a common divisor that's different than 1 for the size of the previous layer.");
            }
            
            if (verbose != 0) {
                std::cout << "subsampling by a factor of " << lcd << std::endl;
            }
            
            return makeGrid<T>(presynapticLayer.width/lcd, presynapticLayer.height/lcd, static_cast<int>(presynapticLayer.sublayers.size()), _addons, std::forward<Args>(args)...);
        }
        
		// ----- LAYER CONNECTION METHODS -----
        
        // connecting a layer that is a convolution of the previous layer, depending on the layer kernel size and the stride. Last set of paramaters are to characterize the synapses. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays. Furthermore, you can select the number of synapses per pair of presynaptic and postsynaptic neurons (the arborescence)
        template <typename T, typename F, typename... Args>
        void convolution(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            // error handling
            if (postsynapticLayer.kernelSize == -1 || postsynapticLayer.stride == -1) {
                throw std::logic_error("cannot connect the layers in a convolutional manner as the layers were not built with that in mind (no kernel or stride in the grid layer to define receptive fields");
            }
            
            // find how many neurons there are before the pre and postsynaptic layers
            int layershift = 0;
            if (!layers.empty()) {
                for (auto i=0; i<presynapticLayer.ID; i++) {
                    layershift += layers[i].neurons.size();
                }
            }
            
            int trimmedColumns = std::abs(postsynapticLayer.width - std::ceil((presynapticLayer.width - postsynapticLayer.stride + 1) / static_cast<float>(postsynapticLayer.stride)));
            
            // finding range to calculate a moore neighborhood
            float range;
            if (postsynapticLayer.kernelSize % 2 == 0) {
                range = postsynapticLayer.kernelSize - std::ceil(postsynapticLayer.kernelSize / static_cast<float>(2)) - 0.5;
            }
            else {
                range = postsynapticLayer.kernelSize - std::ceil(postsynapticLayer.kernelSize / static_cast<float>(2));
            }
            
            // number of neurons surrounding the center
            int mooreNeighbors = (2*range + 1) * (2*range + 1);
            
            // looping through the newly created layer to connect them to the correct receptive fields
            for (auto& convSub: postsynapticLayer.sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    
                    // initialising window on the first center coordinates
                    std::pair<float, float> centerCoordinates((postsynapticLayer.kernelSize-1)/static_cast<float>(2), (postsynapticLayer.kernelSize-1)/static_cast<float>(2));
                    
                    // number of neurons = number of receptive fields in the presynaptic Layer
                    int row = 0; int col = 0;
                    for (auto& n: convSub.neurons) {
                        
                        // finding the coordinates for the presynaptic neurons in each receptive field
                        for (auto i=0; i<mooreNeighbors; i++) {
                            int x = centerCoordinates.first + ((i % postsynapticLayer.kernelSize) - range);
                            int y = centerCoordinates.second + ((i / postsynapticLayer.kernelSize) - range);
                            
                            // 2D to 1D mapping to get the index from x y coordinates
                            int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;
                            
                            // changing the neuron's receptive field coordinates from the default
                            neurons[idx]->setRfCoordinates(row, col);
                            
                            // connecting neurons from the presynaptic layer to the convolutional one, depedning on the number of synapses
                            for (auto i=0; i<number_of_synapses; i++) {
                                // calculating weights and delays according to the provided distribution
                                const std::pair<float, float> weight_delay = lambdaFunction(x, y, convSub.ID);
                                
                                // creating a synapse between the neurons
                                neurons[idx]->makeSynapse<T>(neurons[n].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);
                                
                                // to shift the network runtime by the maximum delay in the clock mode
                                maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                            }
                        }
                        
                        // finding the coordinates for the center of each receptive field
                        centerCoordinates.first += postsynapticLayer.stride;
                        if (centerCoordinates.first >= presynapticLayer.width - trimmedColumns) {
                            centerCoordinates.first = (postsynapticLayer.kernelSize-1)/static_cast<float>(2);
                            centerCoordinates.second += postsynapticLayer.stride;
                        }
                        
                        // updating receptive field indices
                        row += 1;
                        if (row == postsynapticLayer.width) {
                            col += 1;
                            row = 0;
                        }
                    }
                    sublayershift += preSub.neurons.size();
                }
            }
        }
        
        // connecting a subsampled layer to its previous layer. Last set of paramaters are to characterize the synapses. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T, typename F, typename... Args>
        void pooling(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            // error handling
            if (postsynapticLayer.ID - presynapticLayer.ID > 1) {
                throw std::logic_error("the layers aren't immediately following each other");
            }
            
            // find how many neurons there are before the pre and postsynaptic layers
            int layershift = 0;
            if (!layers.empty()) {
                for (auto i=0; i<presynapticLayer.ID; i++) {
                    layershift += layers[i].neurons.size();
                }
            }
            
            float range;
            int lcd = presynapticLayer.width / postsynapticLayer.width;
            
            // if size of kernel is an even number
            if (lcd % 2 == 0) {
                range = lcd - std::ceil(lcd / static_cast<float>(2)) - 0.5;
                // if size of kernel is an odd number
            } else {
                // finding range to calculate a moore neighborhood
                range = lcd - std::ceil(lcd / static_cast<float>(2));
            }
            
            // number of neurons surrounding the center
            int mooreNeighbors = (2*range + 1) * (2*range + 1);
            
            for (auto& poolSub: postsynapticLayer.sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    if (poolSub.ID == preSub.ID) {
                        
                        // initialising window on the first center coordinates
                        std::pair<float, float> centerCoordinates((lcd-1)/static_cast<float>(2), (lcd-1)/static_cast<float>(2));
                        
                        // number of neurons = number of receptive fields in the presynaptic Layer
                        int row = 0; int col = 0;
                        for (auto& n: poolSub.neurons) {
                            
                            // finding the coordinates for the presynaptic neurons in each receptive field
                            for (auto i=0; i<mooreNeighbors; i++) {
                                
                                int x = centerCoordinates.first + ((i % lcd) - range);
                                int y = centerCoordinates.second + ((i / lcd) - range);
                                
                                // 2D to 1D mapping to get the index from x y coordinates
                                int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;
                                
                                // changing the neuron's receptive field coordinates from the default
                                neurons[idx]->setRfCoordinates(row, col);
                                
                                for (auto i=0; i<number_of_synapses; i++) {
                                    // calculating weights and delays according to the provided distribution
                                    const std::pair<float, float> weight_delay = lambdaFunction(x, y, poolSub.ID);
                                    
                                    // connecting neurons from the presynaptic layer to the convolutional one
                                    neurons[idx]->makeSynapse<T>(neurons[n].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);
                                    
                                    // to shift the network runtime by the maximum delay in the clock mode
                                    maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                                }
                            }
                            
                            // finding the coordinates for the center of each receptive field
                            centerCoordinates.first += lcd;
                            if (centerCoordinates.first >= presynapticLayer.width) {
                                centerCoordinates.first = (lcd-1)/static_cast<float>(2);
                                centerCoordinates.second += lcd;
                            }
                            
                            // updating receptive field indices
                            row += 1;
                            if (row == presynapticLayer.width/lcd) {
                                col += 1;
                                row = 0;
                            }
                        }
                    }
                    sublayershift += preSub.neurons.size();
                }
            }
        }
        
        // interconnecting a layer (feedforward, feedback and self-excitation) with randomised weights and delays. lambdaFunction: Takes in one of the classes inside the randomDistributions folder to define a distribution for the weights.
        template <typename T, typename F, typename... Args>
        void reservoir(layer reservoirLayer, int number_of_synapses, F&& lambdaFunction, int feedforwardProbability, int feedbackProbability, int selfExcitationProbability, Args&&... args) {
            // connecting the reservoir
            for (auto pre: reservoirLayer.neurons) {
                for (auto post: reservoirLayer.neurons) {
                    for (auto i=0; i<number_of_synapses; i++) {
                        // calculating weights and delays according to the provided distribution
                        const std::pair<float, float> weight_delay = lambdaFunction(0, 0, 0);
                        
                        // self-excitation probability
                        if (pre == post) {
                            neurons[pre]->makeSynapse<T>(neurons[post].get(), selfExcitationProbability, weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                        } else {
                            // feedforward probability
                            neurons[pre]->makeSynapse<T>(neurons[post].get(), feedforwardProbability, weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                            
                            // feedback probability
                            neurons[post]->makeSynapse<T>(neurons[pre].get(), feedbackProbability, weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                        }
                    }
                }
            }
        }
        
		// connecting two layers according to a weight matrix vector of vectors and a delays matrix vector of vectors (columns for input and rows for output)
        template <typename T, typename... Args>
        void connectivityMatrix(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, std::vector<std::vector<float>> weights, std::vector<std::vector<float>> delays, Args&&... args) {
            
            // error handling
            if (weights.size() != delays.size() && weights[0].size() != delays[0].size()) {
                throw std::logic_error("the weight matrix and delay matrix do not have the same dimensions");
            }
            
            if (postsynapticLayer.neurons.size() != weights[0].size()) {
                throw std::logic_error("the postsynaptic layer doesn't contain the same number of neurons as represented in the matrix");
            }
            
            if (presynapticLayer.neurons.size() != weights.size()) {
                throw std::logic_error("the presynaptic layer doesn't contain the same number of neurons as represented in the matrix");
            }
            
            int preCounter = 0;
            int postCounter = 0;
            for (auto& preSub: presynapticLayer.sublayers) {
                for (auto& preNeuron: preSub.neurons) {
                    for (auto& postSub: postsynapticLayer.sublayers) {
                        for (auto& postNeuron: postSub.neurons) {
                            
                            if (weights[preCounter][postCounter] != 0) {
                                for (auto i=0; i<number_of_synapses; i++) {
                                    neurons[preNeuron]->makeSynapse<T>(neurons[postNeuron].get(), 100, weights[preCounter][postCounter], delays[preCounter][postCounter], std::forward<Args>(args)...);
                                }
                            }
                            
                            // to shift the network runtime by the maximum delay in the clock mode
                            maxDelay = std::max(static_cast<float>(maxDelay), delays[preCounter][postCounter]);
                            
                            // looping through the rows of the weight matrix
                            postCounter += 1;
                        }
                    }
                    preCounter += 1;
                    postCounter = 0;
                }
            }
        }
        
        // one to one connections between layers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T, typename F, typename... Args>
        void oneToOne(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            // error handling
            if (presynapticLayer.neurons.size() != postsynapticLayer.neurons.size() && presynapticLayer.width == postsynapticLayer.width && presynapticLayer.height == postsynapticLayer.height) {
                throw std::logic_error("The presynaptic and postsynaptic layers do not have the same number of neurons. Cannot do a one-to-one connection");
            }
            
            for (auto preSubIdx=0; preSubIdx<presynapticLayer.sublayers.size(); preSubIdx++) {
                for (auto preNeuronIdx=0; preNeuronIdx<presynapticLayer.sublayers[preSubIdx].neurons.size(); preNeuronIdx++) {
                    for (auto postSubIdx=0; postSubIdx<postsynapticLayer.sublayers.size(); postSubIdx++) {
                        for (auto postNeuronIdx=0; postNeuronIdx<postsynapticLayer.sublayers[postSubIdx].neurons.size(); postNeuronIdx++) {
                            if (preNeuronIdx == postNeuronIdx) {
                                for (auto i=0; i<number_of_synapses; i++) {
                                    const std::pair<float, float> weight_delay = lambdaFunction(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->getXYCoordinates().first, neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->getXYCoordinates().second, postsynapticLayer.sublayers[postSubIdx].ID);
                                    neurons[presynapticLayer.sublayers[preSubIdx].neurons[preNeuronIdx]]->makeSynapse<T>(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);

                                    // to shift the network runtime by the maximum delay in the clock mode
                                    maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // all to all connection between layers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T, typename F, typename... Args>
        void allToAll(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            for (auto& preSub: presynapticLayer.sublayers) {
                for (auto& preNeuron: preSub.neurons) {
                    for (auto& postSub: postsynapticLayer.sublayers) {
                        for (auto& postNeuron: postSub.neurons) {
                            for (auto i=0; i<number_of_synapses; i++) {
                                const std::pair<float, float> weight_delay = lambdaFunction(neurons[postNeuron]->getXYCoordinates().first, neurons[postNeuron]->getXYCoordinates().second, postSub.ID);
                                neurons[preNeuron]->makeSynapse<T>(neurons[postNeuron].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);
                                
                                // to shift the network runtime by the maximum delay in the clock mode
                                maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                            }
                        }
                    }
                }
            }
        }

        // interconnecting a layer with soft winner-takes-all synapses, using negative weights
        template <typename T, typename F, typename... Args>
        void lateralInhibition(layer l, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            for (auto& sub: l.sublayers) {
                // intra-sublayer soft WTA
                for (auto& preNeurons: sub.neurons) {
                    for (auto& postNeurons: sub.neurons) {
                        if (preNeurons != postNeurons) {
                            for (auto i=0; i<number_of_synapses; i++) {
                                const std::pair<float, float> weight_delay = lambdaFunction(0, 0, 0);
                                neurons[preNeurons]->makeSynapse<T>(neurons[postNeurons].get(), probability, -1*std::abs(weight_delay.first), weight_delay.second, std::forward<Args>(args)...);
                            }
                        }
                    }
                }
                
                // inter-sublayer soft WTA
                for (auto& subToInhibit: l.sublayers) {
                    if (sub.ID != subToInhibit.ID) {
                        for (auto& preNeurons: sub.neurons) {
                            for (auto& postNeurons: subToInhibit.neurons) {
                                if (neurons[preNeurons]->getRfCoordinates() == neurons[postNeurons]->getRfCoordinates()) {
                                    for (auto i=0; i<number_of_synapses; i++) {
                                        const std::pair<float, float> weight_delay = lambdaFunction(0, 0, 0);
                                        neurons[preNeurons]->makeSynapse<T>(neurons[postNeurons].get(), probability, -1*std::abs(weight_delay.first), weight_delay.second, std::forward<Args>(args)...);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // ----- PUBLIC NETWORK METHODS -----
        // adds a spike to the priority queue
        void injectSpike(spike s) {
            spike_queue.emplace(s);
        }
        
        // overloaded method - creates a spike and adds it to the spike_queue priority queue
        void injectSpike(int neuronIndex, double timestamp) {
            spike_queue.emplace(neurons[neuronIndex]->receiveExternalInput<Dirac>(timestamp, neuronIndex, -1, 1, 0));
        }
        
        
        // adding spikes predicted by the asynchronous network (timestep = 0) for synaptic integration
        void injectPredictedSpike(spike s, spikeType stype) {
            // remove old spike
            std::remove_if(predictedSpikes.begin(), predictedSpikes.end(),[&](spike oldSpike) {
                return oldSpike.propagationSynapse == s.propagationSynapse;
            });
            
            // change type of new spike
            s.type = stype;
            
            // insert the new spike in the correct place
            predictedSpikes.insert(
                std::upper_bound(predictedSpikes.begin(), predictedSpikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }
        
        // add spikes from file to the network
        void injectSpikeFromData(std::vector<input>* data) {
            // error handling
            if (neurons.empty()) {
                throw std::logic_error("add neurons before injecting spikes");
            }
            
            for (auto& event: *data) {
                for (auto& n: layers[0].neurons) {
                    // one dimensional data
                    if (event.x == -1) {
                        if (neurons[n]->getNeuronID() == event.neuronID) {
                            injectSpike(static_cast<int>(n), event.timestamp);
                            break;
                        }
                    // two dimensional data (with or without polarity is the same because polarity is unused)
                    } else {
                        if (neurons[n]->getXYCoordinates().first == event.x && neurons[n]->getXYCoordinates().second == event.y) {
                            injectSpike(static_cast<int>(n), event.timestamp);
                            break;
                        }
                    }
                }
            }
        }
        
        // add a poissonian spike train to the initial spike vector
        void poissonSpikeGenerator(int neuronIndex, double timestamp, float rate, float timestep, float duration) {
            // calculating number of spikes
            int spike_number = std::floor(duration/timestep);
            
            // initialising the random engine
            std::random_device                     device;
            std::mt19937                           randomEngine(device());
            std::uniform_real_distribution<double> distribution(0.0,1.0);
            
            std::vector<double> inter_spike_intervals;
            // generating uniformly distributed random numbers
            for (auto i = 0; i < spike_number; ++i) {
                inter_spike_intervals.emplace_back((- std::log(distribution(randomEngine)) / rate) * 1000);
            }
            
            // computing spike times from inter-spike intervals
            std::vector<double> spike_times(inter_spike_intervals.size(), 0.0);
            spike_times[0] = inter_spike_intervals[0];
            for (auto i=1; i<spike_times.size(); i++) {
                spike_times[i] = spike_times[i-1] + inter_spike_intervals[i];
            }
            std::transform(spike_times.begin(), spike_times.end(), spike_times.begin(), [&](double& st){return st*0.001+timestamp;});
            
            // injecting into the initial spike vector
            for (auto& spike_time: spike_times) {
                spike_queue.emplace(neurons[neuronIndex]->receiveExternalInput<Dirac>(spike_time, neuronIndex, -1, 1, 0));
            }
        }
        
        // turn off learning
        void turnOffLearning() {
            learningStatus = false;
        }
        
        // overloaded function - turn off learning at a specified timestamp
        void turnOffLearning(double timestamp) {
            learningOffSignal = timestamp;
        }
        
        // running through the network asynchronously if timestep = 0 and synchronously otherwise
        void run(double _runtime, double _timestep=0, bool classification=false) {
            // error handling
            if (_timestep < 0) {
                throw std::logic_error("the timestep cannot be negative");
            } else if (_timestep == 0) {
                if (verbose != 0) {
                    std::cout << "Running the network asynchronously" << std::endl;
                }
            } else {
                if (verbose != 0) {
                    std::cout << "Running the network synchronously" << std::endl;
                }
            }
            
            if (_timestep == 0) {
                asynchronous = true;
            }
            
            for (auto& n: neurons) {
                n->initialisation(this);
            }

            for (auto& addon: addons) {
                addon->onStart(this);
            }
			
            if (classification) {
                std::cout << "classification is true. This instance will assume a training run has already been done, and will try to initialise the Decision-Making if it was initialised)" << std::endl;
                
                if (thAddon) {
                    thAddon->reset();
                }
                
                // can now propagate through all layers in case a decision-making layer is present
                for (auto& layer: layers) {
                    layer.do_not_propagate = false;
                }
                
                // during a classification run, labels the neurons if a decision-making layer was used
                prepareDecisionMaking();
            }
                
            std::mutex sync;
            if (thAddon) {
                sync.lock();
            }

                std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();
 
                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                    
                if (_timestep == 0) {
                    eventRunHelper(_runtime, _timestep, classification);
                } else {
                    clockRunHelper(_runtime, _timestep, classification);
                }
                
                std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
                    
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
                }
                    
                for (auto& addon: addons) {
                    addon->onCompleted(this);
                }
            });

            if (thAddon) {
                thAddon->begin(this, &sync);
            }
            
            spikeManager.join();
            
            // resetting network and clearing addons initialised for this particular run
            reset();
        }

        // running through the network asynchronously if timestep = 0 and synchronously otherwise. This overloaded method takes in a training and an optional testing dataset instead of a runtime
        void run(std::vector<input>* trainingData, float _timestep=0, std::vector<input>* testData=nullptr) {
            // the shift variable adds some time to the runtime - to fully visualise current dynamics
            int shift = 20;
            
            if (_timestep == 0) {
                asynchronous = true;
            }
			
            for (auto& n: neurons) {
                n->initialisation(this);
            }
			
            if (learningOffSignal == -1) {
                learningOffSignal = trainingData->back().timestamp+maxDelay+shift;
            }
			
            for (auto& addon: addons) {
                addon->onStart(this);
            }

            std::mutex sync;
            if (thAddon) {
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

                for (auto& addon: addons) {
                    addon->onCompleted(this);
                }
            });

            if (thAddon) {
                thAddon->begin(this, &sync);
            }
            
            spikeManager.join();
            
            // resetting network and clearing addons initialised for this particular run
            reset();
        }

//        // runs through a dataset pattern by pattern. The time is reset after each presentation
//        void run(std::string training_directory, float timestep=0, std::string test_directory="") {
//            // figuring out whether the network is running synchronously or asynchronously
//            if (timestep == 0) {
//                asynchronous = true;
//            }
//
//            // initialising the neurons
//            for (auto& n: neurons) {
//                n->initialisation(this);
//            }
//
//            // initialising the addons
//            for (auto& addon: addons) {
//                addon->onStart(this);
//            }
//
//            // initialising the GUI if available
//            std::mutex sync;
//            if (thAddon) {
//                sync.lock();
//            }
//
//            // running network in separate thread, and GUI in main thread
//            std::thread spikeManager([&] {
//                sync.lock();
//                sync.unlock();
//
//                // walking through training dataset
//
//                // walking through test dataset
//
//                // notifying addons the run is complete
//                for (auto& addon: addons) {
//                    addon->onCompleted(this);
//                }
//            });
//
//            if (thAddon) {
//                thAddon->begin(this, &sync);
//            }
//
//            spikeManager.join();
//
//            // resetting network and clearing addons initialised for this particular run
//            reset();
//        }
        
        // reset the network back to the initial conditions without changing the network build
        void reset() {
            learningStatus = true;
            learningOffSignal = -1;
            
            for (auto& n: neurons) {
                n->resetNeuron(this);
            }
        }
        
        // initialises an addon that needs to run on the main thread
        template <typename T, typename... Args>
        T& makeGUI(Args&&... args) {
            thAddon.reset(new T(std::forward<Args>(args)...));
            return static_cast<T&>(*thAddon);
        }
        
        // initialises an addon and adds it to the addons vector. returns a reference to the add-on
        template <typename T, typename... Args>
        T& makeAddon(Args&&... args) {
            addons.emplace_back(new T(std::forward<Args>(args)...));
            return static_cast<T&>(*addons.back());
        }
        
        // ----- SETTERS AND GETTERS -----
		
        std::vector<std::unique_ptr<Neuron>>& getNeurons() {
            return neurons;
        }

        std::vector<layer>& getLayers() {
            return layers;
        }
        
        std::deque<std::unique_ptr<Addon>>& getAddons() {
            return addons;
        }
        
        std::unique_ptr<MainThreadAddon>& getMainThreadAddon() {
            return thAddon;
        }

        void setMainThreadAddon(MainThreadAddon* newThAddon) {
            thAddon.reset(newThAddon);
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
        
        bool getDecisionMaking() {
            return decision_making;
        }
        
        bool getNetworkType() const {
            return asynchronous;
        }
        
        int getVerbose() const {
            return verbose;
        }
		
        decisionHeuristics& getDecisionParameters() {
            return decision;
        }
        
        // verbose argument (0 for no couts at all, 1 for network-related print-outs and learning rule print-outs, 2 for network and neuron-related print-outs
        void verbosity(int value) {
            if (value >= 0 && value <= 2) {
                verbose = value;
            } else {
                throw std::logic_error("the verbose argument shoud be set to 0 to remove all print-outs, 1 to get network-related print-outs and 2 for network and neuron-related print-outs");
            }
        }
        
    protected:

        // -----PROTECTED NETWORK METHODS -----
        
        // importing training data and running the network through the data
        void train(double timestep, std::vector<input>* trainingData, int shift) {
            injectSpikeFromData(trainingData);
            
            std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
            if (verbose != 0) {
                std::cout << "Training the network..." << std::endl;
            }
            
            if (timestep == 0) {
                eventRunHelper(trainingData->back().timestamp+maxDelay+shift, timestep, false);
            } else {
                clockRunHelper(trainingData->back().timestamp+maxDelay+shift, timestep, false);
            }
            
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
            if (verbose != 0) {
                std::cout << "it took " << elapsed_seconds.count() << "s for the training phase." << std::endl;
            }
        }

        // importing test data and running it through the network for classification
        void predict(double timestep, std::vector<input>* testData, int shift) {
            
            // can now propagate through all layers in case a decision-making layer is present
            for (auto& layer: layers) {
                layer.do_not_propagate = false;
            }
            
            learningStatus = false;
            for (auto& n: neurons) {
                n->resetNeuron(this, false);
            }

            prepareDecisionMaking();
            
            injectSpikeFromData(testData);
			
            for (auto& addon: addons) {
				addon->onPredict(this);
			}
			
            if (thAddon) {
                thAddon->reset();
            }
            
            if (verbose != 0) {
                std::cout << "Running classification based on a trained network..." << std::endl;
            }
            
            if (timestep == 0) {
                eventRunHelper(testData->back().timestamp+maxDelay+shift, timestep, true);
            } else {
                clockRunHelper(testData->back().timestamp+maxDelay+shift, timestep, true);
            }
            
            if (verbose != 0) {
                std::cout << "Done." << std::endl;
            }
        }
        
        
        // helper function that runs the network when event-mode is selected
        void eventRunHelper(double runtime, double timestep, bool classification=false) {
            if (!neurons.empty()) {
                while (!spike_queue.empty() || !predictedSpikes.empty()) {
                    if (!spike_queue.empty() && predictedSpikes.empty()) {
                        requestUpdate(spike_queue.top(), classification);
                        spike_queue.pop();
                    } else if (!predictedSpikes.empty() && spike_queue.empty()) {
                        requestUpdate(predictedSpikes.front(), classification);
                        predictedSpikes.pop_front();
                    } else if (!predictedSpikes.empty() && !spike_queue.empty()) {
                        if (spike_queue.top().timestamp < predictedSpikes.front().timestamp) {
                            requestUpdate(spike_queue.top(), classification);
                            spike_queue.pop();
                        } else if (predictedSpikes.front().timestamp < spike_queue.top().timestamp) {
                            requestUpdate(predictedSpikes.front(), classification);
                            predictedSpikes.pop_front();
                        } else if (predictedSpikes.front().timestamp == spike_queue.top().timestamp) {
                            requestUpdate(spike_queue.top(), classification);
                            spike_queue.pop();
                        }
                    }
                }
            } else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }
        
        // update neuron status asynchronously
        void requestUpdate(spike s, bool classification=false) {
            if (!classification) {
                if (!trainingLabels.empty()) {
                    if (trainingLabels.front().onset <= s.timestamp) {
                        currentLabel = trainingLabels.front().name;
                        trainingLabels.pop_front();
                    }
                }
                
                if (learningOffSignal != -1) {
                    if (learningStatus==true && s.timestamp >= learningOffSignal) {
                        if (verbose != 0) {
                            std::cout << "learning turned off at t=" << s.timestamp << std::endl;
                        }
                        learningStatus = false;
                    }
                }
            } else {
                if (decision_making) {
                    if (s.timestamp - decision_pre_ts >= decision.timer) {
                        // make all the decision neurons fire
                        for (auto& n: layers[decision.layer_number].neurons) {
                            // generate a decision spike if the neuron is connected to anything
                            if (!neurons[n]->getDendriticTree().empty()) {
                                neurons[n]->update(s.timestamp, nullptr, this, spikeType::decision);
                            } else {
                                if (verbose == 2) {
                                    std::cout << "No neurons have specialised for the decision neuron with the label " << neurons[n]->getClassLabel() << std::endl;
                                }
                            }
                        }
                        // saving previous timestamp
                        decision_pre_ts = s.timestamp;
                    }
                }
            }
            neurons[s.propagationSynapse->getPostsynapticNeuronID()]->update(s.timestamp, s.propagationSynapse, this, s.type);
        }
        
        // helper function that runs the network when clock-mode is selected
        void clockRunHelper(double runtime, double timestep, bool classification=false) {
            if (!neurons.empty()) {
                
                // creating vector of the same size as neurons
                std::vector<bool> neuronStatus(neurons.size(), false);
                
                // loop over the full runtime
                for (double i=0; i<runtime; i+=timestep) {
                    // for cross-validation / test phase
                    if (!classification) {
                        // get the current training label if a set of labels are provided
                        if (!trainingLabels.empty()) {
                            if (trainingLabels.front().onset <= i) {
                                currentLabel = trainingLabels.front().name;
                                trainingLabels.pop_front();
                            }
                        }

                        // turn off learning for classification
                        if (learningOffSignal != -1) {
                            if (learningStatus==true && i >= learningOffSignal) {
                                if (verbose != 0) {
                                    std::cout << "learning turned off at t=" << i << std::endl;
                                }
                                learningStatus = false;
                            }
                        }
                    } else {
                        if (decision_making) {
                            if (i - decision_pre_ts >= decision.timer) {
                                // make all the decision neurons fire
                                for (auto& n: layers[decision.layer_number].neurons) {
                                    // generate a decision spike if the neuron is connected to anything
                                    if (!neurons[n]->getDendriticTree().empty()) {
                                        neurons[n]->update(i, nullptr, this, spikeType::decision);
                                    } else {
                                        if (verbose == 2) {
                                            std::cout << "No neurons have specialised for the decision neuron with the label " << neurons[n]->getClassLabel() << std::endl;
                                        }
                                    }
                                }
                                
                                // saving previous timestamp
                                decision_pre_ts = i;
                            }
                        }
                    }

                    while (!spike_queue.empty() && spike_queue.top().timestamp <= i) {
                        // access first element and update corresponding neuron
                        auto index = spike_queue.top().propagationSynapse->getPostsynapticNeuronID();
                        neurons[index]->updateSync(i, spike_queue.top().propagationSynapse, this, timestep, spike_queue.top().type);
                        neuronStatus[index] = true;

                        // remove first element
                        spike_queue.pop();
                    }

                    // update neurons that haven't received a spike
                    for (auto idx=0; idx<neurons.size(); idx++) {
                        if (neuronStatus[idx]) {
                            neuronStatus[idx] = false;
                        } else {
                            // only update neurons if the previous layer is propagating
                            if (neurons[idx]->getLayerID() == 0) {
                                neurons[idx]->updateSync(i, nullptr, this, timestep, spikeType::none);
                            } else {
                                if (!layers[neurons[idx]->getLayerID()-1].do_not_propagate) {
                                    neurons[idx]->updateSync(i, nullptr, this, timestep, spikeType::none);
                                }
                            }
                        }
                    }
                }
            } else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }
		
        // method used to label and set the weights for the neurons connecting to the decision-making layer
        void prepareDecisionMaking() {
            if (decision_making) {

                if (verbose == 1) {
                    std::cout << "assigning labels to neurons and connecting them to their respective decision neuron" << std::endl;
                }
                
                // clearing synapses in case user accidentally created them on decision-making neurons earlier
                for (auto& decision_n: layers[decision.layer_number].neurons) {
                    neurons[decision_n]->getAxonTerminals().clear();
                    neurons[decision_n]->getDendriticTree().clear();
                }
                
                // loop through last layer before DM
                for (auto& pre_decision_n: layers[decision.layer_number-1].neurons) {
                    auto& neuron_to_label = neurons[pre_decision_n];
                    if (!neuron_to_label->getDecisionQueue().empty()) {
                        // resetting the unordered map values to 0 for every neuron
                        for (auto& label: classes_map) {
                            label.second = 0;
                        }
                        
                        // loop through the decision_queue of a neuron and find the number of spikes per label
                        for (auto& label: neuron_to_label->getDecisionQueue()) {
                            ++classes_map[label];
                        }
                        
                        // return the element with the maximum number of spikes
                        auto max_label = *std::max_element(classes_map.begin(), classes_map.end(), [](const std::pair<std::string, int> &p1,
                                                                                                      const std::pair<std::string, int> &p2) {
                                                                                        return p1.second < p2.second;
                                                                                    });
                        
                        // assign label to neuron if element larger than the rejection threshold
                        if (max_label.second / neuron_to_label->getDecisionQueue().size() >= decision.rejection_threshold) {
                            neuron_to_label->setClassLabel(max_label.first);
                        }
                        
                        for (auto& decision_n: layers[decision.layer_number].neurons) {
                            // connect the neuron to its corresponding decision making neuron if they have the same label
                            if (!max_label.first.compare(neurons[decision_n]->getClassLabel())) {
                                neuron_to_label->makeSynapse<Dirac>(neurons[decision_n].get(), 100, 1, 0, synapseType::excitatory);
                            }
                        }
                    }
                }
            }
        }
        
		// ----- IMPLEMENTATION VARIABLES -----
        int                                                 verbose;
        std::priority_queue<spike>                          spike_queue;
        std::deque<spike>                                   predictedSpikes;
        std::vector<layer>                                  layers;
		std::vector<std::unique_ptr<Neuron>>                neurons;
        std::deque<std::unique_ptr<Addon>>                  addons;
        std::unique_ptr<MainThreadAddon>                    thAddon;
		std::deque<label>                                   trainingLabels;
        bool                                                decision_making;
        std::unordered_map<std::string, int>                classes_map;
        std::string                                         currentLabel;
		bool                                                learningStatus;
		double                                              learningOffSignal;
        int                                                 maxDelay;
        bool                                                asynchronous;
        std::mt19937                                        randomEngine;
        decisionHeuristics                                  decision;
        double                                              decision_pre_ts;
    };
}
