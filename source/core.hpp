/* 
 * core.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: Last Version: 28/02/2019
 *
 * Information: the core.hpp contains both the network and the polymorphic Neuron class:
 *  - The Network class acts as a spike manager
 *  - The Neuron class defines a neuron and its parameters. It can take in a pointer to a LearningRuleHandler object to define which learning rule it follows
 */

#pragma once

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

#ifdef TBB
#include "tbb/tbb.h"
#endif

#include "randomDistributions/normal.hpp"
#include "randomDistributions/cauchy.hpp"
#include "randomDistributions/lognormal.hpp"
#include "randomDistributions/uniform.hpp"
#include "dataParser.hpp"
#include "addOn.hpp"
#include "mainThreadAddOn.hpp"
#include "learningRuleHandler.hpp"
#include "synapticKernelHandler.hpp"
#include "dependencies/json.hpp"

#include "synapticKernels/exponential.hpp"
#include "synapticKernels/dirac.hpp"
#include "synapticKernels/step.hpp"

namespace hummus {
    
	class Neuron;
    
    // to be used as feature maps
	struct sublayer {
		std::vector<std::size_t>    neurons;
		int                         ID;
	};
	
    // structure containing a population of neurons
	struct layer {
		std::vector<sublayer>       sublayers;
        std::vector<std::size_t>    neurons;
		int                         ID;
		int                         width;
		int                         height;
	};
	
    struct synapse {
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
        synapse*  propagationSynapse;
        spikeType type;
    };
    
	class Network;
	
	class Neuron {
        
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Neuron(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<int, int> _xyCoordinates, std::vector<LearningRuleHandler*> _learningRules, SynapticKernelHandler* _synapticKernel, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70) :
                neuronID(_neuronID),
                layerID(_layerID),
                sublayerID(_sublayerID),
                rfCoordinates(_rfCoordinates),
                xyCoordinates(_xyCoordinates),
                threshold(_threshold),
                potential(_restingPotential),
                restingPotential(_restingPotential),
                initialSynapse{nullptr, nullptr, 1, 0, 0},
                learningRules(_learningRules),
                eligibilityTrace(0),
                eligibilityDecay(_eligibilityDecay),
                previousSpikeTime(0),
                previousInputTime(0),
                neuronType(0),
                synapticKernel(_synapticKernel),
                adaptation(1),
                synapticEfficacy(1) {}
    	
		virtual ~Neuron(){}
		
		// ----- PUBLIC METHODS -----
		
		// ability to do things inside a neuron, outside the constructor before the network actually runs
		virtual void initialisation(Network* network){}
		
		// asynchronous update method
		virtual void update(double timestamp, synapse* a, Network* network, spikeType type) = 0;
        
		// synchronous update method
		virtual void updateSync(double timestamp, synapse* a, Network* network, double timestep) {
			update(timestamp, a, network, spikeType::none);
		}
        
        // reset a neuron to its initial status
        virtual void resetNeuron(Network* network) {
            previousInputTime = 0;
            previousSpikeTime = 0;
            potential = restingPotential;
            eligibilityTrace = 0;
        }
        
        // write neuron parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
        
        // adds a synapse that connects two Neurons together - used to propagate spikes
        void addSynapse(Neuron* postNeuron, float weight, float delay, int probability=100, bool redundantConnections=true) {
            if (postNeuron) {
                if (connectionProbability(probability)) {
                    if (redundantConnections == false) {
                        int ID = postNeuron->getNeuronID();
                        auto result = std::find_if(postSynapses.begin(), postSynapses.end(), [&](std::unique_ptr<synapse>& a){return a->postNeuron->getNeuronID() == ID;});
                        
                        if (result == postSynapses.end()) {
                            postSynapses.emplace_back(new synapse{this, postNeuron, weight, delay, 0});
                            postNeuron->getPreSynapses().emplace_back(postSynapses.back().get());
                        } 
                    }
                    else {
                        postSynapses.emplace_back(new synapse{this, postNeuron, weight, delay, 0});
                        postNeuron->getPreSynapses().emplace_back(postSynapses.back().get());
                    }
                }
            } else {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
        // initialise the initial synapse when a neuron receives an input event
        spike prepareInitialSpike(double timestamp) {
            if (!initialSynapse.postNeuron) {
                initialSynapse.postNeuron = this;
            }
            return spike{timestamp, &initialSynapse, spikeType::normal};
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
        
        std::pair<int, int> getXYCoordinates() const {
            return xyCoordinates;
        }
        
        void setXYCoordinates(int X, int Y) {
            xyCoordinates.first = X;
            xyCoordinates.second = Y;
        }
        
        std::vector<synapse*>& getPreSynapses() {
            return preSynapses;
        }
        
        std::vector<std::unique_ptr<synapse>>& getPostSynapses() {
            return postSynapses;
        }
        
        synapse& getInitialSynapse() {
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
        
        std::vector<LearningRuleHandler*> getLearningRules() const {
            return learningRules;
        }
        
        float getEligibilityTrace() const {
            return eligibilityTrace;
        }
        
        float getEligibilityDecay() const {
            return eligibilityDecay;
        }
        
        void setEligibilityDecay(float newDecay) {
            eligibilityDecay = newDecay;
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
        
        float getAdaptation() const {
            return adaptation;
        }
        
        float setAdaptation(float newValue) {
            return adaptation = newValue;
        }
        int getType() const {
            return neuronType;
        }
        
        void addLearningRule(LearningRuleHandler* rule) {
            learningRules.push_back(rule);
        }
        
        std::vector<std::pair<int, std::vector<float>>> getLearningInfo() const {
            return learningInfo;
        }
        
        void addLearningInfo(std::pair<int, std::vector<float>> ruleInfo) {
            learningInfo.push_back(ruleInfo);
        }
		
        SynapticKernelHandler* getSynapticKernel() {
			return synapticKernel;
		}
		
    protected:
        
        // winner-take-all algorithm
        virtual void WTA(double timestamp, Network* network) {}
        
        // loops through any learning rules and activates them
        virtual void requestLearning(double timestamp, synapse* a, Network* network){}
        
		// ----- NEURON PARAMETERS -----
        int                                              neuronID;
        int                                              layerID;
        int                                              sublayerID;
        std::pair<int, int>                              rfCoordinates;
        std::pair<int, int>                              xyCoordinates;
		std::vector<synapse*>                            preSynapses;
        std::vector<std::unique_ptr<synapse>>            postSynapses;
        synapse                                          initialSynapse;
        float                                            threshold;
        float                                            potential;
        float                                            restingPotential;
        std::vector<LearningRuleHandler*>                learningRules;
        float                                            eligibilityTrace;
        float                                            eligibilityDecay;
        double                                           previousSpikeTime;
        double                                           previousInputTime;
        float                                            synapticEfficacy;
        float                                            adaptation;
        int                                              neuronType;
        SynapticKernelHandler*                           synapticKernel;
        std::vector<std::pair<int, std::vector<float>>>  learningInfo; // used to save network into JSON format
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
                verbose(0),
                maxDelay(0) {
                    // seeding and initialising random engine with a Mersenne Twister pseudo-random generator
                    std::random_device device;
                    randomEngine = std::mt19937(device());
                }
		
		Network(MainThreadAddOn* _thAddOn) : Network({}, _thAddOn) {}
		
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
                    {"learningRules", nlohmann::json::array()},
					{"synapticKernels", nlohmann::json::array()},
                });
                auto& learningRules = jsonNetwork["layers"].back()["learningRules"];
                for (auto rule: neurons[l.neurons[0]]->getLearningInfo()) {
                    learningRules.push_back({{"ID",rule.first},{"Parameters", rule.second}});
                }
				
                // saving the synaptic kernels and what layer they were used in
				for (auto& s: synapticKernels) {
					if (neurons[l.neurons[0]]->getSynapticKernel()) {
						if (s.get()->getKernelID() == neurons[l.neurons[0]]->getSynapticKernel()->getKernelID()) {
							s->toJson(jsonNetwork["layers"].back()["synapticKernels"]);
						}
					}
				}
				
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
        void addLayer(int _numberOfNeurons, std::vector<LearningRuleHandler*> _learningRules, Args&&... args) {
            
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
                neurons.emplace_back(std::unique_ptr<T>(new T(static_cast<int>(k), layerID, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), _learningRules, std::forward<Args>(args)...)));
                neuronsInLayer.emplace_back(neurons.size()-1);
            }
            
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1});
        }
		
        // adds a 2 dimensional grid of neurons
        template <typename T, typename... Args>
        void add2dLayer(int gridW, int gridH, int _sublayerNumber, std::vector<LearningRuleHandler*> _learningRules, Args&&... args) {
            
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
                    neurons.emplace_back(std::unique_ptr<T>(new T(static_cast<int>(k)+counter, layerID, i, std::pair<int, int>(0, 0), std::pair<int, int>(x, y), _learningRules, std::forward<Args>(args)...)));
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
            layers.emplace_back(layer{sublayers, neuronsInLayer, layerID, gridW, gridH});
        }
		
        // creates a layer that is a convolution of the previous layer, depending on the kernel size and the stride. First set of paramaters are to characterize the synapses. Second set of parameters are parameters for the neuron. We can even add more sublayers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T, typename F, typename... Args>
        void addConvolutionalLayer(layer presynapticLayer, int kernelSize, int stride, F&& lambdaFunction, int probability, int _sublayerNumber, std::vector<LearningRuleHandler*> _learningRules, Args&&... args) {
            
            // find how many neurons have previously spiked
            int layershift = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    if (l.ID != presynapticLayer.ID) {
                        layershift += l.neurons.size();
                    }
                }
            }

            // finding the number of receptive fields
            int newWidth = std::ceil((presynapticLayer.width - kernelSize + 1) / static_cast<float>(stride));
            int newHeight = std::ceil((presynapticLayer.height - kernelSize + 1) / static_cast<float>(stride));
            
            int trimmedColumns = std::abs(newWidth - std::ceil((presynapticLayer.width - stride + 1) / static_cast<float>(stride)));
            int trimmedRows = std::abs(newHeight - std::ceil((presynapticLayer.height - stride + 1) / static_cast<float>(stride)));
            
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
            
            // creating the new layer of neurons
            add2dLayer<T>(newWidth, newHeight, _sublayerNumber, _learningRules, std::forward<Args>(args)...);

            // finding range to calculate a moore neighborhood
            float range;
            if (kernelSize % 2 == 0) {
                range = kernelSize - std::ceil(kernelSize / static_cast<float>(2)) - 0.5;
            }
            else {
                range = kernelSize - std::ceil(kernelSize / static_cast<float>(2));
            }
            
            // number of neurons surrounding the center
            int mooreNeighbors = std::pow((2*range + 1),2);
            
            // looping through the newly created layer to connect them to the correct receptive fields
            for (auto& convSub: layers.back().sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    
                    // initialising window on the first center coordinates
                    std::pair<float, float> centerCoordinates((kernelSize-1)/static_cast<float>(2), (kernelSize-1)/static_cast<float>(2));
                    
                    // number of neurons = number of receptive fields in the presynaptic Layer
                    int row = 0; int col = 0;
                    for (auto& n: convSub.neurons) {
                        
                        // finding the coordinates for the presynaptic neurons in each receptive field
                        for (auto i=0; i<mooreNeighbors; i++) {
                            int x = centerCoordinates.first + ((i % kernelSize) - range);
                            int y = centerCoordinates.second + ((i / kernelSize) - range);

                            // 2D to 1D mapping to get the index from x y coordinates
                            int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;
							
                            // calculating weights and delays according to the provided distribution
                            const std::pair<float, float> weight_delay = lambdaFunction(x, y, convSub.ID);

                            // changing the neuron's receptive field coordinates from the default
                            neurons[idx].get()->setRfCoordinates(row, col);

                            // connecting neurons from the presynaptic layer to the convolutional one
                            neurons[idx].get()->addSynapse(neurons[n].get(), weight_delay.first, weight_delay.second, probability, true);
                            
                            // to shift the network runtime by the maximum delay in the clock mode
                            maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                        }
                        
                        // finding the coordinates for the center of each receptive field
                        centerCoordinates.first += stride;
                        if (centerCoordinates.first >= presynapticLayer.width - trimmedColumns) {
                            centerCoordinates.first = (kernelSize-1)/static_cast<float>(2);
                            centerCoordinates.second += stride;
                        }
                        
                        // updating receptive field indices
                        row += 1;
                        if (row == newWidth) {
                            col += 1;
                            row = 0;
                        }
                    }
                    sublayershift += preSub.neurons.size();
                }
            }
        }
        
        // creates a layer that is a subsampled version of the previous layer, to the nearest divisible grid size (non-overlapping receptive fields). First set of paramaters are to characterize the synapses. Second set of parameters are parameters for the neuron. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T, typename F, typename... Args>
        void addPoolingLayer(layer presynapticLayer, F&& lambdaFunction, int probability, std::vector<LearningRuleHandler*> _learningRules, Args&&... args) {
            
            // find how many neurons have previously spiked
            int layershift = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    if (l.ID != presynapticLayer.ID) {
                        layershift += l.neurons.size();
                    }
                }
            }
            
        	// find greatest common denominator
        	int gcd = -1;
        	for (auto i = 1; i <= presynapticLayer.width && i <= presynapticLayer.height; i++) {
				if (presynapticLayer.width % i == 0 && presynapticLayer.height % i == 0) {
                    if ((presynapticLayer.width  == i || presynapticLayer.height == i) && gcd != -1) {
                        break;
                    } else {
                        gcd = i;
                    }
				}
			}
			
            if (verbose != 0) {
                std::cout << "subsampling by a factor of " << gcd << std::endl;
            }
            
        	// create pooling layer of neurons with correct dimensions
            add2dLayer<T>(presynapticLayer.width/gcd, presynapticLayer.height/gcd, static_cast<int>(presynapticLayer.sublayers.size()), _learningRules, std::forward<Args>(args)...);
			
            float range;
            // if size of kernel is an even number
            if (gcd % 2 == 0) {
                range = gcd - std::ceil(gcd / static_cast<float>(2)) - 0.5;
            // if size of kernel is an odd number
            } else {
                // finding range to calculate a moore neighborhood
                range = gcd - std::ceil(gcd / static_cast<float>(2));
            }
            
            // number of neurons surrounding the center
            int mooreNeighbors = std::pow((2*range + 1),2);
            
            for (auto& poolSub: layers.back().sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    if (poolSub.ID == preSub.ID) {
                        
                        // initialising window on the first center coordinates
                        std::pair<float, float> centerCoordinates((gcd-1)/static_cast<float>(2), (gcd-1)/static_cast<float>(2));
                        
                        // number of neurons = number of receptive fields in the presynaptic Layer
                        int row = 0; int col = 0;
                        for (auto& n: poolSub.neurons) {

                            // finding the coordinates for the presynaptic neurons in each receptive field
                            for (auto i=0; i<mooreNeighbors; i++) {
                                
                                int x = centerCoordinates.first + ((i % gcd) - range);
                                int y = centerCoordinates.second + ((i / gcd) - range);
                                
                                // 2D to 1D mapping to get the index from x y coordinates
                                int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;
                                
                                // calculating weights and delays according to the provided distribution
                                const std::pair<float, float> weight_delay = lambdaFunction(x, y, poolSub.ID);
                                
                                // changing the neuron's receptive field coordinates from the default
                                neurons[idx].get()->setRfCoordinates(row, col);
                                
                                // connecting neurons from the presynaptic layer to the convolutional one
                                neurons[idx].get()->addSynapse(neurons[n].get(), weight_delay.first, weight_delay.second, probability, true);
                                
                                // to shift the network runtime by the maximum delay in the clock mode
                                maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                            }
                            
                            // finding the coordinates for the center of each receptive field
                            centerCoordinates.first += gcd;
                            if (centerCoordinates.first >= presynapticLayer.width) {
                                centerCoordinates.first = (gcd-1)/static_cast<float>(2);
                                centerCoordinates.second += gcd;
                            }
                            
                            // updating receptive field indices
                            row += 1;
                            if (row == presynapticLayer.width/gcd) {
                                col += 1;
                                row = 0;
                            }
                        }
                    }
                    sublayershift += preSub.neurons.size();
                }
            }
			
            
        }
        
        // add a one dimensional layer of decision-making neurons that are labelled according to the provided labels - must be on the last layer
        template <typename T>
        void addDecisionMakingLayer(std::string trainingLabelFilename, SynapticKernelHandler* _synapticKernel, bool _preTrainingLabelAssignment=true, std::vector<LearningRuleHandler*> _learningRules={}, int _refractoryPeriod=1000, bool _homeostasis=false, float _decayPotential=20, float _eligibilityDecay=20, float _decayWeight=0, float _decayHomeostasis=10, float _homeostasisBeta=1, float _threshold=-50, float _restingPotential=-70, float _externalCurrent=100) {
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
            int layerID = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layerID = layers.back().ID+1;
            }
            
            // add decision-making neurons
            std::vector<std::size_t> neuronsInLayer;
            if (preTrainingLabelAssignment) {
                for (auto k=0+shift; k<static_cast<int>(uniqueLabels.size())+shift; k++) {
                    neurons.emplace_back(std::unique_ptr<T>(new T(static_cast<int>(k), layerID, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), _learningRules, _synapticKernel, _homeostasis, _decayPotential, _refractoryPeriod, _eligibilityDecay, _decayWeight, _decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _externalCurrent, uniqueLabels[k-shift])));
                    
                    neuronsInLayer.emplace_back(neurons.size()-1);
                }
            } else {
                for (auto k=0+shift; k<static_cast<int>(uniqueLabels.size())+shift; k++) {
                    neurons.emplace_back(std::unique_ptr<T>(new T(static_cast<int>(k), layerID, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), _learningRules, _synapticKernel, _homeostasis, _decayPotential, _refractoryPeriod, _eligibilityDecay, _decayWeight, _decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _externalCurrent, "")));
                    
                    neuronsInLayer.emplace_back(neurons.size()-1);
                }
            }
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1});
        }
        
        // add a one-dimensional reservoir of randomly interconnected neurons without any learning rule (feedforward, feedback and self-excitation) with randomised weights and no delays. lambdaFunction: Takes in one of the classes inside the randomDistributions folder to define a distribution for the weights.
		
        template <typename T, typename F, typename... Args>
        void addReservoir(int _numberOfNeurons, F&& lambdaFunction, int feedforwardProbability, int feedbackProbability, int selfExcitationProbability, Args&&... args) {
            
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
            
            // creating the reservoir of neurons
            std::vector<std::size_t> neuronsInLayer;
            for (auto k=0+shift; k<_numberOfNeurons+shift; k++) {
                neurons.emplace_back(std::unique_ptr<T>(new T(k, layerID, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), {}, std::forward<Args>(args)...)));
                neuronsInLayer.emplace_back(neurons.size()-1);
            }
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layerID, -1, -1});
            
            // calculating weights and delays according to the provided distribution
			const std::pair<float, float> weight_delay = lambdaFunction(0, 0, 0);
            
            // connecting the reservoir
            for (auto pre: neuronsInLayer) {
                for (auto post: neuronsInLayer) {
                    // self-excitation probability
                    if (pre == post) {
                        neurons[pre].get()->addSynapse(neurons[post].get(), weight_delay.first, 0, selfExcitationProbability, true);
                    } else {
                        // feedforward probability
                        neurons[pre].get()->addSynapse(neurons[post].get(), weight_delay.first, 0, feedforwardProbability, true);
                        
                        // feedback probability
                        neurons[post].get()->addSynapse(neurons[pre].get(), weight_delay.first, 0, feedbackProbability, true);
                    }
                }
            }
        }
        
		// ----- LAYER CONNECTION METHODS -----
		
        // one to one connections between layers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename F>
        void oneToOne(layer presynapticLayer, layer postsynapticLayer, F&& lambdaFunction, int probability=100) {
            // error handling
            if (presynapticLayer.neurons.size() != postsynapticLayer.neurons.size() && presynapticLayer.width == postsynapticLayer.width && presynapticLayer.height == postsynapticLayer.height) {
                throw std::logic_error("The presynaptic and postsynaptic layers do not have the same number of neurons. Cannot do a one-to-one connection");
            }
            
            for (auto preSubIdx=0; preSubIdx<presynapticLayer.sublayers.size(); preSubIdx++) {
                for (auto preNeuronIdx=0; preNeuronIdx<presynapticLayer.sublayers[preSubIdx].neurons.size(); preNeuronIdx++) {
                    for (auto postSubIdx=0; postSubIdx<postsynapticLayer.sublayers.size(); postSubIdx++) {
                        for (auto postNeuronIdx=0; postNeuronIdx<postsynapticLayer.sublayers[postSubIdx].neurons.size(); postNeuronIdx++) {
                            if (preNeuronIdx == postNeuronIdx) {
                                const std::pair<float, float> weight_delay = lambdaFunction(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->getXYCoordinates().first, neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->getXYCoordinates().second, postsynapticLayer.sublayers[postSubIdx].ID);
                                neurons[presynapticLayer.sublayers[preSubIdx].neurons[preNeuronIdx]].get()->addSynapse(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]].get(), weight_delay.first, weight_delay.second, probability, true);

                                // to shift the network runtime by the maximum delay in the clock mode
                                maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                            }
                        }
                    }
                }
            }
        }
        
        // all to all connection between layers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename F>
        void allToAll(layer presynapticLayer, layer postsynapticLayer, F&& lambdaFunction, int probability=100) {
            for (auto& preSub: presynapticLayer.sublayers) {
                for (auto& preNeuron: preSub.neurons) {
                    for (auto& postSub: postsynapticLayer.sublayers) {
                        for (auto& postNeuron: postSub.neurons) {
                            const std::pair<float, float> weight_delay = lambdaFunction(neurons[postNeuron]->getXYCoordinates().first, neurons[postNeuron]->getXYCoordinates().second, postSub.ID);
                            neurons[preNeuron].get()->addSynapse(neurons[postNeuron].get(), weight_delay.first, weight_delay.second, probability, true);
                            
                            // to shift the network runtime by the maximum delay in the clock mode
                            maxDelay = std::max(static_cast<float>(maxDelay), weight_delay.second);
                        }
                    }
                }
            }
        }
		
        // interconnecting a layer with soft winner-takes-all synapses, using negative weights
        void lateralInhibition(layer l, float _weightMean=-1, float _weightstdev=0, int probability=100) {
            if (_weightMean != 0) {
                if (_weightMean > 0 && verbose != 0) {
                    std::cout << "lateral inhibition synapses must have negative weights. The input weight was automatically converted to its negative counterpart" << std::endl;
                }
                
                // generating normal distribution
                std::normal_distribution<> weightRandom(_weightMean, _weightstdev);
                
                for (auto& sub: l.sublayers) {
                    // intra-sublayer soft WTA
                    for (auto& preNeurons: sub.neurons) {
                        for (auto& postNeurons: sub.neurons) {
                            if (preNeurons != postNeurons) {
                                neurons[preNeurons].get()->addSynapse(neurons[postNeurons].get(), -1*std::abs(weightRandom(randomEngine)), 0, probability, true);
                            }
                        }
                    }
                    
                    // inter-sublayer soft WTA
                    for (auto& subToInhibit: l.sublayers) {
                        if (sub.ID != subToInhibit.ID) {
                            for (auto& preNeurons: sub.neurons) {
                                for (auto& postNeurons: subToInhibit.neurons) {
                                    if (neurons[preNeurons]->getRfCoordinates() == neurons[postNeurons]->getRfCoordinates()) {
                                        neurons[preNeurons].get()->addSynapse(neurons[postNeurons].get(), -1*std::abs(weightRandom(randomEngine)), 0, probability, true);
                                    }
                                }
                            }
                        }
                    }
                }
                
            } else {
                throw std::logic_error("lateral inhibition synapses cannot have a null weight");
            }
        }

        // ----- PUBLIC NETWORK METHODS -----
        
        // add spike to the network
        void injectSpike(int neuronIndex, double timestamp) {
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
                return oldSpike.propagationSynapse == s.propagationSynapse;
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
                    if (event.polarity == -1) {
                        // one dimensional data
                        if (event.x == -1) {
                            if (neurons[n]->getNeuronID() == event.neuronID) {
                                injectSpike(static_cast<int>(n), event.timestamp);
                                break;
                            }
                        // two dimensional data
                        } else {
                            if (neurons[n]->getXYCoordinates().first == event.x && neurons[n]->getXYCoordinates().second == event.y) {
                                injectSpike(static_cast<int>(n), event.timestamp);
                                break;
                            }
                        }
                        
                    // 2D data split into sublayers (polarity)
                    } else if (event.polarity == neurons[n]->getSublayerID()) {
                        if (neurons[n]->getXYCoordinates().first == event.x && neurons[n]->getXYCoordinates().second == event.y) {
                            injectSpike(static_cast<int>(n), event.timestamp);
                            break;
                        }
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

                    
                if (_timestep == 0) {
                    eventRunHelper(_runtime, _timestep, false);
                } else {
                    clockRunHelper(_runtime, _timestep, false);
                }
                
                std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
                    
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
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
        
        // initialises a learning rule and adds it to the learning rules vector
        template <typename T, typename... Args>
        T& makeLearningRule(Args&&... args) {
            learningRules.emplace_back(new T(std::forward<Args>(args)...));
            return *dynamic_cast<T*>(learningRules.back().get());
        }
		
        // initialises a synaptic kernel and adds it to the synaptic kernels vector
        template <typename T, typename... Args>
        T& makeSynapticKernel(Args&&... args) {
            synapticKernels.emplace_back(new T(std::forward<Args>(args)...));
            synapticKernels.back().get()->setKernelID(static_cast<int>(synapticKernels.size()));
            return *dynamic_cast<T*>(synapticKernels.back().get());
        }
		
        // ----- SETTERS AND GETTERS -----
		
		std::vector<std::unique_ptr<SynapticKernelHandler>>& getSynapticKernels() {
            return synapticKernels;
        }
		
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
        
        void setMainThreadAddOn(MainThreadAddOn* newThAddon) {
            thAddOn = newThAddon;
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
        
        int getVerbose() const {
            return verbose;
        }
        
        // verbose argument (0 for no couts at all, 1 for network-related print-outs and learning rule print-outs, 2 for network and neuron-related print-outs
        void setVerbose(int value) {
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
            learningStatus = false;
            initialSpikes.clear();
            generatedSpikes.clear();
            for (auto& n: neurons) {
                n->resetNeuron(this);
            }

            injectSpikeFromData(testData);
            
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
                while (!initialSpikes.empty() || !generatedSpikes.empty() || !predictedSpikes.empty()) {
                    
                    std::vector<std::pair<spike,int>> latestSpike;
                    if (!generatedSpikes.empty()) {
                        latestSpike.push_back(std::make_pair(generatedSpikes.front(), 1));
                    }
                    
                    if (!predictedSpikes.empty()) {
                        latestSpike.push_back(std::make_pair(predictedSpikes.front(), 2));
                    }
                    
                    if (!initialSpikes.empty()) {
                        latestSpike.push_back(std::make_pair(initialSpikes.front(), 3));
                    }
                    
                    auto it = std::min_element(latestSpike.begin(), latestSpike.end(), [&](std::pair<spike,int>& a, std::pair<spike,int>& b){ return a.first.timestamp < b.first.timestamp;});
                    auto idx = std::distance(latestSpike.begin(), it);
                    
                    if (latestSpike[idx].second == 1) {
                        requestUpdate(latestSpike[idx].first, classification);
                        generatedSpikes.pop_front();
                    } else if (latestSpike[idx].second == 2) {
                        requestUpdate(latestSpike[idx].first, classification, true);
                        predictedSpikes.pop_front();
                    } else {
                        requestUpdate(latestSpike[idx].first, classification);
                        initialSpikes.pop_front();
                    }
                }
            } else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }
        
        // helper function that runs the network when clock-mode is selected
        void clockRunHelper(double runtime, double timestep, bool classification=false) {
            if (!neurons.empty()) {
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
                                if (verbose != 0) {
                                    std::cout << "learning turned off at t=" << i << std::endl;
                                }
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
                            if (s.propagationSynapse) {
                                return s.propagationSynapse->postNeuron->getNeuronID() == n->getNeuronID();
                            }
                            else {
                                return false;
                            }
                        });
                        
                        local_currentSpikes.resize(std::distance(local_currentSpikes.begin(), it));
                        
                        if (it != currentSpikes.end()) {
                            for (auto& currentSpike: local_currentSpikes) {
                                n->updateSync(i, currentSpike.propagationSynapse, this, timestep);
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
                        spike.propagationSynapse->postNeuron->updateSync(i, spike.propagationSynapse, this, timestep);
                    }
                }
            } else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }
        
        // update neuron status asynchronously
        void requestUpdate(spike s, bool classification=false, bool prediction=false) {
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
            }
            s.propagationSynapse->postNeuron->update(s.timestamp, s.propagationSynapse, this, s.type);
        }
		
		// ----- IMPLEMENTATION VARIABLES -----
        int                                                 verbose;
		std::deque<spike>                                   initialSpikes;
        std::deque<spike>                                   generatedSpikes;
        std::deque<spike>                                   predictedSpikes;
        std::vector<AddOn*>                                 addOns;
        MainThreadAddOn*                                    thAddOn;
        std::vector<layer>                                  layers;
		std::vector<std::unique_ptr<Neuron>>                neurons;
		std::deque<label>                                   trainingLabels;
        std::vector<std::string>                            uniqueLabels;
		bool                                                learningStatus;
		double                                              learningOffSignal;
        int                                                 maxDelay;
        std::string                                         currentLabel;
        bool                                                preTrainingLabelAssignment;
        bool                                                asynchronous;
        std::mt19937                                        randomEngine;
        std::vector<std::unique_ptr<LearningRuleHandler>>   learningRules;
        std::vector<std::unique_ptr<SynapticKernelHandler>> synapticKernels;
    };
}
