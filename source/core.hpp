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
    enum class spike_type {
        initial,
        generated,
        end_of_integration,
        prediction,
        decision,
        none
    };
    
    // parameters for the decision-making layer
    struct decision_heuristics {
        int                         layer_number;
        int                         spike_history_size;
        float                       rejection_threshold; // percentage (0.6 = 60% for a maximum of 1)
        double                      timer;
    };
    
    // the equivalent of feature maps
	struct sublayer {
		std::vector<std::size_t>    neurons;
		int                         id;
	};
	
    // structure organising neurons into layers and sublayers for easier access
	struct layer {
		std::vector<sublayer>       sublayers;
        std::vector<std::size_t>    neurons;
		int                         id;
		int                         width;
		int                         height;
        int                         kernel_size;
        int                         stride;
        bool                        do_not_propagate;
	};

    // spike - propagated between synapses
    struct spike {
        double        timestamp;
        Synapse*      propagation_synapse;
        spike_type    type;
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
                neuron_id(_neuronID),
                layer_id(_layerID),
                sublayer_id(_sublayerID),
                rf_coordinates(_rfCoordinates),
                xy_coordinates(_xyCoordinates),
                threshold(_threshold), //mV
                potential(_restingPotential), //mV
                conductance(_conductance), // pF
                leakage_conductance(_leakageConductance), // nS
                membrane_time_constant(_conductance/_leakageConductance), // ms
                current(0), // pA
                refractory_period(_refractoryPeriod), //ms
                resting_potential(_restingPotential), //mV
                trace(0),
                trace_time_constant(_traceTimeConstant),
                previous_spike_time(0),
                previous_input_time(0),
                class_label(_classLabel),
                neuron_type(0) {
                    // error handling
                    if (membrane_time_constant <= 0) {
                        throw std::logic_error("The potential decay cannot less than or equal to 0");
                    }
                }
    	
		virtual ~Neuron(){}
		
		// ----- PUBLIC METHODS -----
		// ability to do things inside a neuron, outside the constructor before the network actually runs
		virtual void initialisation(Network* network) {}
		
		// asynchronous update method
		virtual void update(double timestamp, Synapse* s, Network* network, spike_type type) = 0;
        
		// synchronous update method
		virtual void update_sync(double timestamp, Synapse* s, Network* network, double timestep, spike_type type) {
			update(timestamp, s, network, type);
		}
        
        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) {
            previous_spike_time = 0;
            potential = resting_potential;
            trace = 0;
            if (clearAddons) {
                relevant_addons.clear();
            }
        }
        
        // write neuron parameters in a JSON format
        virtual void to_json(nlohmann::json& output) {}
        
        // adds a synapse that connects two Neurons together
        template <typename T, typename... Args>
        Synapse* make_synapse(Neuron* postNeuron, int probability, float weight, float delay, Args&&... args) {
            if (postNeuron) {
                if (connection_probability(probability)) {
                    axon_terminals.emplace_back(new T{postNeuron->neuron_id, neuron_id, weight, delay, static_cast<float>(std::forward<Args>(args))...});
                    postNeuron->get_dendritic_tree().emplace_back(axon_terminals.back().get());
                    return axon_terminals.back().get();
                } else {
                    return nullptr;
                }
            } else {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
        // initialise the initial synapse when a neuron receives an input event
        template <typename T, typename... Args>
        spike receive_external_input(double timestamp, Args&&... args) {
            if (!initial_synapse) {
                initial_synapse.reset(new T(std::forward<Args>(args)...));
            }
            return spike{timestamp, initial_synapse.get(), spike_type::initial};
        }
		
        // utility function that returns true or false depending on a probability percentage
        static bool connection_probability(int probability) {
            std::random_device device;
            std::mt19937 random_engine(device());
            std::bernoulli_distribution dist(probability/100.);
            return dist(random_engine);
        }
        
		// ----- SETTERS AND GETTERS -----        
		int get_neuron_id() const {
            return neuron_id;
        }
		
        int get_layer_id() const {
            return layer_id;
        }
        
        int get_sublayer_id() const {
            return sublayer_id;
        }
        
        std::pair<int, int> get_rf_coordinates() const {
            return rf_coordinates;
        }
        
        void set_rf_coordinates(int row, int col) {
            rf_coordinates.first = row;
            rf_coordinates.second = col;
        }
        
        std::pair<float, float> get_xy_coordinates() const {
            return xy_coordinates;
        }
        
        void set_xy_coordinates(float X, float Y) {
            xy_coordinates.first = X;
            xy_coordinates.second = Y;
        }
        
        std::vector<Synapse*>& get_dendritic_tree() {
            return dendritic_tree;
        }
        
        std::vector<std::unique_ptr<Synapse>>& get_axon_terminals() {
            return axon_terminals;
        }
        
        std::unique_ptr<Synapse>& get_initial_synapse() {
            return initial_synapse;
        }
        
        float set_potential(float new_potential) {
            return potential = new_potential;
        }
        
        float get_potential() const {
            return potential;
        }
        
        float get_resting_potential() const {
            return resting_potential;
        }
        
        void set_resting_potential(float newE_l) {
            resting_potential = newE_l;
        }
        
        float get_threshold() const {
            return threshold;
        }
        
        float set_threshold(float _threshold) {
            return threshold = _threshold;
        }
        
        float get_current() const {
            return current;
        }
        void set_current(float newCurrent) {
            current = newCurrent;
        }
        
        float get_trace() const {
            return trace;
        }
        
        void set_trace(float newtrace) {
            trace = newtrace;
        }
        
        float get_trace_time_constant() const {
            return trace_time_constant;
        }
        
        void set_trace_time_constant(float new_constant) {
            trace_time_constant = new_constant;
        }
        
        double get_previous_spike_time() const {
            return previous_spike_time;
        }
        
        double get_previous_input_time() const {
            return previous_input_time;
        }
        
        int get_type() const {
            return neuron_type;
        }
        
        std::vector<Addon*>& get_relevant_addons() {
            return relevant_addons;
        }
        
        void add_relevant_addon(Addon* new_addon) {
            relevant_addons.emplace_back(new_addon);
        }
		
        float get_conductance() const {
            return conductance;
        }
        
        void set_conductance(float k) {
            conductance = k;
        }
        
        void set_leakage_conductance(float k) {
            leakage_conductance = k;
        }
        
        float get_membrane_time_constant() const {
            return membrane_time_constant;
        }
        
        void set_membrane_time_constant(float new_constant) {
            membrane_time_constant = new_constant;
        }
        
        void set_refractory_period(float new_refractory_period) {
            refractory_period = new_refractory_period;
        }
        
        std::deque<std::string>& get_decision_queue() {
            return decision_queue;
        }
        
        std::string get_class_label() const {
            return class_label;
        }
        
        void set_class_label(std::string new_label) {
            class_label = new_label;
        }
        
    protected:
        
        // loops through any learning rules and activates them
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynaptic_neuron, Network* network){}
        
        // ----- NEURON SPATIAL PARAMETERS -----
        int                                        neuron_id;
        int                                        layer_id;
        int                                        sublayer_id;
        std::pair<int, int>                        rf_coordinates;
        std::pair<float, float>                    xy_coordinates;
        
        // ----- SYNAPSES OF THE NEURON -----
        std::vector<Synapse*>                      dendritic_tree;
        std::vector<std::unique_ptr<Synapse>>      axon_terminals;
        std::unique_ptr<Synapse>                   initial_synapse;
        
        // ----- DYNAMIC VARIABLES -----
        float                                      current;
        float                                      potential;
        float                                      trace;
        
        // ----- FIXED PARAMETERS -----
        float                                      threshold;
        float                                      resting_potential;
        float                                      trace_time_constant;
        float                                      conductance;
        float                                      leakage_conductance;
        float                                      membrane_time_constant;
        float                                      refractory_period;
        
        // ----- IMPLEMENTATION PARAMETERS -----
        std::vector<Addon*>                        relevant_addons;
        double                                     previous_spike_time;
        double                                     previous_input_time;
        int                                        neuron_type;
        std::deque<std::string>                    decision_queue;
        std::string                                class_label;
    };
	
    class Network {
        
    public:
		
		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network() :
                learning_status(true),
                asynchronous(false),
                decision_making(false),
                learning_off_signal(-1),
                verbose(0),
                decision_pre_ts(0),
                max_delay(0) {
                    // seeding and initialising random engine with a Mersenne Twister pseudo-random generator
                    std::random_device device;
                    random_engine = std::mt19937(device());
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
                    {"sublayer_number",l.sublayers.size()},
                    {"neuron_number",l.neurons.size()},
                    {"neuron_type",neurons[l.neurons[0]]->get_type()},
                });
            }
			
            // saving the important information needed from the neurons
            for (auto& n: neurons) {
                n->to_json(jsonNetwork["neurons"]);
            }
            
            std::ofstream output_file(filename.append(".json"));
            output_file << output.dump(4);
        }
        
		// ----- NEURON CREATION METHODS -----
		
        // adds one dimensional neurons
        template <typename T, typename... Args>
        layer make_layer(int _numberOfNeurons, std::vector<Addon*> _addons, Args&&... args) {
            
            if (_numberOfNeurons < 0) {
                throw std::logic_error("the number of neurons selected is wrong");
            }
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layer_id = 0;
            if (!layers.empty()) {
				
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
				
                layer_id = layers.back().id+1;
            }

            // building a layer of one dimensional sublayers
            std::vector<std::size_t> neuronsInLayer;
            for (auto k=0+shift; k<_numberOfNeurons+shift; k++) {
                neurons.emplace_back(make_unique<T>(static_cast<int>(k), layer_id, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), std::forward<Args>(args)...));
                neuronsInLayer.emplace_back(neurons.size()-1);
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // building layer structure
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layer_id, -1, -1, -1, -1, false});
            return layers.back();
        }
        
        // takes in training labels and creates DecisionMaking neurons according to the number of classes present - Decision layer should be the last layer
        template <typename T, typename... Args>
        layer make_decision(std::deque<label> _trainingLabels, int _spike_history_size, float _rejection_threshold, double _timer, std::vector<Addon*> _addons, Args&&... args) {
            training_labels = _trainingLabels;
            decision_making = true;
            // do not let the last layer propagate to the decision-making layer during the training phase
            layers.back().do_not_propagate = true;
            
            // add the unique classes to the classes_map
            for (auto& label: training_labels) {
                classes_map.insert({label.name, 0});
            }
            
            int shift = 0;
            
            // find the layer ID
            int layer_id = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += static_cast<int>(l.neurons.size());
                }
                layer_id = layers.back().id+1;
            } else {
                throw std::logic_error("the decision layer can only be on the last layer");
            }
            
            // add decision-making neurons
            std::vector<std::size_t> neuronsInLayer;
            
            int i=0;
            for (auto it: classes_map) {
                neurons.emplace_back(make_unique<T>(i+shift, layer_id, 0, std::pair<int, int>(0, 0), std::pair<int, int>(-1, -1), it.first, std::forward<Args>(args)...));
                neuronsInLayer.emplace_back(neurons.size()-1);
                ++i;
            }
            
            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }
            
            // saving the decision parameters
            decision.layer_number = layer_id;
            decision.spike_history_size = _spike_history_size;
            decision.rejection_threshold = _rejection_threshold;
            decision.timer = _timer;
            
            // building layer structure
            layers.emplace_back(layer{{sublayer{neuronsInLayer, 0}}, neuronsInLayer, layer_id, -1, -1, -1, -1, false});
            return layers.back();
        }
        
        // overload for the makeDecision function that takes in a path to a text label file with the format: label_name timestamp
        template <typename T, typename... Args>
        layer make_decision(std::string trainingLabelFilename, int _spike_history_size, float _rejection_threshold, double _timer, std::vector<Addon*> _addons, Args&&... args) {
            DataParser dataParser;
            auto training_labels = dataParser.read_txt_labels(trainingLabelFilename);
            return make_decision<T>(training_labels, _spike_history_size, _rejection_threshold, _timer, _addons, std::forward<Args>(args)...);
        }
        
        // adds neurons arranged in circles of various radii
        template <typename T, typename... Args>
        layer make_circle(int _numberOfNeurons, std::vector<float> _radii, std::vector<Addon*> _addons, Args&&... args) {
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layer_id = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layer_id = layers.back().id+1;
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
                    neurons.emplace_back(make_unique<T>(static_cast<int>(k)+counter, layer_id, i, std::pair<int, int>(0, 0), std::pair<float, float>(u, v), std::forward<Args>(args)...));
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
            layers.emplace_back(layer{sublayers, neuronsInLayer, layer_id, -1, -1, -1, -1, false});
            return layers.back();
        }
        
        // adds a 2 dimensional grid of neurons
        template <typename T, typename... Args>
        layer make_grid(int gridW, int gridH, int _sublayerNumber, std::vector<Addon*> _addons, Args&&... args) {
            
            // find number of neurons to build
            int numberOfNeurons = gridW * gridH;
            
            unsigned long shift = 0;
            
            // find the layer ID
            int layer_id = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layer_id = layers.back().id+1;
            }
            
            // building a layer of two dimensional sublayers
            int counter = 0;
            std::vector<sublayer> sublayers;
            std::vector<std::size_t> neuronsInLayer;
            for (int i=0; i<_sublayerNumber; i++) {
                std::vector<std::size_t> neuronsInSublayer;
                int x = 0; int y = 0;
                for (auto k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(make_unique<T>(static_cast<int>(k)+counter, layer_id, i, std::pair<int, int>(0, 0), std::pair<int, int>(x, y), std::forward<Args>(args)...));
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
            layers.emplace_back(layer{sublayers, neuronsInLayer, layer_id, gridW, gridH, -1, -1, false});
            return layers.back();
        }
        
        // overloading the makeGrid function to automatically generate a 2D layer according to the previous layer size
        template <typename T, typename... Args>
        layer make_grid(layer presynapticLayer, int _sublayerNumber, int _kernelSize, int _stride, std::vector<Addon*> _addons, Args&&... args) {
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
            int layer_id = 0;
            if (!layers.empty()) {
                for (auto& l: layers) {
                    shift += l.neurons.size();
                }
                layer_id = layers.back().id+1;
            }
            
            // building a layer of two dimensional sublayers
            int counter = 0;
            std::vector<sublayer> sublayers;
            std::vector<std::size_t> neuronsInLayer;
            for (int i=0; i<_sublayerNumber; i++) {
                std::vector<std::size_t> neuronsInSublayer;
                int x = 0; int y = 0;
                for (auto k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(make_unique<T>(static_cast<int>(k)+counter, layer_id, i, std::pair<int, int>(0, 0), std::pair<int, int>(x, y), std::forward<Args>(args)...));
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
            layers.emplace_back(layer{sublayers, neuronsInLayer, layer_id, newWidth, newHeight, _kernelSize, _stride, false});
            return layers.back();
        }
        
        // creates a layer that is a subsampled version of the previous layer, to the nearest divisible grid size
        template <typename T, typename... Args>
        layer make_subsampled_grid(layer presynapticLayer, std::vector<Addon*> _addons, Args&&... args) {
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
            
            return make_grid<T>(presynapticLayer.width/lcd, presynapticLayer.height/lcd, static_cast<int>(presynapticLayer.sublayers.size()), _addons, std::forward<Args>(args)...);
        }
        
		// ----- LAYER CONNECTION METHODS -----
        
        // connecting a layer that is a convolution of the previous layer, depending on the layer kernel size and the stride. Last set of paramaters are to characterize the synapses. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays. Furthermore, you can select the number of synapses per pair of presynaptic and postsynaptic neurons (the arborescence)
        template <typename T, typename F, typename... Args>
        void convolution(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            // error handling
            if (postsynapticLayer.kernel_size == -1 || postsynapticLayer.stride == -1) {
                throw std::logic_error("cannot connect the layers in a convolutional manner as the layers were not built with that in mind (no kernel or stride in the grid layer to define receptive fields");
            }
            
            // find how many neurons there are before the pre and postsynaptic layers
            int layershift = 0;
            if (!layers.empty()) {
                for (auto i=0; i<presynapticLayer.id; i++) {
                    layershift += layers[i].neurons.size();
                }
            }
            
            int trimmedColumns = std::abs(postsynapticLayer.width - std::ceil((presynapticLayer.width - postsynapticLayer.stride + 1) / static_cast<float>(postsynapticLayer.stride)));
            
            // finding range to calculate a moore neighborhood
            float range;
            if (postsynapticLayer.kernel_size % 2 == 0) {
                range = postsynapticLayer.kernel_size - std::ceil(postsynapticLayer.kernel_size / static_cast<float>(2)) - 0.5;
            }
            else {
                range = postsynapticLayer.kernel_size - std::ceil(postsynapticLayer.kernel_size / static_cast<float>(2));
            }
            
            // number of neurons surrounding the center
            int mooreNeighbors = (2*range + 1) * (2*range + 1);
            
            // looping through the newly created layer to connect them to the correct receptive fields
            for (auto& convSub: postsynapticLayer.sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    
                    // initialising window on the first center coordinates
                    std::pair<float, float> centerCoordinates((postsynapticLayer.kernel_size-1)/static_cast<float>(2), (postsynapticLayer.kernel_size-1)/static_cast<float>(2));
                    
                    // number of neurons = number of receptive fields in the presynaptic Layer
                    int row = 0; int col = 0;
                    for (auto& n: convSub.neurons) {
                        
                        // finding the coordinates for the presynaptic neurons in each receptive field
                        for (auto i=0; i<mooreNeighbors; i++) {
                            int x = centerCoordinates.first + ((i % postsynapticLayer.kernel_size) - range);
                            int y = centerCoordinates.second + ((i / postsynapticLayer.kernel_size) - range);
                            
                            // 2D to 1D mapping to get the index from x y coordinates
                            int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;
                            
                            // changing the neuron's receptive field coordinates from the default
                            neurons[idx]->set_rf_coordinates(row, col);
                            
                            // connecting neurons from the presynaptic layer to the convolutional one, depedning on the number of synapses
                            for (auto i=0; i<number_of_synapses; i++) {
                                // calculating weights and delays according to the provided distribution
                                const std::pair<float, float> weight_delay = lambdaFunction(x, y, convSub.id);
                                
                                // creating a synapse between the neurons
                                neurons[idx]->make_synapse<T>(neurons[n].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);
                                
                                // to shift the network runtime by the maximum delay in the clock mode
                                max_delay = std::max(static_cast<float>(max_delay), weight_delay.second);
                            }
                        }
                        
                        // finding the coordinates for the center of each receptive field
                        centerCoordinates.first += postsynapticLayer.stride;
                        if (centerCoordinates.first >= presynapticLayer.width - trimmedColumns) {
                            centerCoordinates.first = (postsynapticLayer.kernel_size-1)/static_cast<float>(2);
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
            if (postsynapticLayer.id - presynapticLayer.id > 1) {
                throw std::logic_error("the layers aren't immediately following each other");
            }
            
            // find how many neurons there are before the pre and postsynaptic layers
            int layershift = 0;
            if (!layers.empty()) {
                for (auto i=0; i<presynapticLayer.id; i++) {
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
                    if (poolSub.id == preSub.id) {
                        
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
                                neurons[idx]->set_rf_coordinates(row, col);
                                
                                for (auto i=0; i<number_of_synapses; i++) {
                                    // calculating weights and delays according to the provided distribution
                                    const std::pair<float, float> weight_delay = lambdaFunction(x, y, poolSub.id);
                                    
                                    // connecting neurons from the presynaptic layer to the convolutional one
                                    neurons[idx]->make_synapse<T>(neurons[n].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);
                                    
                                    // to shift the network runtime by the maximum delay in the clock mode
                                    max_delay = std::max(static_cast<float>(max_delay), weight_delay.second);
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
                            neurons[pre]->make_synapse<T>(neurons[post].get(), selfExcitationProbability, weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                        } else {
                            // feedforward probability
                            neurons[pre]->make_synapse<T>(neurons[post].get(), feedforwardProbability, weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                            
                            // feedback probability
                            neurons[post]->make_synapse<T>(neurons[pre].get(), feedbackProbability, weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                        }
                    }
                }
            }
        }
        
		// connecting two layers according to a weight matrix vector of vectors and a delays matrix vector of vectors (columns for input and rows for output)
        template <typename T, typename... Args>
        void connectivity_matrix(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, std::vector<std::vector<float>> weights, std::vector<std::vector<float>> delays, Args&&... args) {
            
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
                                    neurons[preNeuron]->make_synapse<T>(neurons[postNeuron].get(), 100, weights[preCounter][postCounter], delays[preCounter][postCounter], std::forward<Args>(args)...);
                                }
                            }
                            
                            // to shift the network runtime by the maximum delay in the clock mode
                            max_delay = std::max(static_cast<float>(max_delay), delays[preCounter][postCounter]);
                            
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
        void one_to_one(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
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
                                    const std::pair<float, float> weight_delay = lambdaFunction(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->get_rf_coordinates().first, neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->get_xy_coordinates().second, postsynapticLayer.sublayers[postSubIdx].id);
                                    neurons[presynapticLayer.sublayers[preSubIdx].neurons[preNeuronIdx]]->make_synapse<T>(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);

                                    // to shift the network runtime by the maximum delay in the clock mode
                                    max_delay = std::max(static_cast<float>(max_delay), weight_delay.second);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // all to all connection between layers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T, typename F, typename... Args>
        void all_to_all(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            for (auto& preSub: presynapticLayer.sublayers) {
                for (auto& preNeuron: preSub.neurons) {
                    for (auto& postSub: postsynapticLayer.sublayers) {
                        for (auto& postNeuron: postSub.neurons) {
                            for (auto i=0; i<number_of_synapses; i++) {
                                const std::pair<float, float> weight_delay = lambdaFunction(neurons[postNeuron]->get_rf_coordinates().first, neurons[postNeuron]->get_xy_coordinates().second, postSub.id);
                                neurons[preNeuron]->make_synapse<T>(neurons[postNeuron].get(), probability, weight_delay.first, weight_delay.second, std::forward<Args>(args)...);
                                
                                // to shift the network runtime by the maximum delay in the clock mode
                                max_delay = std::max(static_cast<float>(max_delay), weight_delay.second);
                            }
                        }
                    }
                }
            }
        }

        // interconnecting a layer with soft winner-takes-all synapses, using negative weights
        template <typename T, typename F, typename... Args>
        void lateral_inhibition(layer l, int number_of_synapses, F&& lambdaFunction, int probability, Args&&... args) {
            for (auto& sub: l.sublayers) {
                // intra-sublayer soft WTA
                for (auto& preNeurons: sub.neurons) {
                    for (auto& postNeurons: sub.neurons) {
                        if (preNeurons != postNeurons) {
                            for (auto i=0; i<number_of_synapses; i++) {
                                const std::pair<float, float> weight_delay = lambdaFunction(0, 0, 0);
                                neurons[preNeurons]->make_synapse<T>(neurons[postNeurons].get(), probability, -1*std::abs(weight_delay.first), weight_delay.second, std::forward<Args>(args)...);
                            }
                        }
                    }
                }
                
                // inter-sublayer soft WTA
                for (auto& subToInhibit: l.sublayers) {
                    if (sub.id != subToInhibit.id) {
                        for (auto& preNeurons: sub.neurons) {
                            for (auto& postNeurons: subToInhibit.neurons) {
                                if (neurons[preNeurons]->get_rf_coordinates() == neurons[postNeurons]->get_rf_coordinates()) {
                                    for (auto i=0; i<number_of_synapses; i++) {
                                        const std::pair<float, float> weight_delay = lambdaFunction(0, 0, 0);
                                        neurons[preNeurons]->make_synapse<T>(neurons[postNeurons].get(), probability, -1*std::abs(weight_delay.first), weight_delay.second, std::forward<Args>(args)...);
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
        void inject_spike(spike s) {
            spike_queue.emplace(s);
        }
        
        // overloaded method - creates a spike and adds it to the spike_queue priority queue
        void inject_spike(int neuronIndex, double timestamp) {
            spike_queue.emplace(neurons[neuronIndex]->receive_external_input<Dirac>(timestamp, neuronIndex, -1, 1, 0));
        }
        
        
        // adding spikes predicted by the asynchronous network (timestep = 0) for synaptic integration
        void inject_predicted_spike(spike s, spike_type stype) {
            // remove old spike
            std::remove_if(predicted_spikes.begin(), predicted_spikes.end(),[&](spike oldSpike) {
                return oldSpike.propagation_synapse == s.propagation_synapse;
            });
            
            // change type of new spike
            s.type = stype;
            
            // insert the new spike in the correct place
            predicted_spikes.insert(
                std::upper_bound(predicted_spikes.begin(), predicted_spikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }
        
        // add spikes from an input vector to the network
        void inject_input(std::vector<input>* data) {
            // error handling
            if (neurons.empty()) {
                throw std::logic_error("add neurons before injecting spikes");
            }
            
            for (auto& event: *data) {
                for (auto& n: layers[0].neurons) {
                    // one dimensional data
                    if (event.x == -1) {
                        if (neurons[n]->get_neuron_id() == event.neuron_id) {
                            inject_spike(static_cast<int>(n), event.timestamp);
                            break;
                        }
                    // two dimensional data (with or without polarity is the same because polarity is unused)
                    } else {
                        if (neurons[n]->get_rf_coordinates().first == event.x && neurons[n]->get_xy_coordinates().second == event.y) {
                            inject_spike(static_cast<int>(n), event.timestamp);
                            break;
                        }
                    }
                }
            }
        }
        
        // add a poissonian spike train to the initial spike vector
        void poisson_spike_generator(int neuronIndex, double timestamp, float rate, float timestep, float duration) {
            // calculating number of spikes
            int spike_number = std::floor(duration/timestep);
            
            // initialising the random engine
            std::random_device                     device;
            std::mt19937                           random_engine(device());
            std::uniform_real_distribution<double> distribution(0.0,1.0);
            
            std::vector<double> inter_spike_intervals;
            // generating uniformly distributed random numbers
            for (auto i = 0; i < spike_number; ++i) {
                inter_spike_intervals.emplace_back((- std::log(distribution(random_engine)) / rate) * 1000);
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
                spike_queue.emplace(neurons[neuronIndex]->receive_external_input<Dirac>(spike_time, neuronIndex, -1, 1, 0));
            }
        }
        
        // turn off learning
        void turn_off_learning() {
            learning_status = false;
        }
        
        // overloaded function - turn off learning at a specified timestamp
        void turn_off_learning(double timestamp) {
            learning_off_signal = timestamp;
        }
        
        // running through the network asynchronously if timestep = 0 and synchronously otherwise. This method does not take any data in and just runs the network as is. the only way to add spikes is through the injectSpike / poissonSpikeGenerator or injectSpikesFromData methods
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
                addon->on_start(this);
            }
			
            if (classification) {
                std::cout << "classification is true. This instance will assume a training run has already been done, and will try to initialise the Decision-Making if it was initialised)" << std::endl;
                
                if (th_addon) {
                    th_addon->reset();
                }
                
                // can now propagate through all layers in case a decision-making layer is present
                for (auto& layer: layers) {
                    layer.do_not_propagate = false;
                }
                
                // during a classification run, labels the neurons if a decision-making layer was used
                prepare_decision_making();
            }
                
            std::mutex sync;
            if (th_addon) {
                sync.lock();
            }

                std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();
 
                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                    
                if (_timestep == 0) {
                    event_run_helper(_runtime, _timestep, classification);
                } else {
                    clock_run_helper(_runtime, _timestep, classification);
                }
                
                std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
                    
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
                }
                    
                for (auto& addon: addons) {
                    addon->on_completed(this);
                }
            });

            if (th_addon) {
                th_addon->begin(this, &sync);
            }
            
            spikeManager.join();
            
            // resetting network and clearing addons initialised for this particular run
            reset();
        }

        // running through the network asynchronously if timestep = 0 and synchronously otherwise. This method takes in a vector of inputs from the read_txt_data method
        void run_data(std::vector<input>* trainingData, float _timestep=0, std::vector<input>* testData=nullptr) {
            // the shift variable adds some time to the runtime - to fully visualise current dynamics
            int shift = 20;
            
            if (_timestep == 0) {
                asynchronous = true;
            }
			
            for (auto& n: neurons) {
                n->initialisation(this);
            }
			
            if (learning_off_signal == -1) {
                learning_off_signal = trainingData->back().timestamp+max_delay+shift;
            }
			
            for (auto& addon: addons) {
                addon->on_start(this);
            }

            std::mutex sync;
            if (th_addon) {
                sync.lock();
            }
            
            std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();
                
                // importing training data and running the network through the data
                inject_input(trainingData);
                
                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                if (verbose != 0) {
                    std::cout << "Training the network..." << std::endl;
                }
                
                if (_timestep == 0) {
                    event_run_helper(trainingData->back().timestamp+max_delay+shift, _timestep, false);
                } else {
                    clock_run_helper(trainingData->back().timestamp+max_delay+shift, _timestep, false);
                }
                
                std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s for the training phase." << std::endl;
                }
                
                // importing test data and running it through the network for classification
                if (testData) {
                    // can now propagate through all layers in case a decision-making layer is present
                    for (auto& layer: layers) {
                        layer.do_not_propagate = false;
                    }
                    
                    learning_status = false;
                    for (auto& n: neurons) {
                        n->reset_neuron(this, false);
                    }
                    
                    prepare_decision_making();
                    
                    inject_input(testData);
                    
                    for (auto& addon: addons) {
                        addon->on_predict(this);
                    }
                    
                    if (th_addon) {
                        th_addon->reset();
                    }
                    
                    if (verbose != 0) {
                        std::cout << "Running classification based on a trained network..." << std::endl;
                    }
                    
                    if (_timestep == 0) {
                        event_run_helper(testData->back().timestamp+max_delay+shift, _timestep, true);
                    } else {
                        clock_run_helper(testData->back().timestamp+max_delay+shift, _timestep, true);
                    }
                    
                    if (verbose != 0) {
                        std::cout << "Done." << std::endl;
                    }
                }

                for (auto& addon: addons) {
                    addon->on_completed(this);
                }
            });

            if (th_addon) {
                th_addon->begin(this, &sync);
            }
            
            spikeManager.join();
            
            // resetting network and clearing addons initialised for this particular run
            reset();
        }
        
        // runs the network with data from a .es file
        void run_es() {
            
        }
        
        // runs the network with a database of .es files provided by one of the database generators from the dataParser module
        void run_database() {
            
        }
        
        // reset the network back to the initial conditions without changing the network build
        void reset() {
            learning_status = true;
            learning_off_signal = -1;
            
            for (auto& n: neurons) {
                n->reset_neuron(this);
            }
        }
        
        // initialises an addon that needs to run on the main thread
        template <typename T, typename... Args>
        T& make_gui(Args&&... args) {
            th_addon.reset(new T(std::forward<Args>(args)...));
            return static_cast<T&>(*th_addon);
        }
        
        // initialises an addon and adds it to the addons vector. returns a reference to the add-on
        template <typename T, typename... Args>
        T& make_addon(Args&&... args) {
            addons.emplace_back(new T(std::forward<Args>(args)...));
            return static_cast<T&>(*addons.back());
        }
        
        // ----- SETTERS AND GETTERS -----
		
        std::vector<std::unique_ptr<Neuron>>& get_neurons() {
            return neurons;
        }

        std::vector<layer>& get_layers() {
            return layers;
        }
        
        std::vector<std::unique_ptr<Addon>>& get_addons() {
            return addons;
        }
        
        std::unique_ptr<MainThreadAddon>& get_main_thread_addon() {
            return th_addon;
        }

        void set_main_thread_addon(MainThreadAddon* new_thAddon) {
            th_addon.reset(new_thAddon);
        }

        bool get_learning_status() const {
            return learning_status;
        }
        
        double get_learning_off_signal() const {
            return learning_off_signal;
        }
        
        std::string get_current_label() const {
            return current_label;
        }
        
        bool get_decision_making() {
            return decision_making;
        }
        
        bool get_network_type() const {
            return asynchronous;
        }
        
        int get_verbose() const {
            return verbose;
        }
		
        decision_heuristics& getDecisionParameters() {
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
        
        // helper function that runs the network when event-mode is selected
        void event_run_helper(double runtime, double timestep, bool classification=false) {
            // lambda function to update neuron status asynchronously
            auto requestUpdate = [&](spike s, bool classification) {
                if (!classification) {
                    if (!training_labels.empty()) {
                        if (training_labels.front().onset <= s.timestamp) {
                            current_label = training_labels.front().name;
                            training_labels.pop_front();
                        }
                    }
                    
                    if (learning_off_signal != -1) {
                        if (learning_status==true && s.timestamp >= learning_off_signal) {
                            if (verbose != 0) {
                                std::cout << "learning turned off at t=" << s.timestamp << std::endl;
                            }
                            learning_status = false;
                        }
                    }
                } else {
                    if (decision_making) {
                        if (s.timestamp - decision_pre_ts >= decision.timer) {
                            // make all the decision neurons fire
                            for (auto& n: layers[decision.layer_number].neurons) {
                                // generate a decision spike if the neuron is connected to anything
                                if (!neurons[n]->get_dendritic_tree().empty()) {
                                    neurons[n]->update(s.timestamp, nullptr, this, spike_type::decision);
                                } else {
                                    if (verbose == 2) {
                                        std::cout << "No neurons have specialised for the decision neuron with the label " << neurons[n]->get_class_label() << std::endl;
                                    }
                                }
                            }
                            // saving previous timestamp
                            decision_pre_ts = s.timestamp;
                        }
                    }
                }
                neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, s.type);
            };
            
            if (!neurons.empty()) {
                while (!spike_queue.empty() || !predicted_spikes.empty()) {
                    if (!spike_queue.empty() && predicted_spikes.empty()) {
                        requestUpdate(spike_queue.top(), classification);
                        spike_queue.pop();
                    } else if (!predicted_spikes.empty() && spike_queue.empty()) {
                        requestUpdate(predicted_spikes.front(), classification);
                        predicted_spikes.pop_front();
                    } else if (!predicted_spikes.empty() && !spike_queue.empty()) {
                        if (spike_queue.top().timestamp < predicted_spikes.front().timestamp) {
                            requestUpdate(spike_queue.top(), classification);
                            spike_queue.pop();
                        } else if (predicted_spikes.front().timestamp < spike_queue.top().timestamp) {
                            requestUpdate(predicted_spikes.front(), classification);
                            predicted_spikes.pop_front();
                        } else if (predicted_spikes.front().timestamp == spike_queue.top().timestamp) {
                            requestUpdate(spike_queue.top(), classification);
                            spike_queue.pop();
                        }
                    }
                }
            } else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }
        
        // helper function that runs the network when clock-mode is selected
        void clock_run_helper(double runtime, double timestep, bool classification=false) {
            if (!neurons.empty()) {
                
                // creating vector of the same size as neurons
                std::vector<bool> neuronStatus(neurons.size(), false);
                
                // loop over the full runtime
                for (double i=0; i<runtime; i+=timestep) {
                    // for cross-validation / test phase
                    if (!classification) {
                        // get the current training label if a set of labels are provided
                        if (!training_labels.empty()) {
                            if (training_labels.front().onset <= i) {
                                current_label = training_labels.front().name;
                                training_labels.pop_front();
                            }
                        }

                        // turn off learning for classification
                        if (learning_off_signal != -1) {
                            if (learning_status==true && i >= learning_off_signal) {
                                if (verbose != 0) {
                                    std::cout << "learning turned off at t=" << i << std::endl;
                                }
                                learning_status = false;
                            }
                        }
                    } else {
                        if (decision_making) {
                            if (i - decision_pre_ts >= decision.timer) {
                                // make all the decision neurons fire
                                for (auto& n: layers[decision.layer_number].neurons) {
                                    // generate a decision spike if the neuron is connected to anything
                                    if (!neurons[n]->get_dendritic_tree().empty()) {
                                        neurons[n]->update(i, nullptr, this, spike_type::decision);
                                    } else {
                                        if (verbose == 2) {
                                            std::cout << "No neurons have specialised for the decision neuron with the label " << neurons[n]->get_class_label() << std::endl;
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
                        auto index = spike_queue.top().propagation_synapse->get_postsynaptic_neuron_id();
                        neurons[index]->update_sync(i, spike_queue.top().propagation_synapse, this, timestep, spike_queue.top().type);
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
                            if (neurons[idx]->get_layer_id() == 0) {
                                neurons[idx]->update_sync(i, nullptr, this, timestep, spike_type::none);
                            } else {
                                if (!layers[neurons[idx]->get_layer_id()-1].do_not_propagate) {
                                    neurons[idx]->update_sync(i, nullptr, this, timestep, spike_type::none);
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
        void prepare_decision_making() {
            if (decision_making) {

                if (verbose == 1) {
                    std::cout << "assigning labels to neurons and connecting them to their respective decision neuron" << std::endl;
                }
                
                // clearing synapses in case user accidentally created them on decision-making neurons earlier
                for (auto& decision_n: layers[decision.layer_number].neurons) {
                    neurons[decision_n]->get_axon_terminals().clear();
                    neurons[decision_n]->get_dendritic_tree().clear();
                }
                
                // loop through last layer before DM
                for (auto& pre_decision_n: layers[decision.layer_number-1].neurons) {
                    auto& neuron_to_label = neurons[pre_decision_n];
                    if (!neuron_to_label->get_decision_queue().empty()) {
                        // resetting the unordered map values to 0 for every neuron
                        for (auto& label: classes_map) {
                            label.second = 0;
                        }
                        
                        // loop through the decision_queue of a neuron and find the number of spikes per label
                        for (auto& label: neuron_to_label->get_decision_queue()) {
                            ++classes_map[label];
                        }
                        
                        // return the element with the maximum number of spikes
                        auto max_label = *std::max_element(classes_map.begin(), classes_map.end(), [](const std::pair<std::string, int> &p1,
                                                                                                      const std::pair<std::string, int> &p2) {
                                                                                        return p1.second < p2.second;
                                                                                    });
                        
                        // assign label to neuron if element larger than the rejection threshold
                        if (max_label.second / neuron_to_label->get_decision_queue().size() >= decision.rejection_threshold) {
                            neuron_to_label->set_class_label(max_label.first);
                        }
                        
                        for (auto& decision_n: layers[decision.layer_number].neurons) {
                            // connect the neuron to its corresponding decision making neuron if they have the same label
                            if (!max_label.first.compare(neurons[decision_n]->get_class_label())) {
                                neuron_to_label->make_synapse<Dirac>(neurons[decision_n].get(), 100, 1, 0, synapseType::excitatory);
                            }
                        }
                    }
                }
            }
        }
        
		// ----- IMPLEMENTATION VARIABLES -----
        int                                                 verbose;
        std::priority_queue<spike>                          spike_queue;
        std::deque<spike>                                   predicted_spikes;
        std::vector<layer>                                  layers;
		std::vector<std::unique_ptr<Neuron>>                neurons;
        std::vector<std::unique_ptr<Addon>>                 addons;
        std::unique_ptr<MainThreadAddon>                    th_addon;
		std::deque<label>                                   training_labels;
        bool                                                decision_making;
        std::unordered_map<std::string, int>                classes_map;
        std::string                                         current_label;
		bool                                                learning_status;
		double                                              learning_off_signal;
        int                                                 max_delay;
        bool                                                asynchronous;
        std::mt19937                                        random_engine;
        decision_heuristics                                 decision;
        double                                              decision_pre_ts;
    };
}
