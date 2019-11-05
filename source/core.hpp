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
#include <set>

// external Dependencies
#include "../third_party/json.hpp"
#include "../third_party/sepia.hpp"
#include "../third_party/numpy.hpp"

// random distributions
#include "random_distributions/lognormal.hpp"
#include "random_distributions/uniform.hpp"
#include "random_distributions/normal.hpp"
#include "random_distributions/cauchy.hpp"

// data parser
#include "data_parser.hpp"

// addons
#include "addon.hpp"
#include "main_addon.hpp"

// synapse models
#include "synapse.hpp"
#include "synapses/exponential.hpp"
#include "synapses/square.hpp"
#include "synapses/memristor.hpp"

#ifdef TBB
#include "tbb/tbb.h"
#endif

namespace hummus {

    // used for the event-based mode only in order to predict spike times with dynamic currents
    enum class spike_type {
        initial, // input spikes (real spike)
        generated, // spikes generates by the network (real spike)
        end_of_integration, // asynchronous - updating synapses when they become inactive (not a real spike)
        prediction, // asynchronous - future theoretical spike time (not a real spike)
        decision, // for decision-making (real spike)
        trigger_up, // for ulpec - voltage > resting_potential
        trigger_down, // for ulpec - voltage < resting_potential
        trigger_down_to_up, // for ulpec - postsynaptic pulse
        end_trigger_up, // end of the trigger_up waveform
        end_trigger_down, // end of the trigger_down waveform
        none // synchronous - for updates at every clock (not a real spike)
    };

    // parameters for the decision-making layer
    struct decision_heuristics {
        int                           layer_number; // decision_making layer id
        int                           spike_history_size; // how many spikes to take into consideration for the heuristics
        int                           rejection_threshold; // percentage of spikes that need to belong to the same class in order for a neuron to be labelled
        float                         timer; // selects how often a decision neuron fires. for es files: set to 0 if Decision is to be made at the end of the file
    };

    // receptive_field
    struct receptive_field {
        std::vector<std::size_t>      neurons; // neuron indices belonging to the receptive field
        int                           id; // receptive field ID
    };

    // the equivalent of feature maps
	struct sublayer {
        std::vector<receptive_field>  receptive_fields; // receptive fields of a sublayer
		std::vector<std::size_t>      neurons; // neuron indices belonging to a sublayer
		int                           id; // sublayer ID
	};

    // structure organising neurons into layers and sublayers for easier access
	struct layer {
		std::vector<sublayer>         sublayers; // sublayers belonging to layer
        std::vector<std::size_t>      neurons; // neuron indices belonging to layer
		int                           id; // layer ID
        bool                          active = true; // whether or not a layer receives spikes
		int                           width = -1; // width of the layer (if make_grid is used)
		int                           height = -1; // height of the layer (if make_grid is used)
        int                           kernel_size = -1; // size of the kernel (if make_grid is used with a previous layer as input)
        int                           stride = -1; // stride of the kernel (if make_grid is used with a previous layer as input)
	};

    // spike - propagated between synapses
    struct spike {
        double        timestamp; // timestamp of the spike (arbitrary unit but make sure to stay consistent with all the other parameters)
        Synapse*      propagation_synapse; // which synapse is propagating a spike - for access to pre and post-synaptic neurons to know where to send the spike
        spike_type    type; // type of spike (to differentiate between real spikes and other spikes used by the network)

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
        Neuron(int _neuron_id, int _layer_id, int _sublayer_id, int _rf_id, std::pair<int, int> _xy_coordinates, int _refractory_period=3, float _capacitance=200, float _leakage_conductance=10, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70, std::string _classLabel="") :
                neuron_id(_neuron_id),
                layer_id(_layer_id),
                sublayer_id(_sublayer_id),
                rf_id(_rf_id),
                xy_coordinates(_xy_coordinates),
                current(0), // pA
                potential(_restingPotential), //mV
                trace(0),
                threshold(_threshold), //mV
                resting_potential(_restingPotential), //mV
                trace_time_constant(_traceTimeConstant),
                capacitance(_capacitance), // pF
                leakage_conductance(_leakage_conductance), // nS
                membrane_time_constant(_capacitance/_leakage_conductance), // ms
                refractory_period(_refractory_period), //ms
                active(true),
                previous_spike_time(0),
                previous_input_time(0),
                neuron_type(0),
                class_label(_classLabel) {
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
		virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) = 0;

		// synchronous update method
		virtual void update_sync(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) {
			update(timestamp, s, network, timestep, type);
		}

        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) {
            active = true;
            previous_input_time = 0;
            previous_spike_time = 0;
            potential = resting_potential;
            trace = 0;

            for (auto& dendrite: dendritic_tree) {
                dendrite->reset();
            }

            for (auto& axon_terminal: axon_terminals) {
                axon_terminal->reset();
            }

            if (clearAddons) {
                relevant_addons.clear();
            }
        }

        // adds a synapse that connects two Neurons together
        template <typename T = Synapse, typename... Args>
        Synapse* make_synapse(Neuron* post_neuron, float weight, float delay, Args&&... args) {
            if (post_neuron) {
                axon_terminals.emplace_back(new T{post_neuron->neuron_id, neuron_id, weight, delay, static_cast<float>(std::forward<Args>(args))...});
                post_neuron->get_dendritic_tree().emplace_back(axon_terminals.back().get());
                return axon_terminals.back().get();
            } else {
                throw std::logic_error("Neuron does not exist");
            }
        }

        // initialise the initial synapse when a neuron receives an event
        template <typename T = Synapse, typename... Args>
        spike receive_external_input(double timestamp, spike_type type, Args&&... args) {
            if (!initial_synapse) {
                initial_synapse.reset(new T(std::forward<Args>(args)...));
            }
            return spike{timestamp, initial_synapse.get(), type};
        }

        // share information - generic getter that can be used for accessing child members from parent
        virtual float share_information() { return 0; }

        // write neuron parameters in a JSON format
        virtual void to_json(nlohmann::json& output) {}

		// ----- SETTERS AND GETTERS -----
        bool get_activity() const {
            return active;
        }

        void set_activity(bool a) {
            active = false;
        }

		int get_neuron_id() const {
            return neuron_id;
        }

        int get_layer_id() const {
            return layer_id;
        }

        int get_sublayer_id() const {
            return sublayer_id;
        }

        int get_rf_id() const {
            return rf_id;
        }

        void set_rf_id(int id) {
            rf_id = id;
        }

        std::pair<int, int> get_xy_coordinates() const {
            return xy_coordinates;
        }

        void set_xy_coordinates(int X, int Y) {
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

        float get_capacitance() const {
            return capacitance;
        }

        void set_capacitance(float k) {
            capacitance = k;
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

        void set_refractory_period(int new_refractory_period) {
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

        // winner_takes_all implementation
        virtual void winner_takes_all(double timestamp, Network* network) {}

        // ----- NEURON SPATIAL PARAMETERS -----
        int                                       neuron_id;
        int                                       layer_id;
        int                                       sublayer_id;
        int                                       rf_id;
        std::pair<int, int>                       xy_coordinates;

        // ----- SYNAPSES OF THE NEURON -----
        std::vector<Synapse*>                     dendritic_tree;
        std::vector<std::unique_ptr<Synapse>>     axon_terminals;
        std::unique_ptr<Synapse>                  initial_synapse;

        // ----- DYNAMIC VARIABLES -----
        float                                     current;
        float                                     potential;
        float                                     trace;

        // ----- FIXED PARAMETERS -----
        float                                     threshold;
        float                                     resting_potential;
        float                                     trace_time_constant;
        float                                     capacitance;
        float                                     leakage_conductance;
        float                                     membrane_time_constant;
        int                                       refractory_period;

        // ----- IMPLEMENTATION PARAMETERS -----
        bool                                      active;
        std::vector<Addon*>                       relevant_addons;
        double                                    previous_spike_time;
        double                                    previous_input_time;
        int                                       neuron_type;
        std::deque<std::string>                   decision_queue;
        std::string                               class_label;
    };

    class Network {

    public:

		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network() :
                verbose(0),
                decision_making(false),
                learning_status(true),
                learning_off_signal(-1),
                max_delay(0),
                asynchronous(false),
                decision_pre_ts(0) {}

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

            int shift = 0;

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
            for (int k=0+shift; k<_numberOfNeurons+shift; k++) {
                neurons.emplace_back(std::make_unique<T>(k, layer_id, 0, 0, std::pair(-1, -1), std::forward<Args>(args)...));

                neuronsInLayer.emplace_back(neurons.size()-1);
            }

            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }

            // building layer structure
            layers.emplace_back(layer{{sublayer{{}, neuronsInLayer, 0}}, neuronsInLayer, layer_id});
            return layers.back();
        }

        // takes in training labels and creates DecisionMaking neurons according to the number of classes present - Decision layer should be the last layer
        template <typename T, typename... Args>
        layer make_decision(std::deque<label> _trainingLabels, int _spike_history_size, int _rejection_threshold, float _timer, std::vector<Addon*> _addons, Args&&... args) {
            training_labels = _trainingLabels;
            decision_making = true;

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
            for (const auto& it: classes_map) {
                neurons.emplace_back(std::make_unique<T>(i+shift, layer_id, 0, 0, std::pair(-1, -1), it.first, std::forward<Args>(args)...));
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
            layers.emplace_back(layer{{sublayer{{}, neuronsInLayer, 0}}, neuronsInLayer, layer_id, false});

            return layers.back();
        }

        // overload for the makeDecision function that takes in a path to a text label file with the format: label_name timestamp
        template <typename T, typename... Args>
        layer make_decision(std::string trainingLabelFilename, int _spike_history_size, int _rejection_threshold, float _timer, std::vector<Addon*> _addons, Args&&... args) {
            DataParser dataParser;
            auto training_labels = dataParser.read_txt_labels(trainingLabelFilename);
            return make_decision<T>(training_labels, _spike_history_size, _rejection_threshold, _timer, _addons, std::forward<Args>(args)...);
        }

        // adds neurons arranged in circles of various radii
        template <typename T, typename... Args>
        layer make_circle(int _numberOfNeurons, std::vector<float> _radii, std::vector<Addon*> _addons, Args&&... args) {
            int shift = 0;

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
            float inv_number_neurons = 1. / _numberOfNeurons;
            for (int i=0; i<static_cast<int>(_radii.size()); i++) {
                std::vector<std::size_t> neuronsInSublayer;
                for (int k=0+shift; k<_numberOfNeurons+shift; k++) {
                    // we round the coordinates because the precision isn't needed and xy_coordinates are int
                    int u = static_cast<int>(std::round(_radii[i] * std::cos(2*M_PI*(k-shift) * inv_number_neurons)));
                    int v = static_cast<int>(std::round(_radii[i] * std::sin(2*M_PI*(k-shift) * inv_number_neurons)));
                    neurons.emplace_back(std::make_unique<T>(k+counter, layer_id, i, 0, std::pair(u, v), std::forward<Args>(args)...));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);
                }
                sublayers.emplace_back(sublayer{{}, neuronsInSublayer, i});

                // to shift the neuron IDs with the sublayers
                counter += _numberOfNeurons;
            }

            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }

            // building layer structure
            layers.emplace_back(layer{sublayers, neuronsInLayer, layer_id});
            return layers.back();
        }

        // adds a 2 dimensional grid of neurons
        template <typename T, typename... Args>
        layer make_grid(int gridW, int gridH, int _sublayerNumber, std::vector<Addon*> _addons, Args&&... args) {

            // find number of neurons to build
            int numberOfNeurons = gridW * gridH;

            int shift = 0;

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
                for (int k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(std::make_unique<T>(k+counter, layer_id, i, 0, std::pair(x, y), std::forward<Args>(args)...));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);

                    x += 1;
                    if (x == gridW) {
                        y += 1;
                        x = 0;
                    }
                }
                sublayers.emplace_back(sublayer{{}, neuronsInSublayer, i});

                // to shift the neuron IDs with the sublayers
                counter += numberOfNeurons;
            }

            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }

            // building layer structure
            layers.emplace_back(layer{sublayers, neuronsInLayer, layer_id, true, gridW, gridH});
            return layers.back();
        }

        // overloading the makeGrid function to automatically generate a 2D layer according to the previous layer size
        template <typename T, typename... Args>
        layer make_grid(layer presynapticLayer, int _sublayerNumber, int _kernelSize, int _stride, std::vector<Addon*> _addons, Args&&... args) {
            // finding the number of receptive fields
            float inv_stride = 1./static_cast<float>(_stride);
            int newWidth = std::ceil((presynapticLayer.width - _kernelSize + 1) * inv_stride);
            int newHeight = std::ceil((presynapticLayer.height - _kernelSize + 1) * inv_stride);

            int trimmedColumns = std::abs(newWidth - std::ceil((presynapticLayer.width - _stride + 1) * inv_stride));
            int trimmedRows = std::abs(newHeight - std::ceil((presynapticLayer.height - _stride + 1) * inv_stride));

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

            int shift = 0;

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
                for (int k=0+shift; k<numberOfNeurons+shift; k++) {
                    neurons.emplace_back(std::make_unique<T>(k+counter, layer_id, i, 0, std::pair(x, y), std::forward<Args>(args)...));
                    neuronsInSublayer.emplace_back(neurons.size()-1);
                    neuronsInLayer.emplace_back(neurons.size()-1);

                    x += 1;
                    if (x == newWidth) {
                        y += 1;
                        x = 0;
                    }
                }
                sublayers.emplace_back(sublayer{{}, neuronsInSublayer, i});

                // to shift the neuron IDs with the sublayers
                counter += numberOfNeurons;
            }

            // looping through addons and adding the layer to the neuron mask
            for (auto& addon: _addons) {
                addon->activate_for(neuronsInLayer);
            }

            // building layer structure
            layers.emplace_back(layer{sublayers, neuronsInLayer, layer_id, true, newWidth, newHeight, _kernelSize, _stride});
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
        template <typename T = Synapse, typename F, typename... Args>
        void convolution(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int connection_ratio, Args&&... args) {
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
                range = postsynapticLayer.kernel_size - std::ceil(postsynapticLayer.kernel_size * 0.5) - 0.5;
            }
            else {
                range = postsynapticLayer.kernel_size - std::ceil(postsynapticLayer.kernel_size * 0.5);
            }

            // number of neurons surrounding the center
            int mooreNeighbors = (2*range + 1) * (2*range + 1);

            int number_of_connections =  static_cast<int>(postsynapticLayer.sublayers.size()) * static_cast<int>(presynapticLayer.sublayers.size()) * static_cast<int>(postsynapticLayer.sublayers[0].neurons.size()) * mooreNeighbors * number_of_synapses;

            auto successful_connections = find_successful_connections(connection_ratio, number_of_connections);

            // looping through the newly created layer to connect them to the correct receptive fields
            int conn_idx = 0;
            for (auto& convSub: postsynapticLayer.sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    std::vector<receptive_field> rf;

                    // initialising window on the first center coordinates
                    std::pair centerCoordinates((postsynapticLayer.kernel_size-1) * 0.5, (postsynapticLayer.kernel_size-1) * 0.5);

                    // number of neurons = number of receptive fields in the presynaptic Layer
                    int rf_id = 0;
                    for (auto& n: convSub.neurons) {

                        std::vector<size_t> rf_neurons;
                        // finding the coordinates for the presynaptic neurons in each receptive field
                        for (auto i=0; i<mooreNeighbors; i++) {
                            int x = centerCoordinates.first + ((i % postsynapticLayer.kernel_size) - range);
                            int y = centerCoordinates.second + ((i / postsynapticLayer.kernel_size) - range);

                            // 2D to 1D mapping to get the index from x y coordinates
                            int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;

                            // changing the neuron's receptive field id from the default
                            neurons[idx]->set_rf_id(rf_id);
                            rf_neurons.emplace_back(static_cast<size_t>(idx));

                            // connecting neurons from the presynaptic layer to the convolutional one, depedning on the number of synapses
                            for (auto i=0; i<number_of_synapses; i++) {
                                // calculating weights and delays according to the provided distribution
                                const std::pair weight_delay = lambdaFunction(x, y, convSub.id);

                                // creating a synapse between the neurons
                                neurons[idx]->make_synapse<T>(neurons[n].get(), weight_delay.first, weight_delay.second, std::forward<Args>(args)...);

                                // to shift the network runtime by the maximum delay in the clock mode
                                max_delay = std::max(max_delay, weight_delay.second);
                                conn_idx++;
                            }
                        }

                        rf.emplace_back(receptive_field{rf_neurons, rf_id});

                        // finding the coordinates for the center of each receptive field
                        centerCoordinates.first += postsynapticLayer.stride;
                        if (centerCoordinates.first >= presynapticLayer.width - trimmedColumns) {
                            centerCoordinates.first = (postsynapticLayer.kernel_size-1) * 0.5;
                            centerCoordinates.second += postsynapticLayer.stride;
                        }

                        // updating receptive field index
                        rf_id++;
                    }

                    layers[presynapticLayer.id].sublayers[preSub.id].receptive_fields = rf;
                    sublayershift += preSub.neurons.size();
                }
            }
        }
        
        // connecting a subsampled layer to its previous layer. Last set of paramaters are to characterize the synapses. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T = Synapse, typename F, typename... Args>
        void pooling(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int connection_ratio, Args&&... args) {
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
                range = lcd - std::ceil(lcd * 0.5) - 0.5;
                // if size of kernel is an odd number
            } else {
                // finding range to calculate a moore neighborhood
                range = lcd - std::ceil(lcd * 0.5);
            }

            // number of neurons surrounding the center
            int mooreNeighbors = (2*range + 1) * (2*range + 1);

            int number_of_connections = static_cast<int>(presynapticLayer.sublayers.size()) * static_cast<int>(postsynapticLayer.sublayers[0].neurons.size()) * mooreNeighbors * number_of_synapses;

            auto successful_connections = find_successful_connections(connection_ratio, number_of_connections);

            int conn_idx = 0;
            for (auto& poolSub: postsynapticLayer.sublayers) {
                int sublayershift = 0;
                for (auto& preSub: presynapticLayer.sublayers) {
                    std::vector<receptive_field> rf;

                    if (poolSub.id == preSub.id) {

                        // initialising window on the first center coordinates
                        std::pair centerCoordinates((lcd-1) * 0.5, (lcd-1) * 0.5);

                        // number of neurons = number of receptive fields in the presynaptic Layer
                        int rf_id = 0;
                        for (auto& n: poolSub.neurons) {
                            std::vector<size_t> rf_neurons;
                            // finding the coordinates for the presynaptic neurons in each receptive field
                            for (auto i=0; i<mooreNeighbors; i++) {

                                int x = centerCoordinates.first + ((i % lcd) - range);
                                int y = centerCoordinates.second + ((i / lcd) - range);

                                // 2D to 1D mapping to get the index from x y coordinates
                                int idx = (x + presynapticLayer.width * y) + layershift + sublayershift;

                                // changing the neuron's receptive field coordinates from the default
                                neurons[idx]->set_rf_id(rf_id);
                                rf_neurons.emplace_back(static_cast<size_t>(idx));

                                for (auto i=0; i<number_of_synapses; i++) {
                                    // calculating weights and delays according to the provided distribution
                                    const std::pair weight_delay = lambdaFunction(x, y, poolSub.id);

                                    // connecting neurons from the presynaptic layer to the convolutional one
                                    neurons[idx]->make_synapse<T>(neurons[n].get(), weight_delay.first, weight_delay.second, std::forward<Args>(args)...);

                                    // to shift the network runtime by the maximum delay in the clock mode
                                    max_delay = std::max(max_delay, weight_delay.second);
                                    conn_idx++;
                                }
                            }

                            rf.emplace_back(receptive_field{rf_neurons, rf_id});

                            // finding the coordinates for the center of each receptive field
                            centerCoordinates.first += lcd;
                            if (centerCoordinates.first >= presynapticLayer.width) {
                                centerCoordinates.first = (lcd-1) * 0.5;
                                centerCoordinates.second += lcd;
                            }

                            // updating receptive field indices
                            rf_id ++;
                        }
                    }
                    layers[presynapticLayer.id].sublayers[preSub.id].receptive_fields = rf;
                    sublayershift += preSub.neurons.size();
                }
            }
        }
        
        // interconnecting a layer (feedforward, feedback and self-excitation) with randomised weights and delays. lambdaFunction: Takes in one of the classes inside the randomDistributions folder to define a distribution for the weights.
        template <typename T = Synapse, typename F, typename... Args>
        void reservoir(layer reservoirLayer, int number_of_synapses, F&& lambdaFunction, int feedforward_connection_ratio, int feedback_connection_ratio, int self_excitation_connection_ratio, Args&&... args) {

            int number_of_feedforward = (static_cast<int>(reservoirLayer.neurons.size()) - 1) * (static_cast<int>(reservoirLayer.neurons.size()) - 1) * number_of_synapses;
            auto successful_feedforward = find_successful_connections(feedforward_connection_ratio, number_of_feedforward);

            int number_of_feedback = (static_cast<int>(reservoirLayer.neurons.size()) - 1) * (static_cast<int>(reservoirLayer.neurons.size()) - 1) * number_of_synapses;
            auto successful_feedback = find_successful_connections(feedback_connection_ratio, number_of_feedback);

            int number_of_self_excitation = static_cast<int>(reservoirLayer.neurons.size()) * number_of_synapses;
            auto successful_self_excitation = find_successful_connections(self_excitation_connection_ratio, number_of_self_excitation);

            int idx = 0;
            int idx_se = 0;
            // connecting the reservoir
            for (auto pre: reservoirLayer.neurons) {
                for (auto post: reservoirLayer.neurons) {
                    for (auto i=0; i<number_of_synapses; i++) {
                        // calculating weights and delays according to the provided distribution
                        const std::pair weight_delay = lambdaFunction(0, 0, 0);

                        // self-excitation connection_ratio
                        if (pre == post) {
                            if (successful_self_excitation[idx_se]) {
                                neurons[pre]->make_synapse<T>(neurons[post].get(), weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                            }
                            idx_se++;
                        } else {
                            // feedforward connection_ratio
                            if (successful_feedforward[idx]) {
                                neurons[pre]->make_synapse<T>(neurons[post].get(), weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                            }

                            // feedback connection_ratio
                            if (successful_feedback[idx]) {
                                neurons[post]->make_synapse<T>(neurons[pre].get(), weight_delay.first, weight_delay.first, std::forward<Args>(args)...);
                            }
                            idx++;
                        }
                    }
                }
            }
        }
        
		// connecting two layers according to a weight matrix vector of vectors and a delays matrix vector of vectors (columns for input and rows for output)
        template <typename T = Synapse, typename... Args>
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
                                    neurons[preNeuron]->make_synapse<T>(neurons[postNeuron].get(), weights[preCounter][postCounter], delays[preCounter][postCounter], std::forward<Args>(args)...);
                                }
                            }

                            // to shift the network runtime by the maximum delay in the clock mode
                            max_delay = std::max(max_delay, delays[preCounter][postCounter]);

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
        template <typename T = Synapse, typename F, typename... Args>
        void one_to_one(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int connection_ratio, Args&&... args) {
            // error handling
            if (presynapticLayer.neurons.size() != postsynapticLayer.neurons.size() && presynapticLayer.width == postsynapticLayer.width && presynapticLayer.height == postsynapticLayer.height) {
                throw std::logic_error("The presynaptic and postsynaptic layers do not have the same number of neurons. Cannot do a one-to-one connection");
            }

            int number_of_connections = static_cast<int>(presynapticLayer.neurons.size()) * number_of_synapses;
            auto successful_connections = find_successful_connections(connection_ratio, number_of_connections);
            int idx = 0;
            for (int preSubIdx=0; preSubIdx<static_cast<int>(presynapticLayer.sublayers.size()); preSubIdx++) {
                for (int preNeuronIdx=0; preNeuronIdx<static_cast<int>(presynapticLayer.sublayers[preSubIdx].neurons.size()); preNeuronIdx++) {
                    for (int postSubIdx=0; postSubIdx<static_cast<int>(postsynapticLayer.sublayers.size()); postSubIdx++) {
                        for (int postNeuronIdx=0; postNeuronIdx<static_cast<int>(postsynapticLayer.sublayers[postSubIdx].neurons.size()); postNeuronIdx++) {
                            if (preNeuronIdx == postNeuronIdx) {
                                for (int i=0; i<number_of_synapses; i++) {

                                    if (successful_connections[idx]) {

                                        const std::pair weight_delay = lambdaFunction(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->get_xy_coordinates().first, neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]]->get_xy_coordinates().second, postsynapticLayer.sublayers[postSubIdx].id);
                                    neurons[presynapticLayer.sublayers[preSubIdx].neurons[preNeuronIdx]]->make_synapse<T>(neurons[postsynapticLayer.sublayers[postSubIdx].neurons[postNeuronIdx]].get(), weight_delay.first, weight_delay.second, std::forward<Args>(args)...);

                                        // to shift the network runtime by the maximum delay in the clock mode
                                        max_delay = std::max(max_delay, weight_delay.second);
                                    }

                                    idx++;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // all to all connection between layers. lambdaFunction: Takes in either a lambda function (operating on x, y and the sublayer depth) or one of the classes inside the randomDistributions folder to define a distribution for the weights and delays
        template <typename T = Synapse, typename F, typename... Args>
        void all_to_all(layer presynapticLayer, layer postsynapticLayer, int number_of_synapses, F&& lambdaFunction, int connection_ratio, Args&&... args) {

            int number_of_connections = static_cast<int>(presynapticLayer.neurons.size()) * static_cast<int>(postsynapticLayer.neurons.size()) * number_of_synapses;
            auto successful_connections = find_successful_connections(connection_ratio, number_of_connections);

            int idx = 0;
            for (auto& preSub: presynapticLayer.sublayers) {
                for (auto& preNeuron: preSub.neurons) {
                    for (auto& postSub: postsynapticLayer.sublayers) {
                        for (auto& postNeuron: postSub.neurons) {
                            for (auto i=0; i<number_of_synapses; i++) {

                                if (successful_connections[idx]) {
                                    const std::pair weight_delay = lambdaFunction(neurons[postNeuron]->get_xy_coordinates().first, neurons[postNeuron]->get_xy_coordinates().second, postSub.id);

                                    neurons[preNeuron]->make_synapse<T>(neurons[postNeuron].get(), weight_delay.first, weight_delay.second, std::forward<Args>(args)...);

                                    // to shift the network runtime by the maximum delay in the clock mode
                                    max_delay = std::max(max_delay, weight_delay.second);
                                }

                                idx++;
                            }
                        }
                    }
                }
            }
        }
        
        // interconnecting a layer with soft winner-takes-all synapses, using negative weights
        template <typename T = Synapse, typename F, typename... Args>
        void lateral_inhibition(layer current_layer, int number_of_synapses, F&& lambdaFunction, int connection_ratio, Args&&... args) {

            size_t number_of_connections = 0;
            auto& l = layers[current_layer.id];
            auto& s = l.sublayers[0];

            if (s.receptive_fields.empty()) {
                size_t intra_connections = (s.neurons.size() - 1) * s.neurons.size() * l.sublayers.size() * number_of_synapses;
                size_t inter_connections = s.neurons.size() * s.neurons.size() * (l.sublayers.size() - 1) * l.sublayers.size() * number_of_synapses;
                number_of_connections = intra_connections + inter_connections;
            } else {
                size_t intra_connections = (s.receptive_fields[0].neurons.size() - 1) * s.neurons.size() * l.sublayers.size() * number_of_synapses;
                size_t inter_connections = s.receptive_fields[0].neurons.size() * s.neurons.size() * (l.sublayers.size() - 1) * l.sublayers.size() * number_of_synapses;
                number_of_connections = intra_connections + inter_connections;
            }

            auto successful_connections = find_successful_connections(connection_ratio, static_cast<int>(number_of_connections));

            int idx = 0;
            for (auto& sub: l.sublayers) {
                // intra-sublayer soft WTA
                for (auto& preNeurons: sub.neurons) {
                    for (auto& postNeurons: sub.neurons) {
                        if (preNeurons != postNeurons && neurons[preNeurons]->get_rf_id() == neurons[postNeurons]->get_rf_id()) {
                            for (auto i=0; i<number_of_synapses; i++) {
                                const std::pair weight_delay = lambdaFunction(0, 0, 0);
                                neurons[preNeurons]->make_synapse<T>(neurons[postNeurons].get(), -1*std::abs(weight_delay.first), weight_delay.second, std::forward<Args>(args)...);
                                idx++;
                            }
                        }
                    }
                }

                // inter-sublayer soft WTA
                for (auto& subToInhibit: l.sublayers) {
                    if (sub.id != subToInhibit.id) {
                        for (auto& preNeurons: sub.neurons) {
                            for (auto& postNeurons: subToInhibit.neurons) {
                                if (neurons[preNeurons]->get_rf_id() == neurons[postNeurons]->get_rf_id()) {
                                    for (auto i=0; i<number_of_synapses; i++) {
                                        const std::pair weight_delay = lambdaFunction(0, 0, 0);
                                        neurons[preNeurons]->make_synapse<T>(neurons[postNeurons].get(), -1*std::abs(weight_delay.first), weight_delay.second, std::forward<Args>(args)...);
                                        idx++;
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
        void inject_spike(int neuronIndex, double timestamp, spike_type type = spike_type::initial) {
            spike_queue.emplace(neurons.at(neuronIndex)->receive_external_input(timestamp, type, neuronIndex, -1, 1, 0));
        }

        // adding spikes predicted by the asynchronous network (timestep = 0) for synaptic integration
        void inject_predicted_spike(spike s, spike_type stype) {
            // remove old spike
            predicted_spikes.erase(std::remove_if(
                                                  predicted_spikes.begin(),
                                                  predicted_spikes.end(),[&](spike oldSpike) { return oldSpike.propagation_synapse == s.propagation_synapse; }),
                                   predicted_spikes.end());

            // change type of new spike
            s.type = stype;

            // insert the new spike in the correct place
            predicted_spikes.insert(
                std::upper_bound(predicted_spikes.begin(), predicted_spikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }

        // add spikes from an event vector to the network
        void inject_input(const std::vector<event>& data, spike_type type = spike_type::initial) {
            // error handling
            if (layers.empty()) {
                throw std::logic_error("add a layer of neurons before injecting spikes");
            }

            for (auto& event: data) {
                // one dimensional data
                if (event.x == -1) {
                    inject_spike(event.neuron_id, event.timestamp, type); // the neuron_id can represent the sublayer so no need to account for it
                // two dimensional data
                } else {
                    // 2D to 1D mapping for the first layer of the network
                    int sublayer_shift = 0;
                    for (auto sub: layers[0].sublayers) { // there is no neuron_id so if there's more than one initial sublayer we inject the spike in all of them
                        int idx = (event.x + layers[0].width * event.y) + sublayer_shift;
                        inject_spike(idx, event.timestamp);
                        sublayer_shift += sub.neurons.size();
                    }
                }
            }
        }

        // add a poissonian spike train to the initial spike vector
        void poisson_spike_generator(int neuronIndex, double timestamp, float rate, float timestep, float duration) {
            // calculating number of spikes
            int spike_number = std::floor(duration/timestep);

            // initialising the random engine
            std::random_device                      device;
            std::mt19937                            random_engine(device());
            std::uniform_real_distribution<double>  distribution(0.0,1.0);

            std::vector<double> inter_spike_intervals;
            double inv_rate = 1. / rate;
            // generating uniformly distributed random numbers
            for (auto i = 0; i < spike_number; ++i) {
                inter_spike_intervals.emplace_back((- std::log(distribution(random_engine)) * inv_rate) * 1000);
            }

            // computing spike times from inter-spike intervals
            std::vector<double> spike_times(inter_spike_intervals.size(), 0.0);
            spike_times[0] = inter_spike_intervals[0];
            for (int i=1; i<static_cast<int>(spike_times.size()); i++) {
                spike_times[i] = spike_times[i-1] + inter_spike_intervals[i];
            }
            std::transform(spike_times.begin(), spike_times.end(), spike_times.begin(), [&](double& st){return st*0.001+timestamp;});

            // injecting into the initial spike vector
            for (auto& spike_time: spike_times) {
                spike_queue.emplace(neurons[neuronIndex]->receive_external_input(spike_time, spike_type::initial, neuronIndex, -1, 1, 0));
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
        void run(double _runtime, float _timestep=0, bool classification=false) {
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
                std::cout << "This instance is for classification only. No learning is being done." << std::endl;

                if (th_addon) {
                    th_addon->reset();
                }

                // can now propagate to decision-making layer if present
                if (decision_making) {
                    layers[decision.layer_number].active = true;
                }

                // during a classification run, labels the neurons if a decision-making layer was used
                prepare_decision_making();
            }

            std::mutex sync;
            if (th_addon) {
                sync.lock();
            }

            std::atomic_bool running(true);
            std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

                if (_timestep == 0) {
                    async_run_helper(&running, classification);
                } else {
                    sync_run_helper(&running, _runtime, _timestep, classification);
                }

                std::chrono::duration<float> elapsed_seconds = std::chrono::system_clock::now()-start;
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s" << std::endl;
                }

                for (auto& addon: addons) {
                    addon->on_completed(this);
                }
            });

            if (th_addon) {
                th_addon->begin(this, &sync);
                running.store(false, std::memory_order_relaxed);
            }

            spikeManager.join();
        }

        // running through the network asynchronously if timestep = 0 and synchronously otherwise. This method takes in a vector of inputs from the read_txt_data method
        void run_data(const std::vector<event>& trainingData, float _timestep=0, const std::vector<event>& testData={}) {

            if (_timestep == 0) {
                asynchronous = true;
            }

            for (auto& n: neurons) {
                n->initialisation(this);
            }

            for (auto& addon: addons) {
                addon->on_start(this);
            }

            std::mutex sync;
            if (th_addon) {
                sync.lock();
            }

            std::atomic_bool running(true);
            std::thread spikeManager([&] {
                sync.lock();
                sync.unlock();

                // importing training data and running the network through the data
                inject_input(trainingData);

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                if (verbose != 0) {
                    std::cout << "Running training instance..." << std::endl;
                }

                if (_timestep == 0) {
                    async_run_helper(&running, false);
                } else {
                    sync_run_helper(&running, trainingData.back().timestamp+max_delay, _timestep, false);
                }

                std::chrono::duration<float> elapsed_seconds = std::chrono::system_clock::now()-start;
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s" << std::endl;
                }

                // importing test data and running it through the network for classification
                if (!testData.empty()) {

                    // can now propagate to decision-making layer if present
                    if (decision_making) {
                        layers[decision.layer_number].active = true;
                    }

                    learning_status = false;
                    reset_network(false);

                    prepare_decision_making();

                    inject_input(testData);

                    for (auto& addon: addons) {
                        addon->on_predict(this);
                    }

                    if (th_addon) {
                        th_addon->reset();
                    }

                    start = std::chrono::system_clock::now();
                    if (verbose != 0) {
                        std::cout << "Running classification instance..." << std::endl;
                    }

                    if (_timestep == 0) {
                        async_run_helper(&running, true);
                    } else {
                        sync_run_helper(&running, testData.back().timestamp+max_delay, _timestep, true);
                    }

                    elapsed_seconds = std::chrono::system_clock::now()-start;
                    if (verbose != 0) {
                        std::cout << "it took " << elapsed_seconds.count() << "s" << std::endl;
                    }
                }

                for (auto& addon: addons) {
                    addon->on_completed(this);
                }
            });

            if (th_addon) {
                th_addon->begin(this, &sync);
                running.store(false, std::memory_order_relaxed);
            }

            spikeManager.join();
        }

        // running asynchronously through one .es file - relies on the sepia header
        void run_es(const std::string filename, bool classification=false, uint64_t t_max=UINT64_MAX, uint64_t t_min=0, int polarity=2, uint16_t x_max=UINT16_MAX, uint16_t x_min=0, uint16_t y_max=UINT16_MAX, uint16_t y_min=0) {
            asynchronous = true;

            for (auto& n: neurons) {
                n->initialisation(this);
            }

            for (auto& addon: addons) {
                addon->on_start(this);
            }

            if (classification) {
                std::cout << "This instance is for classification only. No learning is being done." << std::endl;

                if (th_addon) {
                    th_addon->reset();
                }

                // can now propagate to decision-making layer if present
                if (decision_making) {
                    layers[decision.layer_number].active = true;
                }

                // during a classification run, labels the neurons if a decision-making layer was used
                prepare_decision_making();
            }

            std::mutex sync;
            if (th_addon) {
                sync.lock();
            }

            std::atomic_bool running(true);
            auto loop = std::thread([&]() {
                sync.lock();
                sync.unlock();

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

                auto header = sepia::read_header(sepia::filename_to_ifstream(filename));

                if (header.event_stream_type == sepia::type::dvs) {
                    sepia::join_observable<sepia::type::dvs>(
                                                            sepia::filename_to_ifstream(filename),
                                                            [&](sepia::dvs_event event) {
                                                                // stopping the event collection beyond a certain temporal threshold
                                                                if (event.t > t_max || !running.load(std::memory_order_relaxed)) {
                                                                    throw sepia::end_of_file();
                                                                }

                                                                // temporal crop and spatial crop and filtering the selected polarity
                                                                if (polarity == 2) {
                                                                    if (event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                        es_run_helper(static_cast<double>(event.t), event.x, event.y);
                                                                    }
                                                                } else if (polarity == 0 || polarity == 1) {
                                                                    if (static_cast<int>(event.is_increase) == polarity && event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                        es_run_helper(static_cast<double>(event.t), event.x, event.y);
                                                                    }
                                                                } else {
                                                                    throw std::logic_error("polarity is 0 for OFF events, 1 for ON events and 2 for both");
                                                                }
                                                            });
                } else if (header.event_stream_type == sepia::type::atis) {
                    sepia::join_observable<sepia::type::atis>(
                                                              sepia::filename_to_ifstream(filename),
                                                              [&](sepia::atis_event event) {
                                                                  // stopping the event collection beyond a certain temporal threshold
                                                                  if (event.t > t_max || !running.load(std::memory_order_relaxed)) {
                                                                      throw sepia::end_of_file();
                                                                  }

                                                                  if (polarity == 2) {
                                                                      // filtering out gray level events, the selected polarity, temporal crop and spatial crop
                                                                      if (!event.is_threshold_crossing && event.polarity == static_cast<int>(polarity) && event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                          es_run_helper(static_cast<double>(event.t), event.x, event.y);
                                                                      }
                                                                  } else if (polarity == 0 || polarity == 1) {
                                                                      // filtering out gray level events, the selected polarity, temporal crop and spatial crop
                                                                      if (!event.is_threshold_crossing && static_cast<int>(event.polarity) == polarity && event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                          es_run_helper(static_cast<double>(event.t), event.x, event.y);
                                                                      }
                                                                  } else {
                                                                      throw std::logic_error("polarity is 0 for OFF events, 1 for ON events and 2 for both");
                                                                  }
                                                              });
                } else {
                    // throw error
                    throw std::logic_error("unknown header type");
                }

                // going through any leftover spikes after the last event is propagated
                async_run_helper(&running, false, true);

                std::chrono::duration<float> elapsed_seconds = std::chrono::system_clock::now()-start;
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
                }

                for (auto& addon: addons) {
                    addon->on_completed(this);
                }
            });
            if (th_addon) {
                th_addon->begin(this, &sync);
                running.store(false, std::memory_order_relaxed);
            }

            loop.join();
        }

        // running asynchronously through a database of .es files - relies on the sepia header
        void run_database(const std::vector<std::string>& training_database, const std::vector<std::string>& testing_database={}, uint64_t t_max=UINT64_MAX, uint64_t t_min=0, int polarity=2, uint16_t x_max=UINT16_MAX, uint16_t x_min=0, uint16_t y_max=UINT16_MAX, uint16_t y_min=0) {

            asynchronous = true;

            for (auto& n: neurons) {
                n->initialisation(this);
            }

            for (auto& addon: addons) {
                addon->on_start(this);
            }

            std::mutex sync;
            if (th_addon) {
                sync.lock();
            }

            std::atomic_bool running(true);
            auto loop = std::thread([&]() {
                sync.lock();
                sync.unlock();

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                if (verbose != 0) {
                    std::cout << "Running training instance..." << std::endl;
                }

                // loop through each .es file in the training database
                auto idx = 0;
                for (auto filename : training_database) {
                    
                    if (verbose == 2) {
                        std::cout << filename << std::endl;
                    }

                    if (!running.load(std::memory_order_relaxed)) {
                        break;
                    }

                    auto header = sepia::read_header(sepia::filename_to_ifstream(filename));

                    // get the current label for the database - one label per pattern
                    if (!training_labels.empty()) {
                        current_label = training_labels[idx].name;
                    }

                    if (header.event_stream_type == sepia::type::dvs) {

                        sepia::join_observable<sepia::type::dvs>(
                                                                 sepia::filename_to_ifstream(filename),
                                                                 [&](sepia::dvs_event event) {
                                                                     // stopping the event collection beyond a certain temporal threshold
                                                                     if (event.t > t_max || !running.load(std::memory_order_relaxed)) {
                                                                         throw sepia::end_of_file();
                                                                     }

                                                                     // temporal crop and spatial crop and polarity selection
                                                                     if (polarity == 2) {
                                                                         if (event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                            es_run_helper(static_cast<double>(event.t), static_cast<int>(event.x), static_cast<int>(event.y));
                                                                         }
                                                                     } else if (polarity == 0 || polarity == 1) {
                                                                         if (static_cast<int>(event.is_increase) == polarity && event.t >= t_min && event.x >= x_min && event.x <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                            es_run_helper(static_cast<double>(event.t), static_cast<int>(event.x), static_cast<int>(event.y));
                                                                         }
                                                                     } else {
                                                                         throw std::logic_error("polarity is 0 for OFF events, 1 for ON events and 2 for both");
                                                                     }
                                                                 });

                    } else if (header.event_stream_type == sepia::type::atis) {
                        sepia::join_observable<sepia::type::atis>(
                                                                  sepia::filename_to_ifstream(filename),
                                                                  [&](sepia::atis_event event) {
                                                                      // stopping the event collection beyond a certain temporal threshold
                                                                      if (event.t > t_max || !running.load(std::memory_order_relaxed)) {
                                                                          throw sepia::end_of_file();
                                                                      }

                                                                      // filtering out gray level events, temporal crop and spatial crop and polarity selection
                                                                      if (polarity == 2) {
                                                                          if (!event.is_threshold_crossing && event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                              es_run_helper(static_cast<double>(event.t), static_cast<int>(event.x), static_cast<int>(event.y));
                                                                          }
                                                                      } else if (polarity == 0 || polarity == 1) {
                                                                          if (!event.is_threshold_crossing && static_cast<int>(event.polarity) == polarity && event.t >= t_min && event.x >= x_min && event.x <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                              es_run_helper(static_cast<double>(event.t), static_cast<int>(event.x), static_cast<int>(event.y));
                                                                          }
                                                                      } else {
                                                                          throw std::logic_error("polarity is 0 for OFF events, 1 for ON events and 2 for both");
                                                                      }
                                                                  });
                    } else {
                        // throw error
                        throw std::logic_error("unknown header type");
                    }

                    // going through any leftover spikes after the last event is propagated
                    async_run_helper(&running, false, true);

                    // sending the on_pattern_end addon message
                    for (auto& addon: addons) {
                        addon->on_pattern_end(this);
                    }
                    
                    idx++;

                    reset_network(false);
                }

                std::chrono::duration<float> elapsed_seconds = std::chrono::system_clock::now()-start;
                if (verbose != 0) {
                    std::cout << "it took " << elapsed_seconds.count() << "s" << std::endl;
                }

                if (!testing_database.empty()) {

                    // can now propagate to decision-making layer if present
                    if (decision_making) {
                        layers[decision.layer_number].active = true;
                    }

                    learning_status = false;
                    reset_network(false);

                    prepare_decision_making();

                    for (auto& addon: addons) {
                        addon->on_predict(this);
                    }

                    if (th_addon) {
                        th_addon->reset();
                    }

                    start = std::chrono::system_clock::now();
                    if (verbose != 0) {
                        std::cout << "Running classification instance..." << std::endl;
                    }

                    // loop through each .es file in the testing database
                    for (auto filename : testing_database) {
                    
                        if (verbose == 2) {
                            std::cout << filename << std::endl;
                        }

                        if (!running.load(std::memory_order_relaxed)) {
                            break;
                        }

                        auto header = sepia::read_header(sepia::filename_to_ifstream(filename));

                        double final_t = 0;
                        if (header.event_stream_type == sepia::type::dvs) {
                            sepia::join_observable<sepia::type::dvs>(
                                                                     sepia::filename_to_ifstream(filename),
                                                                     [&](sepia::dvs_event event) {
                                                                         // stopping the event collection beyond a certain temporal threshold
                                                                         if (event.t > t_max || !running.load(std::memory_order_relaxed)) {
                                                                             throw sepia::end_of_file();
                                                                         }

                                                                         // temporal crop and spatial crop and polarity selection
                                                                         if (polarity == 2) {
                                                                             if (event.t >= t_min && event.x >= x_min && event.y <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                                 final_t = static_cast<double>(event.t);
                                                                                 es_run_helper(final_t, static_cast<int>(event.x), static_cast<int>(event.y), true);
                                                                             }
                                                                         } else if (polarity == 0 || polarity == 1) {
                                                                             if (static_cast<int>(event.is_increase) == polarity && event.t >= t_min && event.x >= x_min && event.x <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                                 final_t = static_cast<double>(event.t);
                                                                                 es_run_helper(final_t, static_cast<int>(event.x), static_cast<int>(event.y), true);
                                                                             }
                                                                         } else {
                                                                             throw std::logic_error("polarity is 0 for OFF events, 1 for ON events and 2 for both");
                                                                         }
                                                                     });
                        } else if (header.event_stream_type == sepia::type::atis) {
                            sepia::join_observable<sepia::type::atis>(
                                                                      sepia::filename_to_ifstream(filename),
                                                                      [&](sepia::atis_event event) {
                                                                          // stopping the event collection beyond a certain temporal threshold
                                                                          if (event.t > t_max || !running.load(std::memory_order_relaxed)) {
                                                                              throw sepia::end_of_file();
                                                                          }

                                                                          // filtering out gray level events, temporal crop and spatial crop and polarity selection
                                                                          if (polarity == 2) {
                                                                              if (!event.is_threshold_crossing && event.t >= t_min && event.x >= x_min && event.x <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                                  final_t = static_cast<double>(event.t);
                                                                                  es_run_helper(final_t, static_cast<int>(event.x), static_cast<int>(event.y), true);
                                                                              }
                                                                          } else if (polarity == 0 || polarity == 1) {
                                                                              if (!event.is_threshold_crossing && static_cast<int>(event.polarity) == polarity && event.t >= t_min && event.x >= x_min && event.x <= x_max && event.y >= y_min && event.y <= y_max) {
                                                                                  final_t = static_cast<double>(event.t);
                                                                                  es_run_helper(final_t, static_cast<int>(event.x), static_cast<int>(event.y), true);
                                                                              }
                                                                          } else {
                                                                              throw std::logic_error("polarity is 0 for OFF events, 1 for ON events and 2 for both");
                                                                          }
                                                                      });
                        } else {
                            // throw error
                            throw std::logic_error("unknown header type");
                        }

                        // going through any leftover spikes after the last event is propagated
                        async_run_helper(&running, true, true);

                        // sending the on_pattern_end addon message
                        for (auto& addon: addons) {
                            addon->on_pattern_end(this);
                        }
                        
                        if (decision_making && decision.timer == 0) {
                            choose_winner_eof(final_t, 0);
                        } else if (decision_making && decision.timer > 0) {
                            // sending an eof signal when a decision timer is used in order to be handle the fact that we can have multiple classifications per pattern
                            for (auto& addon: addons) {
                                addon->decision_failed(final_t, this);
                            }
                        }

                        reset_network(false);
                    }

                    elapsed_seconds = std::chrono::system_clock::now()-start;
                    if (verbose != 0) {
                        std::cout << "it took " << elapsed_seconds.count() << "s" << std::endl;
                    }
                }

                for (auto& addon: addons) {
                    addon->on_completed(this);
                }
            });

            if (th_addon) {
                th_addon->begin(this, &sync);
                running.store(false, std::memory_order_relaxed);
            }

            loop.join();
        }

        // reset the network back to the initial conditions without changing the network build
        void reset_network(bool clear_addons=true) {
            decision_pre_ts = 0;

            for (auto& n: neurons) {
                n->reset_neuron(this, clear_addons);
            }

            if (th_addon) {
                th_addon->reset();
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

        std::unique_ptr<MainAddon>& get_main_thread_addon() {
            return th_addon;
        }

        void set_main_thread_addon(MainAddon* new_thAddon) {
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

        bool is_asynchronous() const {
            return asynchronous;
        }

        int get_verbose() const {
            return verbose;
        }

        decision_heuristics& get_decision_parameters() {
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

        void es_run_helper(double t, int x, int y, bool classification=false) {

            // 1. find neuron corresponding to the event coordinates through 2D to 1D mapping
            int idx = (x + layers[0].width * y);

            // 2. make sure the neuron is actually from the input layer
            if (neurons.at(idx)->get_layer_id() != 0) {
                throw std::logic_error("the input layer does not contain enough neurons.");
            }

            // 3. start the spike propagation workflow

            // if spike_queue and predicted_spikes are both empty: propagate the event through the correct input neuron
            if (spike_queue.empty() && predicted_spikes.empty()) {
                spike s = neurons[idx]->receive_external_input(t, spike_type::initial, idx, -1, 1, 0);
                neurons[idx]->update(t, s.propagation_synapse, this, 0, s.type);
            } else {
                // propagate all spikes occuring before the event timestamp
                while ((!spike_queue.empty() && spike_queue.top().timestamp < t) || (!predicted_spikes.empty() && predicted_spikes.front().timestamp < t)) {
                    if (!spike_queue.empty() && predicted_spikes.empty()) {
                        auto& s = spike_queue.top();
                        neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, 0, s.type);
                        spike_queue.pop();
                    } else if (!predicted_spikes.empty() && spike_queue.empty()) {
                        auto& s = predicted_spikes.front();
                        neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, 0, s.type);
                        predicted_spikes.pop_front();
                    } else if (!predicted_spikes.empty() && !spike_queue.empty()) {
                        if (spike_queue.top().timestamp < predicted_spikes.front().timestamp) {
                            auto& s = spike_queue.top();
                            neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, 0, s.type);
                            spike_queue.pop();
                        } else if (predicted_spikes.front().timestamp < spike_queue.top().timestamp) {
                            auto& s = predicted_spikes.front();
                            neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, 0, s.type);
                            predicted_spikes.pop_front();
                        } else {
                            auto& s = spike_queue.top();
                            neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, 0, s.type);
                            spike_queue.pop();

                            auto& s2 = predicted_spikes.front();
                            neurons[s2.propagation_synapse->get_postsynaptic_neuron_id()]->update(s2.timestamp, s2.propagation_synapse, this, 0, s2.type);
                            predicted_spikes.pop_front();
                        }
                    }
                }

                // propagate the event through the correct input neuron
                spike s = neurons[idx]->receive_external_input(t, spike_type::initial, idx, -1, 1, 0);
                neurons[idx]->update(t, s.propagation_synapse, this, 0, s.type);
            }

            if (decision_making && classification && decision.timer > 0) {
                choose_winner_online(t, 0);
            }
        }

        // helper method that runs the network when event-mode is selected (timestep = 0)
        void async_run_helper(std::atomic_bool* running, bool classification=false, bool eof=false) {
            // lambda function to update neuron status asynchronously
            auto requestUpdate = [&](spike s, bool classification) {
                if (!classification) {
                    if (!eof) {
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
                    }
                } else {
                    if (decision_making && decision.timer > 0) {
                        choose_winner_online(s.timestamp, 0);
                    }
                }
                neurons[s.propagation_synapse->get_postsynaptic_neuron_id()]->update(s.timestamp, s.propagation_synapse, this, 0, s.type);
            };

            if (!neurons.empty()) {
                while (!spike_queue.empty() || !predicted_spikes.empty()) {

                    if (!running->load(std::memory_order_relaxed)) {
                        break;
                    }

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
                        } else {
                            requestUpdate(spike_queue.top(), classification);
                            spike_queue.pop();

                            requestUpdate(predicted_spikes.front(), classification);
                            predicted_spikes.pop_front();
                        }
                    }
                }
            } else {
                throw std::runtime_error("add neurons to the network before running it");
            }
        }

        // helper function that runs the network when clock-mode is selected (timestep > 0)
        void sync_run_helper(std::atomic_bool* running, double runtime, float timestep, bool classification=false) {
            if (!neurons.empty()) {

                // creating vector of the same size as neurons
                std::vector<bool> neuronStatus(neurons.size(), false);

                // loop over the full runtime
                for (double i=0; i<runtime; i+=timestep) {
                    // to close everything if GUI is closed
                    if (!running->load(std::memory_order_relaxed)) {
                        break;
                    }

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
                            choose_winner_online(i, timestep);
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
                    for (int idx=0; idx<static_cast<int>(neurons.size()); idx++) {
                        if (neuronStatus[idx]) {
                            neuronStatus[idx] = false;
                        } else {
                            // only update neurons if the previous layer is propagating
                            if (neurons[idx]->get_layer_id() == 0) {
                                neurons[idx]->update_sync(i, nullptr, this, timestep, spike_type::none);
                            } else {
                                if (layers[neurons[idx]->get_layer_id()].active) {
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
                        for (auto label: neuron_to_label->get_decision_queue()) {
                            ++classes_map[label];
                        }

                        // return the element with the maximum number of spikes
                        auto max_label = *std::max_element(classes_map.begin(), classes_map.end(), [](const std::pair<std::string, int> &p1,
                                                                                                      const std::pair<std::string, int> &p2) {
                                                                                        return p1.second < p2.second;
                                                                                    });

                        // assign label to neuron if element larger than the rejection threshold and does not hold less spikes than the spike_history_size
                        int inv_queue_size = 100 / neuron_to_label->get_decision_queue().size();
                        if (max_label.second * inv_queue_size >= decision.rejection_threshold && max_label.second >= decision.spike_history_size) {
                            neuron_to_label->set_class_label(max_label.first);
                        }

                        for (auto& decision_n: layers[decision.layer_number].neurons) {
                            // connect the neuron to its corresponding decision making neuron if they have the same label
                            if (!max_label.first.compare(neurons[decision_n]->get_class_label())) {
                                neuron_to_label->make_synapse(neurons[decision_n].get(), 1, 0, synapse_type::excitatory);
                            }
                        }
                    }
                }

                if (verbose == 1) {
                    for (auto& decision_n: layers[decision.layer_number].neurons) {
                        if (neurons[decision_n]->get_dendritic_tree().empty()) {
                            std::cout << "WARNING: No neurons have specialised for the decision neuron with the label " << neurons[decision_n]->get_class_label() << std::endl;
                        }
                    }
                }
            }
        }

        void choose_winner_online(double t, float timestep) {
            if (t - decision_pre_ts >= decision.timer) {
                // get intensities from all DecisionMaking neurons
                int winner_neuron = -1; float previous_intensity = -1.0f;
                for (auto& n: layers[decision.layer_number].neurons) {
                    if (!neurons[n]->get_dendritic_tree().empty()) {
                        float normalised_intensity = static_cast<float>(neurons[n]->share_information()) / neurons[n]->get_dendritic_tree().size();
                        if (normalised_intensity > previous_intensity && normalised_intensity > 0) {
                            winner_neuron = static_cast<int>(n);
                            previous_intensity = normalised_intensity;
                        }
                    }
                }

                // update the best DecisionMaking neuron
                if (winner_neuron != -1) {
                    neurons[winner_neuron]->update(t, nullptr, this, timestep, spike_type::decision);
                } else {
                    if (verbose >= 1) {
                        std::cout << "at t=" << t << " No decision could be made" << std::endl;
                    }
                }

                // saving previous timestamp
                decision_pre_ts = t;
            }
        }
        void choose_winner_eof(double t, float timestep) {
            
            // get intensities from all DecisionMaking neurons
            int winner_neuron = -1; float previous_intensity = -1.0f;
            for (auto& n: layers[decision.layer_number].neurons) {
                if (!neurons[n]->get_dendritic_tree().empty()) {
                    if (float normalised_intensity = static_cast<float>(neurons[n]->share_information()) / neurons[n]->get_dendritic_tree().size(); normalised_intensity > previous_intensity && normalised_intensity > 0) {
                        winner_neuron = static_cast<int>(n);
                        previous_intensity = normalised_intensity;
                    }
                }
            }

            // update the best DecisionMaking neuron
            if (winner_neuron != -1) {
                neurons[winner_neuron]->update(t, nullptr, this, timestep, spike_type::decision);
            } else {
                for (auto& addon: addons) {
                    addon->decision_failed(t, this);
                }
                if (verbose >= 1) {
                    std::cout << "at t=" << t << " No decision could be made" << std::endl;
                }
            }
        }

        std::vector<bool> find_successful_connections(int connection_ratio, int all_connections) {
            if (connection_ratio < 100) {
                std::vector<bool> connectivity_map(all_connections, false);
                std::vector<int> indices(all_connections);
                std::iota(indices.begin(), indices.end(), 0);

                std::random_device  device;
                std::seed_seq       seed{device(), device(), device(), device(), device(), device(), device(), device()};
                std::mt19937        random_engine(seed);

                // calculate how many successful connections there should be according to the connection_ratio
                int successful_connections = (connection_ratio * all_connections) / 100;

                // Fisher–Yates shuffle to select successful probabilities without replacement
                std::shuffle(indices.begin(), indices.end(), random_engine);

                for (auto it = indices.begin(); it != std::next(indices.begin(), successful_connections); ++it) {
                    connectivity_map[*it] = true;
                }

                return connectivity_map;

            } else {
                return std::vector<bool>(all_connections, true);
            }
        }

		// ----- IMPLEMENTATION VARIABLES -----
        int                                     verbose;
        std::priority_queue<spike>              spike_queue;
        std::deque<spike>                       predicted_spikes;
        std::vector<layer>                      layers;
		std::vector<std::unique_ptr<Neuron>>    neurons;
        std::vector<std::unique_ptr<Addon>>     addons;
        std::unique_ptr<MainAddon>              th_addon;
		std::deque<label>                       training_labels;
        bool                                    decision_making;
        std::unordered_map<std::string, int>    classes_map;
        std::string                             current_label;
		bool                                    learning_status;
		double                                  learning_off_signal;
        float                                   max_delay;
        bool                                    asynchronous;
        decision_heuristics                     decision;
        double                                  decision_pre_ts;
    };
}
