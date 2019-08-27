/*
 * decisionMaking.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Decision-making neurons act as our classifier, roughly approximating a histogram activity-dependent classification. They should always be on the last layer of a network.
 * 
 * NEURON TYPE 2 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class DecisionMaking : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        DecisionMaking(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, std::string _classLabel="", int _refractoryPeriod=10, float _conductance=200, float _leakageConductance=10, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _refractoryPeriod, _conductance, _leakageConductance, _traceTimeConstant, _threshold, _restingPotential, _classLabel),
                active(true),
                inhibition_time(0) {
                    
            // DecisionMaking neuron type = 2 for JSON save
            neuronType = 2;
        }
		
		virtual ~DecisionMaking(){}
		
        // ----- PUBLIC DECISION MAKING NEURON METHODS -----
        virtual void initialisation(Network* network) override {
            // searching for addons that are relevant to this neuron. if addons do not have a mask they are automatically relevant / not filtered out
            for (auto& addon: network->getAddons()) {
                if (addon->getNeuronMask().empty()) {
                    addRelevantAddon(addon.get());
                } else {
                    auto it = std::find(addon->getNeuronMask().begin(), addon->getNeuronMask().end(), static_cast<size_t>(neuronID));
                    if (it != addon->getNeuronMask().end()) {
                        addRelevantAddon(addon.get());
                    }
                }
            }
        }
        
        virtual void update(double timestamp, Synapse* s, Network* network, spikeType type) override {
            // checking if the neuron is in a refractory period
            if (timestamp - inhibition_time >= refractoryPeriod) {
                active = true;
            }
            
            if (type == spikeType::decision) {
                if (active && intensity > 0) {
                    // function that converts the intensity to a delay
                    float intensity_to_latency = 10 * 1 - std::exp(- intensity/dendriticTree.size());
                    
                    // make the neuron fire so we can get the decision
                    potential = threshold;
                    
                    if (network->getVerbose() == 1) {
                        std::cout << "t=" << timestamp << " class " << classLabel << " --> DECISION" << std::endl;
                    }
                    
                    for (auto& addon: relevantAddons) {
                        addon->neuronFired(timestamp, s, this, network);
                    }
                    
                    if (network->getMainThreadAddon()) {
                        network->getMainThreadAddon()->neuronFired(timestamp, s, this, network);
                    }
                    
                    // propagating the decision spike
                    if (!network->getLayers()[layerID].do_not_propagate) {
                        for (auto& axonTerminal: axonTerminals) {
                            network->injectSpike(spike{timestamp + intensity_to_latency, axonTerminal.get(), spikeType::generated});
                        }
                    }
                    
                    // inhibiting the other decision_making neurons
                    winner_takes_all(timestamp, network);
                }
                
                intensity = 0;
                
            } else if (type != spikeType::none){
                ++intensity;
            }
        }
        
        // write neuron parameters in a JSON format
        virtual void toJson(nlohmann::json& output) override{
            // general neuron parameters
            output.push_back({
                {"Type",neuronType},
                {"layerID",layerID},
                {"sublayerID", sublayerID},
                {"receptiveFieldCoordinates", rfCoordinates},
                {"XYCoordinates", xyCoordinates},
                {"traceTimeConstant", traceTimeConstant},
                {"threshold", threshold},
                {"restingPotential", restingPotential},
                {"refractoryPeriod", refractoryPeriod},
                {"dendriticSynapses", nlohmann::json::array()},
                {"axonalSynapses", nlohmann::json::array()},
            });
            
            // dendritic synapses (preSynapse)
            auto& dendriticSynapses = output.back()["dendriticSynapses"];
            for (auto& dendrite: dendriticTree) {
                dendrite->toJson(dendriticSynapses);
            }
            
            // axonal synapses (postSynapse)
            auto& axonalSynapses = output.back()["axonalSynapses"];
            for (auto& axonTerminal: axonTerminals) {
                axonTerminal->toJson(axonalSynapses);
            }
        }
        
    protected:
        
        void winner_takes_all(double timestamp, Network* network) {
            for (auto& n: network->getLayers()[layerID].neurons) {
                auto& neuron = network->getNeurons()[n];
                
                // inhibit all the other neurons in the same layer
                if (neuron->getNeuronID() != neuronID) {
                    dynamic_cast<DecisionMaking*>(neuron.get())->setActivity(false);
                    dynamic_cast<DecisionMaking*>(neuron.get())->setInhibitionTime(timestamp);
                }
            }
        }
        
        // ----- SETTERS AND GETTERS -----
        void setActivity(bool new_state) {
            active = new_state;
        }
        
        void setInhibitionTime(double new_time) {
            inhibition_time = new_time;
        }
        
		// ----- DECISION-MAKING NEURON PARAMETERS -----
        float    intensity;
        bool     active;
        double   inhibition_time;
	};
}
