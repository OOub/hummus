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

#include "../core.hpp"
#include "../dependencies/json.hpp"

namespace hummus {
	class DecisionMaking : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        DecisionMaking(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, std::string _classLabel="", float _conductance=200, float _leakageConductance=10, int _refractoryPeriod=3, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _conductance, _leakageConductance, _refractoryPeriod, _traceTimeConstant, _threshold, _restingPotential, _classLabel){
                    
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
        }
        
        virtual void updateSync(double timestamp, Synapse* s, Network* network, double timestep, spikeType type) override {
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
                {"conductance", conductance},
                {"leakageConductance", leakageConductance},
                {"classLabel", classLabel},
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
        
		// ----- SETTERS AND GETTERS -----
        
    protected:
    
		// ----- DECISION-MAKING NEURON PARAMETERS -----

	};
}
