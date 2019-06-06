/*
 * input.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: input neurons take in spikes or events and instantly propagate them in the network. The potential does not decay.
 *
 * NEURON TYPE 0 (in JSON SAVE FILE)
 */

#pragma once

#include "../core.hpp"
#include "../dependencies/json.hpp"

namespace hummus {
	class Input : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		Input(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, int _refractoryPeriod=0, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _eligibilityDecay, _threshold, _restingPotential),
                active(true),
                refractoryPeriod(_refractoryPeriod) {}
		
		virtual ~Input(){}
		
		// ----- PUBLIC INPUT NEURON METHODS -----
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
            if (timestamp - previousSpikeTime >= refractoryPeriod) {
                active = true;
            }
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-(timestamp - previousSpikeTime)/eligibilityDecay);
            
            // instantly making the input neuron fire at every input spike
            if (active) {
                potential = threshold;
                eligibilityTrace = 1;
                
                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuronID << " w=" << s->getWeight() << " d=" << s->getDelay() << " --> INPUT" << std::endl;
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->incomingSpike(timestamp, s, this, network);
                }
                
                for (auto& addon: relevantAddons) {
                    addon->neuronFired(timestamp, s, this, network);
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->neuronFired(timestamp, s, this, network);
                }
                
                for (auto& axonTerminal : axonTerminals) {
                    network->injectGeneratedSpike(spike{timestamp + axonTerminal->getDelay(), axonTerminal.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, s, this, network);
                previousSpikeTime = timestamp;
                potential = restingPotential;
                active = false;
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->statusUpdate(timestamp, s, this, network);
                }
            }
		}
        
        virtual void updateSync(double timestamp, Synapse* s, Network* network, double timestep) override {
            
            if (timestamp != 0 && timestamp - previousSpikeTime == 0) {
                timestep = 0;
            }
            
            // checking if the neuron is in a refractory period
            if (timestamp - previousSpikeTime >= refractoryPeriod) {
                active = true;
            }
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
            
            if (s && active) {
                potential = threshold;
                eligibilityTrace = 1;
                
                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuronID << " w=" << s->getWeight() << " d=" << s->getDelay() << " --> INPUT" << std::endl;
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->incomingSpike(timestamp, s, this, network);
                }
                
                for (auto& addon: relevantAddons) {
                    addon->neuronFired(timestamp, s, this, network);
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->neuronFired(timestamp, s, this, network);
                }
                
                for (auto& axonTerminal : axonTerminals) {
                    network->injectGeneratedSpike(spike{timestamp + axonTerminal->getDelay(), axonTerminal.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, s, this, network);
                previousSpikeTime = timestamp;
                potential = restingPotential;
                active = false;
            } else {
                if (timestep > 0) {
                    for (auto& addon: relevantAddons) {
                        addon->timestep(timestamp, this, network);
                    }
                    if (network->getMainThreadAddon()) {
                        network->getMainThreadAddon()->timestep(timestamp, this, network);
                    }
                }
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
                {"eligibilityDecay", eligibilityDecay},
                {"threshold", threshold},
                {"restingPotential", restingPotential},
                {"refractoryPeriod", refractoryPeriod},
                {"dendriticSynapses", nlohmann::json::array()},
                {"axonalSynapses", nlohmann::json::array()},
            });
            
            // dendritic synapses (preSynapse)
            auto& dendriticSynapses = output.back()["dendriticSynapses"];
            for (auto& dendrite: dendriticTree) {
                dendriticSynapses.push_back({
                    {"type", dendrite->getType()},
                    {"weight", dendrite->getWeight()},
                    {"delay", dendrite->getDelay()},
                });
            }
            
            // axonal synapses (postSynapse)
            auto& axonalSynapses = output.back()["axonalSynapses"];
            for (auto& axonTerminal: axonTerminals) {
                axonalSynapses.push_back({
                    {"type", axonTerminal->getType()},
                    {"postNeuronID", axonTerminal->getPostsynapticNeuronID()},
                    {"weight", axonTerminal->getWeight()},
                    {"delay", axonTerminal->getDelay()},
                });
            }
        }
        
        // ----- SETTERS AND GETTERS -----
        void setRefractoryPeriod(float newRefractoryPeriod) {
            refractoryPeriod = newRefractoryPeriod;
        }
        
    protected:
        
        // loops through any learning rules and activates them
        virtual void requestLearning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->getLearningStatus()) {
                for (auto& addon: relevantAddons) {
                    addon->learn(timestamp, s, postsynapticNeuron, network);
                }
            }
        }
        
        // ----- INPUT NEURON PARAMETERS -----
        float refractoryPeriod;
        bool  active;
        
	};
}
