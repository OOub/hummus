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
		Input(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, SynapticKernelHandler* _synapticKernel=nullptr, int _refractoryPeriod=0, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _synapticKernel, _eligibilityDecay, _threshold, _restingPotential),
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
        
		virtual void update(double timestamp, synapse* a, Network* network, spikeType type) override {
            
            // checking if the neuron is in a refractory period
            if (timestamp - previousSpikeTime >= refractoryPeriod) {
                active = true;
            }
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-(timestamp - previousSpikeTime)/eligibilityDecay);
            
            // instantly making the input neuron fire at every input spike
            if (active) {
                a->previousInputTime = timestamp;
                potential = threshold;
                eligibilityTrace = 1;
                
                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuronID << " w=" << a->weight << " d=" << a->delay << " --> INPUT" << std::endl;
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->incomingSpike(timestamp, a, network);
                }
                
                for (auto& addon: relevantAddons) {
                    addon->neuronFired(timestamp, a, network);
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->neuronFired(timestamp, a, network);
                }
                
                for (auto& p : postSynapses) {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, a, network);
                previousSpikeTime = timestamp;
                potential = restingPotential;
                active = false;
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->statusUpdate(timestamp, a, network);
                }
            }
		}
        
        virtual void updateSync(double timestamp, synapse* a, Network* network, double timestep) override {
            
            if (timestamp != 0 && timestamp - previousSpikeTime == 0) {
                timestep = 0;
            }
            
            // checking if the neuron is in a refractory period
            if (timestamp - previousSpikeTime >= refractoryPeriod) {
                active = true;
            }
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
            
            if (a && active) {
                a->previousInputTime = timestamp;
                potential = threshold;
                eligibilityTrace = 1;
                
                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuronID << " w=" << a->weight << " d=" << a->delay << " --> INPUT" << std::endl;
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->incomingSpike(timestamp, a, network);
                }
                
                for (auto& addon: relevantAddons) {
                    addon->neuronFired(timestamp, a, network);
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->neuronFired(timestamp, a, network);
                }
                
                for (auto& p : postSynapses) {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, a, network);
                previousSpikeTime = timestamp;
                potential = restingPotential;
                active = false;
            } else {
                if (timestep > 0) {
                    for (auto& addon: relevantAddons) {
                        addon->timestep(timestamp, network, this);
                    }
                    if (network->getMainThreadAddon()) {
                        network->getMainThreadAddon()->timestep(timestamp, network, this);
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
            for (auto& preS: preSynapses) {
                dendriticSynapses.push_back({
                    {"weight", preS->weight},
                    {"delay", preS->delay},
                });
            }
            
            // axonal synapses (postSynapse)
            auto& axonalSynapses = output.back()["axonalSynapses"];
            for (auto& postS: postSynapses) {
                axonalSynapses.push_back({
                    {"postNeuronID", postS->postNeuron->getNeuronID()},
                    {"weight", postS->weight},
                    {"delay", postS->delay},
                });
            }
        }
        
        // ----- SETTERS AND GETTERS -----
        void setRefractoryPeriod(float newRefractoryPeriod) {
            refractoryPeriod = newRefractoryPeriod;
        }
        
    protected:
        
        // loops through any learning rules and activates them
        virtual void requestLearning(double timestamp, synapse* a, Network* network) override {
            if (network->getLearningStatus()) {
                for (auto& addon: relevantAddons) {
                    addon->learn(timestamp, a, network);
                }
            }
        }
        
        // ----- INPUT NEURON PARAMETERS -----
        float refractoryPeriod;
        bool  active;
        
	};
}
