/*
 * decisionMaking.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Decision-making neurons inherit from LIF neurons with the addition of a label for classification purposes. They should always be on the last layer of a network.
 *
 * NEURON TYPE 2 (in JSON SAVE FILE)
 */

#pragma once

#include "../core.hpp"
#include "../dependencies/json.hpp"
#include "LIF.hpp"

namespace hummus {
	class DecisionMaking : public LIF {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		DecisionMaking(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, std::string _classLabel="", bool _homeostasis=false, float _decayPotential=20, float _decayCurrent=10, int _refractoryPeriod=3, bool _wta=false, bool _burstingActivity=false, float _eligibilityDecay=20, float _decayHomeostasis=20, float _homeostasisBeta=0.1, float _threshold=-50, float _restingPotential=-70) :
                LIF(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _homeostasis, _decayPotential, _decayCurrent, _refractoryPeriod, _wta, _burstingActivity, _eligibilityDecay, _decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential),
                classLabel(_classLabel) {
            // DecisionMaking neuron type = 2 for JSON save
            neuronType = 2;
        }
		
		virtual ~DecisionMaking(){}
		
        // ----- PUBLIC DECISION MAKING NEURON METHODS -----
        virtual void initialisation(Network* network) override {
            // initialising the label tracker according to the number of unique labels (if the labelTracker was not already initialised in a previous run instance
            if (labelTracker.empty()) {
                for (auto label: network->getUniqueLabels())
                {
                    labelTracker.push_back(0);
                }
            }
            
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
            if (type == spikeType::normal) {
                // checking if the neuron is inhibited
                if (inhibited && timestamp - inhibitionTime >= refractoryPeriod) {
                    inhibited = false;
                }
                
                // checking if the neuron is in a refractory period
                if (timestamp - previousSpikeTime >= refractoryPeriod) {
                    active = true;
                }
				
                // updating the current
                current *= std::exp(-(timestamp-previousInputTime)/decayCurrent);
                
                // eligibility trace decay
                eligibilityTrace *= std::exp(-(timestamp-previousInputTime)/eligibilityDecay);
                
                // potential decay
                potential = restingPotential + (potential-restingPotential)*std::exp(-(timestamp-previousInputTime)/decayPotential);
                
                // threshold decay
                if (homeostasis) {
                    threshold = restingThreshold + (threshold-restingThreshold)*std::exp(-(timestamp-previousInputTime)/decayHomeostasis);
                }
                
                if (active && !inhibited) {
                    // calculating the potential
                    potential = restingPotential + current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential) * std::exp(-(timestamp-previousInputTime)/decayPotential);
                    
                    // updating the threshold
                    if (homeostasis) {
                        threshold += homeostasisBeta/decayHomeostasis;
                    }
					
                    // synaptic integration
                    current += s->receiveSpike(timestamp);
                    
                    if (network->getVerbose() == 2) {
                        std::cout << "t=" << timestamp << " " << s->getPresynapticNeuronID() << "->" << neuronID << " w=" << s->getWeight() << " d=" << s->getDelay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
                    }
                    
                    for (auto& addon: relevantAddons) {
                        if (potential < threshold) {
                            addon->incomingSpike(timestamp, s, this, network);
                        }
                    }
                    if (network->getMainThreadAddon()) {
                        network->getMainThreadAddon()->incomingSpike(timestamp, s, this, network);
                    }
                    
                    if (s->getWeight() >= 0) {
                        // calculating time at which potential = threshold
                        double predictedTimestamp = decayPotential * (- std::log( - threshold + restingPotential + current) + std::log( current - potential + restingPotential)) + timestamp;
                        
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + s->getSynapseTimeConstant()) {
                            network->injectPredictedSpike(spike{predictedTimestamp, s, spikeType::prediction}, spikeType::prediction);
                        } else {
                            network->injectPredictedSpike(spike{timestamp + s->getSynapseTimeConstant(), s, spikeType::endOfIntegration}, spikeType::endOfIntegration);
                        }
                    } else {
                        potential = restingPotential + current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential);
                    }
                }
            } else if (type == spikeType::prediction) {
                if (active && !inhibited) {
                    potential = restingPotential + current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential);
                }
            } else if (type == spikeType::endOfIntegration) {
                if (active && !inhibited) {
                    potential = restingPotential + current * (1 - std::exp(-s->getSynapseTimeConstant()/decayPotential)) + (potential - restingPotential) * std::exp(-s->getSynapseTimeConstant()/decayPotential);
                }
            }
            
            if (network->getMainThreadAddon()) {
                network->getMainThreadAddon()->statusUpdate(timestamp, s, this, network);
            }
            
            if (potential >= threshold) {
                
                auto it = std::find(network->getUniqueLabels().begin(), network->getUniqueLabels().end(), network->getCurrentLabel());
                auto idx = std::distance(network->getUniqueLabels().begin(), it);
                labelTracker[idx] += 1;
                
                eligibilityTrace = 1;

                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << s->getPresynapticNeuronID() << "->" << neuronID << " w=" << s->getWeight() << " d=" << s->getDelay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
                }
                
                for (auto& addon: relevantAddons) {
                    addon->neuronFired(timestamp, s, this, network);
                }
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->neuronFired(timestamp, s, this, network);
                }
                
                for (auto& axonTerminal: axonTerminals) {
                    network->injectGeneratedSpike(spike{timestamp + axonTerminal->getDelay(), axonTerminal.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, s, this, network);
                
                previousSpikeTime = timestamp;
                potential = restingPotential;
                if (!burstingActivity) {
                    current = 0;
                }
                active = false;
                
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->statusUpdate(timestamp, s, this, network);
                }
            }
            
            // updating the timestamp when a synapse was propagating a spike
            previousInputTime = timestamp;
            s->setPreviousInputTime(timestamp);
        }
        
        virtual void updateSync(double timestamp, Synapse* s, Network* network, double timestep) override {
            // handling multiple spikes at the same timestamp (to prevent excessive decay)
            if (timestamp != 0 && timestamp - previousSpikeTime == 0) {
                timestep = 0;
            }
            
            // checking if the neuron is inhibited
            if (inhibited && timestamp - inhibitionTime >= refractoryPeriod) {
                inhibited = false;
            }
            
            // checking if the neuron is in a refractory period
            if (timestamp - previousSpikeTime >= refractoryPeriod) {
                active = true;
            }
            
            // updating the current
            current *= std::exp(-timestep/decayCurrent);
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
            
            // potential decay
            potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential);
            
            // threshold decay
            if (homeostasis) {
                threshold = restingThreshold + (threshold-restingThreshold)*std::exp(-timestep/decayHomeostasis);
            }
            
            // neuron inactive during refractory period
            if (active && !inhibited) {
                if (s) {
                    // updating the threshold
                    if (homeostasis) {
                        threshold += homeostasisBeta/decayHomeostasis;
                    }
                    
                    // integrating spike
					current += s->receiveSpike(timestamp);
                    
                    activeSynapse = s;
                    
                    // updating the timestamp when a synapse was propagating a spike
                    previousInputTime = timestamp;
                    s->setPreviousInputTime(timestamp);
                    
                    if (network->getVerbose() == 2) {
                        std::cout << "t=" << timestamp << " " << s->getPresynapticNeuronID() << "->" << neuronID << " w=" << s->getWeight() << " d=" << s->getDelay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
                    }
                    
                    for (auto& addon: relevantAddons) {
                        if (potential < threshold) {
                            addon->incomingSpike(timestamp, s, this, network);
                        }
                    }
                    if (network->getMainThreadAddon()) {
                        network->getMainThreadAddon()->incomingSpike(timestamp, s, this, network);
                    }
                }
				
				potential += current * (1 - std::exp(-timestep/decayPotential));
            }
            
            if (s) {
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->statusUpdate(timestamp, s, this, network);
                }
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
            
            if (potential >= threshold) {
                
                auto it = std::find(network->getUniqueLabels().begin(), network->getUniqueLabels().end(), network->getCurrentLabel());
                auto idx = std::distance(network->getUniqueLabels().begin(), it);
                labelTracker[idx] += 1;
                
                eligibilityTrace = 1;
                
                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << activeSynapse->getPresynapticNeuronID() << "->" << neuronID << " w=" << activeSynapse->getWeight() << " d=" << activeSynapse->getDelay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
                }
                
                for (auto& addon: relevantAddons) {
                    addon->neuronFired(timestamp, activeSynapse, this, network);
                }
                if (network->getMainThreadAddon()) {
                    network->getMainThreadAddon()->neuronFired(timestamp, activeSynapse, this, network);
                }
                
                for (auto& axonTerminal : axonTerminals) {
                    network->injectGeneratedSpike(spike{timestamp + axonTerminal->getDelay(), axonTerminal.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, activeSynapse, this, network);
                
                previousSpikeTime = timestamp;
                potential = restingPotential;
                if (!burstingActivity) {
                    current = 0;
                }
                active = false;
            }
        }
        
        virtual void resetNeuron(Network* network, bool clearAddons=true) override {
            // resetting parameters
            previousInputTime = 0;
            previousSpikeTime = 0;
            current = 0;
            potential = restingPotential;
            eligibilityTrace = 0;
            inhibited = false;
            active = true;
            threshold = restingThreshold;
            
            if (clearAddons) {
                relevantAddons.clear();
            }
            
            // making sure in a new run the classLabel isn't overwritten
            if (!network->getPreTrainingLabelAssignment() && classLabel == "") {
                // associating the appropriate label to the decision-making neuron
                auto it = std::max_element(labelTracker.begin(), labelTracker.end());
                auto idx = std::distance(labelTracker.begin(), it);
                classLabel = network->getUniqueLabels()[idx];
                
                if (network->getVerbose() == 2) {
                    std::cout << neuronID << " specialised to the " << classLabel << " label" << std::endl;
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
                {"decayPotential", decayPotential},
                {"decayCurrent", decayCurrent},
                {"burstingActivity", burstingActivity},
                {"homeostasis", homeostasis},
                {"restingThreshold", restingThreshold},
                {"decayHomeostasis", decayHomeostasis},
                {"homeostasisBeta", homeostasisBeta},
                {"wta", wta},
                {"classLabel", classLabel},
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
		std::string getClassLabel() const {
			return classLabel;
		}
		
		void setClassLabel(std::string newLabel) {
			classLabel = newLabel;
		}
        
    protected:
    
		// ----- DECISION-MAKING NEURON PARAMETERS -----
        std::string        classLabel;
        std::vector<int>   labelTracker;
	};
}
