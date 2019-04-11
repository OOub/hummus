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
 * NEURON TYPE 3 (in JSON SAVE FILE)
 */

#pragma once

#include "../core.hpp"
#include "../dependencies/json.hpp"
#include "LIF.hpp"

namespace hummus {
	class DecisionMaking : public LIF {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		DecisionMaking(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<int, int> _xyCoordinates, std::vector<LearningRuleHandler*> _learningRules, SynapticKernelHandler* _synapticKernel, bool _homeostasis=false, float _decayPotential=20, int _refractoryPeriod=3, bool _wta=false, bool _burstingActivity=false, float _eligibilityDecay=20, float _decayWeight=0, float _decayHomeostasis=20, float _homeostasisBeta=0.1, float _threshold=-50, float _restingPotential=-70, float _externalCurrent=100, std::string _classLabel="") :
                    LIF(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _learningRules, _synapticKernel, _homeostasis, _decayPotential, _refractoryPeriod, _wta, _burstingActivity, _eligibilityDecay, _decayWeight ,_decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _externalCurrent),
                    classLabel(_classLabel) {
                // DecisionMaking neuron type = 2 for JSON save
                neuronType = 3;
            }
		
		virtual ~DecisionMaking(){}
		
        // ----- PUBLIC DECISION MAKING NEURON METHODS -----
        virtual void initialisation(Network* network) override {
            // checking which synaptic kernel was chosen in the asynchronous network
            if (network->getNetworkType() == true) {
                if (Exponential *kernel = dynamic_cast<Exponential*>(synapticKernel)) {
                	throw std::logic_error("the event-based LIF neuron does not work with the Exponential kernel, as the biexponential model it is based on, does not have an analytical solution");
				}
            }
            
            // checking if any children of the globalLearningRuleHandler class were initialised and adding them to the Addons vector
            for (auto& rule: learningRules) {
                if (AddOn *globalRule = dynamic_cast<AddOn*>(rule)) {
                    if (std::find(network->getAddOns().begin(), network->getAddOns().end(), dynamic_cast<AddOn*>(rule)) == network->getAddOns().end()) {
                        network->getAddOns().emplace_back(dynamic_cast<AddOn*>(rule));
                    }
                }
            }
            
            // initialising the label tracker according to the number of unique labels
            for (auto label: network->getUniqueLabels())
            {
                labelTracker.push_back(0);
            }
        }
        
        virtual void update(double timestamp, synapse* a, Network* network, spikeType type) override {
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
                current = synapticKernel->updateCurrent(timestamp, 0, previousInputTime, current);
                
                // eligibility trace decay
                eligibilityTrace *= std::exp(-(timestamp-previousInputTime)*adaptation/eligibilityDecay);
                
                // potential decay
                potential = restingPotential + (potential-restingPotential)*std::exp(-(timestamp-previousInputTime)*adaptation/decayPotential);
                
                // threshold decay
                if (homeostasis) {
                    threshold = restingThreshold + (threshold-restingThreshold)*std::exp(-(timestamp-previousInputTime)*adaptation/decayHomeostasis);
                }
                
                // synapse weight decay - synaptic pruning
                if (decayWeight != 0) {
                    a->weight *= std::exp(-(timestamp-previousInputTime)*synapticEfficacy/decayWeight);
                }
                
                if (active && !inhibited) {
                    // calculating the potential
                    potential = restingPotential + current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential) * std::exp(-(timestamp-previousInputTime)/decayPotential);
                    
                    // updating the threshold
                    if (homeostasis) {
                        threshold += homeostasisBeta/decayHomeostasis;
                    }
					
                    // synaptic integration
                    current = synapticKernel->integrateSpike(current, externalCurrent, a->weight);
                    
                    if (network->getVerbose() == 2) {
                        std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
                    }
                    
                    for (auto addon: network->getAddOns()) {
                        if (potential < threshold) {
                            addon->incomingSpike(timestamp, a, network);
                        }
                    }
                    if (network->getMainThreadAddOn()) {
                        network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
                    }
                    
                    if (a->weight >= 0) {
                        // calculating time at which potential = threshold
                        double predictedTimestamp = decayPotential * (- std::log( - threshold + restingPotential + current) + std::log( current - potential + restingPotential)) + timestamp;
                        
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + synapticKernel->getSynapseTimeConstant()) {
                            network->injectPredictedSpike(spike{predictedTimestamp, a, spikeType::prediction}, spikeType::prediction);
                        } else {
                            network->injectPredictedSpike(spike{timestamp + synapticKernel->getSynapseTimeConstant(), a, spikeType::endOfIntegration}, spikeType::endOfIntegration);
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
                    potential = restingPotential + current * (1 - std::exp(-synapticKernel->getSynapseTimeConstant()/decayPotential)) + (potential - restingPotential) * std::exp(-synapticKernel->getSynapseTimeConstant()/decayPotential);
                }
            }
            
            if (network->getMainThreadAddOn()) {
                network->getMainThreadAddOn()->statusUpdate(timestamp, a, network);
            }
            
            if (potential >= threshold) {
                
                auto it = std::find(network->getUniqueLabels().begin(), network->getUniqueLabels().end(), network->getCurrentLabel());
                auto idx = std::distance(network->getUniqueLabels().begin(), it);
                labelTracker[idx] += 1;
                
                eligibilityTrace = 1;

                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
                }
                
                for (auto addon: network->getAddOns()) {
                    addon->neuronFired(timestamp, a, network);
                }
                
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->neuronFired(timestamp, a, network);
                }
                
                for (auto& p : postSynapses) {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, a, network);
                
                previousSpikeTime = timestamp;
                potential = restingPotential;
                if (!burstingActivity) {
                    current = 0;
                }
                active = false;
                
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->statusUpdate(timestamp, a, network);
                }
            }
            
            // updating the timestamp when a synapse was propagating a spike
            previousInputTime = timestamp;
            a->previousInputTime = timestamp;
        }
        
        virtual void updateSync(double timestamp, synapse* a, Network* network, double timestep) override {
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
			current = synapticKernel->updateCurrent(timestamp, timestep, previousInputTime, current);
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep*adaptation/eligibilityDecay);
            
            // potential decay
            potential = restingPotential + (potential-restingPotential)*std::exp(-timestep*adaptation/decayPotential);
            
            // threshold decay
            if (homeostasis) {
                threshold = restingThreshold + (threshold-restingThreshold)*std::exp(-timestep*adaptation/decayHomeostasis);
            }
            
            if (a) {
                // synapse weight decay - synaptic pruning
                if (decayWeight != 0) {
                    a->weight *= std::exp(-(timestamp-previousSpikeTime)*synapticEfficacy/decayWeight);
                }
            }
            
            // neuron inactive during refractory period
            if (active && !inhibited) {
                if (a) {
                    // updating the threshold
                    if (homeostasis) {
                        threshold += homeostasisBeta/decayHomeostasis;
                    }
                    
                    // integrating spike
					current = synapticKernel->integrateSpike(current, externalCurrent, a->weight);
                    
                    activeSynapse = a;
                    
                    // updating the timestamp when a synapse was propagating a spike
                    previousInputTime = timestamp;
                    a->previousInputTime = timestamp;
                    
                    if (network->getVerbose() == 2) {
                        std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
                    }
                    
                    for (auto addon: network->getAddOns()) {
                        if (potential < threshold) {
                            addon->incomingSpike(timestamp, a, network);
                        }
                    }
                    if (network->getMainThreadAddOn()) {
                        network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
                    }
                }
				
				potential += current * (1 - std::exp(-timestep/decayPotential));
            }
            
            if (a) {
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->statusUpdate(timestamp, a, network);
                }
            } else {
                if (timestep > 0) {
                    for (auto addon: network->getAddOns()) {
                        addon->timestep(timestamp, network, this);
                    }
                    if (network->getMainThreadAddOn()) {
                        network->getMainThreadAddOn()->timestep(timestamp, network, this);
                    }
                }
            }
            
            if (potential >= threshold) {
                
                auto it = std::find(network->getUniqueLabels().begin(), network->getUniqueLabels().end(), network->getCurrentLabel());
                auto idx = std::distance(network->getUniqueLabels().begin(), it);
                labelTracker[idx] += 1;
                
                eligibilityTrace = 1;
                
                if (network->getVerbose() == 2) {
                    std::cout << "t=" << timestamp << " " << (activeSynapse->preNeuron ? activeSynapse->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeSynapse->weight << " d=" << activeSynapse->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
                }
                
                for (auto addon: network->getAddOns()) {
                    addon->neuronFired(timestamp, activeSynapse, network);
                }
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->neuronFired(timestamp, activeSynapse, network);
                }
                
                for (auto& p : postSynapses) {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, activeSynapse, network);
                
                previousSpikeTime = timestamp;
                potential = restingPotential;
                if (!burstingActivity) {
                    current = 0;
                }
                active = false;
            }
        }
        
        virtual void resetNeuron(Network* network) override {
            // resetting parameters
            previousInputTime = 0;
            previousSpikeTime = 0;
            current = 0;
            potential = restingPotential;
            eligibilityTrace = 0;
            inhibited = false;
            active = true;
            threshold = restingThreshold;
            
            if (!network->getPreTrainingLabelAssignment()) {
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
                {"externalCurrent", externalCurrent},
                {"burstingActivity", burstingActivity},
                {"homeostasis", homeostasis},
                {"restingThreshold", restingThreshold},
                {"decayWeight", decayWeight},
                {"decayHomeostasis", decayHomeostasis},
                {"homeostasisBeta", homeostasisBeta},
                {"wta", wta},
                {"classLabel", classLabel},
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
