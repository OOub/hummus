/*
 * decisionMakingNeuron.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Decision-making neurons inherit from LIF neurons with the addition of a label for classification purposes. They should always be on the last layer of a network.
 */

#pragma once

#include "../core.hpp"
#include "LIF.hpp"

namespace adonis {
	class DecisionMakingNeuron : public LIF {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		DecisionMakingNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={},  bool _homeostasis=false, float _externalCurrent=100, float _resetCurrent=10, float _decayPotential=20, int _refractoryPeriod=1000, float _eligibilityDecay=20, float _decayWeight=0, float _decayHomeostasis=10, float _homeostasisBeta=1, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9, std::string _classLabel="") :
                LIF(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _homeostasis, _externalCurrent, _resetCurrent, _decayPotential, _refractoryPeriod, true, false, _eligibilityDecay, _decayWeight ,_decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _membraneResistance),
                classLabel(_classLabel){}
		
		virtual ~DecisionMakingNeuron(){}
		
        // ----- PUBLIC DECISION MAKING NEURON METHODS -----
        virtual void initialisation(Network* network) override {
            // checking if any children of the globalLearningRuleHandler class were initialised and adding them to the Addons vector
            for (auto& rule: learningRuleHandler) {
                if(AddOn* globalRule = dynamic_cast<AddOn*>(rule)) {
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
        
        virtual void update(double timestamp, axon* a, Network* network, spikeType type) override {
            if (type == normal) {
                
                // checking if the neuron is inhibited
                if (inhibited && timestamp - inhibitionTime >= refractoryPeriod) {
                    inhibited = false;
                }
                
                // checking if the neuron is in a refractory period
                if (timestamp - previousSpikeTime >= refractoryPeriod) {
                    active = true;
                }
                
                // reset the current to 0 in the absence of incoming spikes by the time t + resetCurrent
                if (timestamp - previousInputTime > resetCurrent) {
                    current = 0;
                }
                
                // eligibility trace decay
                eligibilityTrace *= std::exp(-(timestamp-previousInputTime)/eligibilityDecay);
                
                // potential decay
                potential = restingPotential + (potential-restingPotential)*std::exp(-(timestamp-previousInputTime)/decayPotential);
                
                // threshold decay
                if (homeostasis) {
                    threshold = restingThreshold + (threshold-restingThreshold)*std::exp(-(timestamp-previousInputTime)/decayHomeostasis);
                }
                
                // axon weight decay - synaptic pruning
                if (decayWeight != 0) {
                    a->weight *= std::exp(-(timestamp-previousInputTime)*synapticEfficacy/decayWeight);
                }
                
                if (active && !inhibited) {
                    
                    // calculating the potential
                    potential = restingPotential + membraneResistance * current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential) * std::exp(-(timestamp-previousInputTime)/decayPotential);
                    
                    // updating the threshold
                    if (homeostasis) {
                        threshold += homeostasisBeta/decayHomeostasis;
                    }
                    
                    // updating the current
                    current += externalCurrent*a->weight;
#ifndef NDEBUG
                    std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
#endif
                    for (auto addon: network->getAddOns()) {
                        if (potential < threshold) {
                            addon->incomingSpike(timestamp, a, network);
                        }
                    }
                    if (network->getMainThreadAddOn()) {
                        network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
                    }
                    
                    if (a->weight >= 0)
                    {
                        // calculating time at which potential = threshold
                        double predictedTimestamp = decayPotential * (- std::log( - threshold + restingPotential + membraneResistance * current) + std::log( membraneResistance * current - potential + restingPotential)) + timestamp;
                        
                        // calculating the potential at time t + resetCurrent
                        endOfIntegrationPotential = restingPotential + membraneResistance * current * (1 - std::exp(-(resetCurrent)/decayPotential)) + (endOfIntegrationPotential - restingPotential) * std::exp(-(decayPotential)/decayPotential);
                        
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + resetCurrent) {
                            network->injectPredictedSpike(spike{predictedTimestamp, a, prediction});
                        } else {
                            network->injectPredictedSpike(spike{timestamp + resetCurrent, a, endOfIntegration});
                        }
                    } else {
                        potential = restingPotential + membraneResistance * current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential) * std::exp(-(timestamp-previousInputTime)/decayPotential);
                    }
                }
            } else if (type == prediction){
                if (active && !inhibited) {
                    current += externalCurrent*a->weight;
                    potential = restingPotential + membraneResistance * current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential) * std::exp(-(timestamp-previousInputTime)/decayPotential);
                }
            } else if (type == endOfIntegration) {
                if (active && !inhibited) {
                    current += externalCurrent*a->weight;
                    if (endOfIntegrationPotential >= threshold) {
                        potential = restingPotential + membraneResistance * current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential) * std::exp(-(timestamp-previousInputTime)/decayPotential);
                    } else {
                        potential = endOfIntegrationPotential;
                    }
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
#ifndef NDEBUG
                std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
#endif
                
                for (auto addon: network->getAddOns()) {
                    addon->neuronFired(timestamp, a, network);
                }
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->neuronFired(timestamp, a, network);
                }
                
                for (auto& p : postAxons) {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), normal});
                }
                
                requestLearning(timestamp, a, network);
                
                previousSpikeTime = timestamp;
                endOfIntegrationPotential = restingPotential;
                potential = restingPotential;
                current = 0;
                active = false;
                
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->statusUpdate(timestamp, a, network);
                }
            }
            
            // updating the timestamp when an axon was propagating a spike
            previousInputTime = timestamp;
            a->previousInputTime = timestamp;
        }
        
        virtual void updateSync(double timestamp, axon* a, Network* network, double timestep) override {
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
            
            // reset the current to 0 in the absence of incoming spikes by the time t + resetCurrent
            if (timestamp - previousInputTime > resetCurrent) {
                current = 0;
            }
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
            
            // potential decay
            potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential);
            
            // threshold decay
            if (homeostasis) {
                threshold = restingThreshold + (threshold-restingThreshold)*exp(-timestep/decayHomeostasis);
            }
            
            if (a) {
                // axon weight decay - synaptic pruning
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
                    
                    // updating the current
                    current += externalCurrent*a->weight;
                    activeAxon = a;
                    
                    // updating the timestamp when an axon was propagating a spike
                    previousInputTime = timestamp;
                    a->previousInputTime = timestamp;
                    
#ifndef NDEBUG
                    std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
#endif
                    for (auto addon: network->getAddOns()) {
                        if (potential < threshold) {
                            addon->incomingSpike(timestamp, a, network);
                        }
                    }
                    if (network->getMainThreadAddOn()) {
                        network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
                    }
                }
                
                // membrane potential equation
                potential += membraneResistance * current * (1 - std::exp(-timestep/decayPotential));
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
                
#ifndef NDEBUG
                std::cout << "t=" << timestamp << " " << (activeAxon->preNeuron ? activeAxon->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeAxon->weight << " d=" << activeAxon->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
#endif
                
                for (auto addon: network->getAddOns()) {
                    addon->neuronFired(timestamp, activeAxon, network);
                }
                if (network->getMainThreadAddOn()) {
                    network->getMainThreadAddOn()->neuronFired(timestamp, activeAxon, network);
                }
                
                for (auto& p : postAxons) {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), normal});
                }
                
                requestLearning(timestamp, activeAxon, network);
                
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
                std::cout << neuronID << " specialised to the " << classLabel << " label" << std::endl;
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
        
        // loops through any learning rules and activates them
        void requestLearning(double timestamp, axon* a, Network* network) override {
            if (network->getLearningStatus()) {
                if (!learningRuleHandler.empty()) {
                    for (auto& learningRule: learningRuleHandler) {
                        learningRule->learn(timestamp, a, network);
                    }
                }
            }
            WTA(timestamp, network);
        }
    
		// ----- DECISION-MAKING NEURON PARAMETERS -----
        std::string        classLabel;
        std::vector<int>   labelTracker;
	};
}
