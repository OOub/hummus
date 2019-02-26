/*
 * IF.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 19/02/2019
 *
 * Information: integrate-and-fire neuron without any leakage
 */

#pragma once

#include "../core.hpp"
#include "LIF.hpp"

namespace adonis {
	class IF : public LIF {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		IF(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={},  bool _timeDependentCurrent=false, bool _homeostasis=false, float _resetCurrent=10, int _refractoryPeriod=3, bool _wta=false, bool _burstingActivity=false, float _eligibilityDecay=20, float _decayWeight=0, float _decayHomeostasis=10, float _homeostasisBeta=1, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9, float _externalCurrent=100) :
            LIF(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _timeDependentCurrent, _homeostasis, _resetCurrent, 1, _refractoryPeriod, true, false, _eligibilityDecay, _decayWeight ,_decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _membraneResistance, _externalCurrent){}
		
		virtual ~IF(){}
		
		// ----- PUBLIC INPUT NEURON METHODS -----
		void initialisation(Network* network) override {
			for (auto& rule: learningRuleHandler) {
				if(AddOn* globalRule = dynamic_cast<AddOn*>(rule)) {
					if (std::find(network->getAddOns().begin(), network->getAddOns().end(), dynamic_cast<AddOn*>(rule)) == network->getAddOns().end()) {
						network->getAddOns().emplace_back(dynamic_cast<AddOn*>(rule));
					}
				}
			}
		}
		
        virtual void update(double timestamp, axon* a, Network* network, spikeType type) override {
            if (type == spikeType::normal) {
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
                    
                    if (a->weight >= 0) {
                        // calculating time at which potential = threshold
                        double predictedTimestamp = decayPotential * (- std::log( - threshold + restingPotential + membraneResistance * current) + std::log( membraneResistance * current - potential + restingPotential)) + timestamp;
                        
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + resetCurrent) {
                            network->injectPredictedSpike(spike{predictedTimestamp, a, spikeType::prediction}, spikeType::prediction);
                        } else {
                            network->injectPredictedSpike(spike{timestamp + resetCurrent, a, spikeType::endOfIntegration}, spikeType::endOfIntegration);
                        }
                    } else {
                        potential = restingPotential + membraneResistance * current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential);
                    }
                }
            } else if (type == spikeType::prediction) {
                if (active && !inhibited) {
                    potential = restingPotential + membraneResistance * current * (1 - std::exp(-(timestamp-previousInputTime)/decayPotential)) + (potential - restingPotential);
                }
            } else if (type == spikeType::endOfIntegration) {
                if (active && !inhibited) {
                    potential = restingPotential + membraneResistance * current * (1 - std::exp(-resetCurrent/decayPotential)) + (potential - restingPotential) * std::exp(-resetCurrent/decayPotential);
                }
            }
            
            if (network->getMainThreadAddOn()) {
                network->getMainThreadAddOn()->statusUpdate(timestamp, a, network);
            }
            
            if (potential >= threshold) {
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
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), spikeType::normal});
                }
                
                requestLearning(timestamp, a, network);
                
                previousSpikeTime = timestamp;
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
            
            if (timeDependentCurrent ) {
                // current decay
                current *= std::exp(-timestep/resetCurrent);
            } else {
                // reset the current to 0 in the absence of incoming spikes by the time t + resetCurrent
                if (timestamp - previousInputTime > resetCurrent) {
                    current = 0;
                }
            }
            
            // threshold decay
            if (homeostasis) {
                threshold = restingThreshold + (threshold-restingThreshold)*exp(-timestep/decayHomeostasis);
            }
            
            if (a) {
                // axon weight decay - synaptic pruning
                if (decayWeight != 0) {
                    a->weight *= std::exp(-(timestamp-previousInputTime)*synapticEfficacy/decayWeight);
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
                
                if (timeDependentCurrent ) {
                    // membrane potential equation for time-dependant current (double exponential model)
                    potential += (membraneResistance*resetCurrent/(resetCurrent - decayPotential)) * current * (std::exp(-timestep/resetCurrent) - std::exp(-timestep/decayPotential));
                } else {
                    // membrane potential equation for constant current
                    potential += membraneResistance * current * (1 - std::exp(-timestep/decayPotential));
                }
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
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get(), spikeType::normal});
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
	};
}