/*
 * LIF.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/01/2019
 *
 * Information: leaky integrate and fire (LIF) neuron model with current dynamics.
 *
 * NEURON TYPE 1 (in JSON SAVE FILE)
 */

#pragma once

#include "../core.hpp"
#include "../dependencies/fastapprox/fastexp.h"
#include "../dependencies/fastapprox/fastlog.h"
#include "../dependencies/json.hpp"
#include "../synapses/exponential.hpp"
#include "../synapses/dirac.hpp"
#include "../synapses/pulse.hpp"

namespace hummus {
    
	class LIF : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		LIF(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, bool _homeostasis=false, float _membrane_time_constant=20, int _refractoryPeriod=3, bool _wta=false, bool _burstingActivity=false, float _eligibilityDecay=20, float _decayHomeostasis=20, float _homeostasisBeta=0.1, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _eligibilityDecay, _threshold, _restingPotential),
                refractoryPeriod(_refractoryPeriod),
                membrane_time_constant(_membrane_time_constant),
                active(true),
                burstingActivity(_burstingActivity),
                homeostasis(_homeostasis),
                restingThreshold(_threshold),
                decayHomeostasis(_decayHomeostasis),
                homeostasisBeta(_homeostasisBeta),
                inhibited(false),
                inhibitionTime(0),
                wta(_wta) {
					
			// error handling
    	    if (membrane_time_constant <= 0) {
                throw std::logic_error("The potential decay cannot less than or equal to 0");
            }
					
            // LIF neuron type == 1 (for JSON save)
            neuronType = 1;
		}
		
		virtual ~LIF(){}
		
		// ----- PUBLIC LIF METHODS -----        
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
            if (type == spikeType::normal) {
                // checking if the neuron is inhibited
                if (inhibited && timestamp - inhibitionTime >= refractoryPeriod) {
                    inhibited = false;
                }
                
                // checking if the neuron is in a refractory period
                if (timestamp - previousSpikeTime >= refractoryPeriod) {
                    active = true;
                }
				
                // updating current of synapses
                float total_current = 0;
                for (auto& synapse: dendriticTree) {
                    total_current += synapse->update(timestamp);
                }
                current = total_current;
                
                // eligibility trace decay
                eligibilityTrace *= fast_exp(-(timestamp-previousInputTime)/eligibilityDecay);
                
                // potential decay
                potential = restingPotential + (potential-restingPotential)*fast_exp(-(timestamp-previousInputTime)/membrane_time_constant);
                
                // threshold decay
                if (homeostasis) {
                    threshold = restingThreshold + (threshold-restingThreshold)*fast_exp(-(timestamp-previousInputTime)/decayHomeostasis);
                }
            
                if (active && !inhibited) {
					// calculating the potential
                    potential = restingPotential + current * (1 - fast_exp(-(timestamp-previousInputTime)/membrane_time_constant)) + (potential - restingPotential) * fast_exp(-(timestamp-previousInputTime)/membrane_time_constant);
                    
                    // updating the threshold
                    if (homeostasis) {
                        threshold += homeostasisBeta/decayHomeostasis;
                    }
					
                    // sending spike to relevant synapse
                    s->receiveSpike(timestamp);
                    
                    // integating synaptic currents
                    total_current = 0;
                    for (auto& synapse: dendriticTree) {
                        total_current += synapse->getSynapticCurrent();
                    }
                    current = total_current;

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
                        double predictedTimestamp = membrane_time_constant * (- fast_log2( - threshold + restingPotential + current) + fast_log2( current - potential + restingPotential)) + timestamp;
                        
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + s->getSynapseTimeConstant()) {
                            network->injectPredictedSpike(spike{predictedTimestamp, s, spikeType::prediction}, spikeType::prediction);
                        } else {
                            network->injectPredictedSpike(spike{timestamp + s->getSynapseTimeConstant(), s, spikeType::endOfIntegration}, spikeType::endOfIntegration);
                        }
                    } else {
                        potential = restingPotential + current * (1 - fast_exp(-(timestamp-previousInputTime)/membrane_time_constant)) + (potential - restingPotential);
                    }
                }
            } else if (type == spikeType::prediction) {
                if (active && !inhibited) {
                    potential = restingPotential + current * (1 - fast_exp(-(timestamp-previousInputTime)/membrane_time_constant)) + (potential - restingPotential);
                }
            } else if (type == spikeType::endOfIntegration) {
                if (active && !inhibited) {
                    potential = restingPotential + current * (1 - fast_exp(-s->getSynapseTimeConstant()/membrane_time_constant)) + (potential - restingPotential) * fast_exp(-s->getSynapseTimeConstant()/membrane_time_constant);
                }
            }
        
            if (network->getMainThreadAddon()) {
                network->getMainThreadAddon()->statusUpdate(timestamp, s, this, network);
            }

            if (potential >= threshold) {
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
                
                for (auto& axonTerminal : axonTerminals) {
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
            
            // updating current of synapses
            float total_current = 0;
            for (auto& synapse: dendriticTree) {
                total_current += synapse->update(timestamp);
            }
            current = total_current;
            
            // eligibility trace decay
            eligibilityTrace *= fast_exp(-timestep/eligibilityDecay);
            
			// potential decay
            potential = restingPotential + (potential-restingPotential)*fast_exp(-timestep/membrane_time_constant);
            
			// threshold decay
			if (homeostasis) {
                threshold = restingThreshold + (threshold-restingThreshold)*fast_exp(-timestep/decayHomeostasis);
			}
                
			// neuron inactive during refractory period
			if (active && !inhibited) {
				if (s) {
					// updating the threshold
					if (homeostasis) {
						threshold += homeostasisBeta/decayHomeostasis;
					}
                    
                    // sending spike to relevant synapse
                    s->receiveSpike(timestamp);
                    
                    // integating synaptic currents
                    total_current = 0;
                    for (auto& synapse: dendriticTree) {
                        total_current += synapse->getSynapticCurrent();
                    }
                    current = total_current;
                    
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
				
                potential += current * (1 - fast_exp(-timestep/membrane_time_constant));
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

				for (auto& axonTerminal: axonTerminals) {
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
                {"membrane_time_constant", membrane_time_constant},
                {"burstingActivity", burstingActivity},
                {"homeostasis", homeostasis},
                {"restingThreshold", restingThreshold},
                {"decayHomeostasis", decayHomeostasis},
                {"homeostasisBeta", homeostasisBeta},
                {"wta", wta},
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
		bool getActivity() const {
			return active;
		}
		
		float getMembraneTimeConstant() const {
            return membrane_time_constant;
        }
		
        void setMembraneTimeConstant(float newmembrane_time_constant) {
            membrane_time_constant = newmembrane_time_constant;
        }
		
		void setInhibition(double timestamp, bool inhibitionStatus) {
			inhibitionTime = timestamp;
			inhibited = inhibitionStatus;
		}
		
        void setRefractoryPeriod(float newRefractoryPeriod) {
            refractoryPeriod = newRefractoryPeriod;
        }
        
        void setBurstingActivity(bool newBool) {
            burstingActivity = newBool;
        }
        
        void setHomeostasis(bool newBool) {
            homeostasis = newBool;
        }
        
        void setRestingThreshold(float newThres) {
            restingThreshold = newThres;
        }
        
        void setDecayHomeostasis(float newDH) {
            decayHomeostasis = newDH;
        }
        
        void setHomeostasisBeta(float newHB) {
            homeostasisBeta = newHB;
        }
        
        void setWTA(bool newBool) {
            wta = newBool;
        }
        
	protected:
        
        // winner-take-all algorithm
		virtual void WTA(double timestamp, Network* network) override {
            for (auto& sub: network->getLayers()[layerID].sublayers) {
                // intra-sublayer hard WTA
                if (sub.ID == sublayerID) {
                    for (auto& n: sub.neurons) {
                        if (network->getNeurons()[n]->getNeuronID() != neuronID && network->getNeurons()[n]->getRfCoordinates() == rfCoordinates) {
                            network->getNeurons()[n]->setPotential(restingPotential);
                            if (LIF* neuron = dynamic_cast<LIF*>(network->getNeurons()[n].get())) {
                                dynamic_cast<LIF*>(network->getNeurons()[n].get())->current = 0;
                                dynamic_cast<LIF*>(network->getNeurons()[n].get())->inhibited = true;
                                dynamic_cast<LIF*>(network->getNeurons()[n].get())->inhibitionTime = timestamp;
                            }
                        }
                    }
                // inter-sublayer hard WTA
                } else {
                    for (auto& n: sub.neurons) {
                        if (network->getNeurons()[n]->getRfCoordinates() == rfCoordinates) {
                            network->getNeurons()[n]->setPotential(restingPotential);
                            if (LIF* neuron = dynamic_cast<LIF*>(network->getNeurons()[n].get())) {
                                dynamic_cast<LIF*>(network->getNeurons()[n].get())->current = 0;
                                dynamic_cast<LIF*>(network->getNeurons()[n].get())->inhibited = true;
                                dynamic_cast<LIF*>(network->getNeurons()[n].get())->inhibitionTime = timestamp;
                            }
                        }
                    }
                }
            }
		}
		
        // loops through any learning rules and activates them
        virtual void requestLearning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->getLearningStatus()) {
                if (!relevantAddons.empty()) {
                    for (auto& addon: relevantAddons) {
                        addon->learn(timestamp, s, postsynapticNeuron, network);
                    }
                }
            }
            if (wta) {
                WTA(timestamp, network);
            }
        }
        
		// ----- LIF PARAMETERS -----
		float                                    membrane_time_constant;
		bool                                     active;
		bool                                     inhibited;
		double                                   inhibitionTime;
		float                                    refractoryPeriod;
		bool                                     burstingActivity;
		bool                                     homeostasis;
		float                                    restingThreshold;
		float                                    decayHomeostasis;
		float                                    homeostasisBeta;
		bool                                     wta;
		Synapse*                                 activeSynapse;
	};
}
