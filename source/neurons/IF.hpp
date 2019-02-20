/*
 * IF.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 19/02/2019
 *
 * Information: integrate-and-fire neuron without any leak
 */

#pragma once

#include "../core.hpp"

namespace adonis {
	class IF : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		IF(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _eligibilityDecay=20, float _threshold=1, float _restingPotential=0, float _membraneResistance=1) :
                Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _eligibilityDecay, _threshold, _restingPotential, _membraneResistance){}
		
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
		
		void update(double timestamp, axon* a, Network* network, spikeType type) override {
            
		}
        
        void updateSync(double timestamp, axon* a, Network* network, double timestep) override {
            
        }
	};
}
