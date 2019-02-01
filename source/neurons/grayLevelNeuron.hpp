/*
 * grayLevelNeuron.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/01/2019
 *
 * Information: experimental neurons that calculate gray levels from events originating from a neuromorphic camera
 */

#pragma once

#include "../core.hpp"

namespace adonis {
	class GrayLevelNeuron : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		GrayLevelNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9) :
                Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _eligibilityDecay, _threshold, _restingPotential, _membraneResistance){}
		
		virtual ~GrayLevelNeuron(){}
		
		// ----- PUBLIC GRAY LEVEL NEURON METHODS -----
		
		void update(double timestamp, axon* a, Network* network) override {
		}
        
    
	};
}
