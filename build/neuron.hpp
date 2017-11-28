/*
 * neuron.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/11/2017
 *
 * Information:
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

namespace baal
{
	class Neuron;
	
	struct projection
	{
		Neuron*     preNeuron;
		Neuron*     postNeuron;
		float       weight;
		float       delay;
		bool        isInitial;
	};
	
	struct spike
    {
        float       timestamp;
        projection* postProjection;
    };
	
    class Neuron
    {
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
    	Neuron(int16_t _neuronID, int16_t _layerID, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _decaySynapticEfficacy=1, float _synapticEfficacy=1, float _threshold=-50, float _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1e-10, float _currentBurnout=3.1e-9) :
			neuronID(_neuronID),
			layerID(_layerID),
			decayCurrent(_decayCurrent),
			decayPotential(_decayCurrent),
			refractoryPeriod(_refractoryPeriod),
			decaySynapticEfficacy(_decaySynapticEfficacy),
			synapticEfficacy(_synapticEfficacy),
			threshold(_threshold),
			restingPotential(_restingPotential),
			resetPotential(_resetPotential),
			inputResistance(_inputResistance),
			externalCurrent(_externalCurrent),
			currentBurnout(_currentBurnout),
			current(0),
			potential(_restingPotential),
			activity(false),
			initialProjection{nullptr, nullptr, 1, 0, true},
			fireCounter(0),
            timeStart(0),
            timeEnd(0),
            previousTimestamp(0),
            lastSpikeTime(0)
    	{
			if (decayCurrent == 0)
            {
                throw std::logic_error("The current decay cannot be 0");
            }
			
    	    if (decayPotential == 0)
            {
                throw std::logic_error("The potential decay cannot be 0");
            }
		}
		
		Neuron(const Neuron&) = delete; // copy constructor
        Neuron(Neuron&&) = default; // move constructor
        Neuron& operator=(const Neuron&) = delete; // copy assign constructor
        Neuron& operator=(Neuron&&) = default; // move assign constructor
		
		virtual ~Neuron()
		{
			if (counterLog)
            {
                *counterLog << "Neuron " << neuronID << " fired " << fireCounter << " times\n";
            }
		}
		
		// ----- PUBLIC NEURON METHODS -----
		void addProjection(Neuron* postNeuron, float weight, float delay)
        {
            if (postNeuron)
            {
                postProjections.emplace_back(std::unique_ptr<projection>(new projection{this, postNeuron, weight, delay, false}));
                postNeuron->preProjections.push_back(postProjections.back().get());
            }
            else
            {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
		template<typename Network>
		void update(float timestamp, float timestep, spike s, Network* network)
		{
			if (s.postProjection && s.postProjection->isInitial)
            {
                network->getPlasticTime().push_back(timestamp);
                network->getPlasticNeurons().push_back(this);
            }
			
            if (timestamp - lastSpikeTime >= refractoryPeriod)
            {
                activity = false;
            }

			// potential decay equation
//			potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential); // omar equation
			potential += (restingPotential - potential)*(timestep/decayPotential); // fabian equation
			
			// neuron inactive during refractory period
			if (current > currentBurnout)
			{
				current = currentBurnout;
			}
			
			if (!activity)
			{
//				potential += inputResistance*current; // omar equation
				potential += inputResistance*current*(timestep/decayPotential); // fabian equation
			}
			
			if (s.postProjection)
			{
//				current += externalCurrent*s.postProjection->weight; // omar equation
				current += (-current+externalCurrent)*s.postProjection->weight*(timestep/decayCurrent); // fabian equation
				activeProjection = *s.postProjection;
			}
//			current *= std::exp(-timestep/decayCurrent); // omar equation
			else // fabian equation
			{
				current += -current*(timestep/decayCurrent);
			}
			previousTimestamp = timestamp;

		
			if (s.postProjection)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (s.postProjection->preNeuron ? s.postProjection->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << s.postProjection->weight << " d=" << s.postProjection->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> EMITTED" << std::endl;
				#endif
				for (auto delegate: network->getDelegates())
				{
					delegate->getArrivingSpike(timestamp, s.postProjection, false, false, network, this);
				}
			}
			else
			{
				for (auto delegate: network->getDelegates())
				{
					delegate->getArrivingSpike(timestamp, nullptr, false, true, network, this);
				}
			}
			
			if (potential >= threshold)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (activeProjection.preNeuron ? activeProjection.preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeProjection.weight << " d=" << activeProjection.delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> SPIKED" << std::endl;
				#endif
				for (auto delegate: network->getDelegates())
				{
					delegate->getArrivingSpike(timestamp, &activeProjection, true, false, network, this);
				}
			
				for (auto& p : postProjections)
				{
					network->injectGeneratedSpike(spike{timestamp + p->delay, p.get()});
				}

				if (counterLog)
				{
					if (timestamp >= timeStart && timestamp <= timeEnd)
					{
						fireCounter++;
					}
				}
				
				delayLearning(network);
				
				lastSpikeTime = timestamp;
				potential = resetPotential;
				current = 0;
				activity = true;
			}
		}
		
		spike prepareInitialSpike(float timestamp)
        {
            if (!initialProjection.postNeuron)
            {
                initialProjection.postNeuron = this;
            }
            return spike{timestamp, &initialProjection};
        }
		
        void spikeCountLogger(float _timeStart, float _timeEnd, std::string filename)
        {
            timeStart = _timeStart;
            timeEnd = _timeEnd;
            counterLog.reset(new std::ofstream(filename));
        }
		
    	// ----- SETTERS AND GETTERS -----
		int16_t getNeuronID() const
        {
            return neuronID;
        }
		
		int16_t getLayerID() const
		{
			return layerID;
		}
		
		float getThreshold() const
        {
            return threshold;
        }
		
        float setThreshold(float _threshold)
        {
            return threshold = _threshold;
        }
		
        float getPotential() const
        {
            return potential;
        }
		
        float setPotential(float newPotential)
        {
            return potential = newPotential;
        }
		
        float getCurrent() const
        {
        	return current;
		}
		
		
    	// ----- PROTECTED NEURON METHODS -----
    	template<typename Network>
		void delayLearning(Network* network)
		{
			if (layerID != 0)
			{
				std::vector<float> timeDifferences;
				float tMax = *std::max_element(network->getPlasticTime().begin(), network->getPlasticTime().end());

				#ifndef NDEBUG
				std::cout << "patterns contains " << network->getInputSpikeCounter() << " spikes" << std::endl;
				std::cout << "max time is: " << tMax << std::endl;
				#endif

				// looping through presynaptic neurons belonging to the pattern
				int cpt = 0;
				for (auto& plasticNeurons: network->getPlasticNeurons())
				{
					// looping through each presynaptic neuron's projections
					for (auto& plasticProjections: plasticNeurons->postProjections)
					{
						// checking if it's the winner postNeuron
						if (plasticProjections->postNeuron->getNeuronID() == this->getNeuronID())
						{
							timeDifferences.push_back(tMax - network->getPlasticTime().at(cpt) - plasticProjections->delay);
							// std::cout << "ts: " << network->getPlasticTime().at(cpt) << " tdif: " << timeDifferences.back() << std::endl;
							// delay learning rule
							if (timeDifferences.back() > 0)
							{
								plasticProjections->delay += ((inputResistance*plasticProjections->weight)/decayPotential) * std::exp(-timeDifferences.back()/decayPotential) * (synapticEfficacy *(-std::exp(-std::pow(timeDifferences.back(),2)) + 1));
								#ifndef NDEBUG
								std::cout << "time diff: " << timeDifferences.back() << " delay change: " << (-(inputResistance*plasticProjections->weight)/decayPotential) * std::exp(-timeDifferences.back()/decayPotential) * - synapticEfficacy << std::endl;
								#endif
								std::cout << plasticProjections->delay << std::endl;
							}
							else if (timeDifferences.back() < 0)
							{
								plasticProjections->delay += (-(inputResistance*plasticProjections->weight)/decayPotential) * std::exp(timeDifferences.back()/decayPotential) * (synapticEfficacy *(-std::exp(-std::pow(timeDifferences.back(),2)) + 1));
								#ifndef NDEBUG
								std::cout << "time diff: " << timeDifferences.back() << " delay change: " << (-(inputResistance*plasticProjections->weight)/decayPotential) * std::exp(-timeDifferences.back()/decayPotential) * synapticEfficacy << std::endl;
								#endif
//								std::cout << plasticProjections->delay << std::endl;
							}
							else
							{
								plasticProjections->delay += (inputResistance*plasticProjections->weight) * (synapticEfficacy *(-std::exp(-std::pow(timeDifferences.back(),2)) + 1));
//								std::cout << plasticProjections->delay << std::endl;
							}
						}
					}
					cpt++;
				}

				if (decaySynapticEfficacy > 0)
				{
					synapticEfficacy *= std::exp(-1/decaySynapticEfficacy);
				}
			
				resetAfterLearning(network);
			
			}
		}
		
		template <typename Network>
        void resetAfterLearning(Network* network)
        {
			network->getPlasticTime().clear();
			network->getPlasticNeurons().clear();
			network->getGeneratedSpikes().clear();
			network->setInputSpikeCounter(0);
        }
		
		// ----- NEURON PARAMETERS -----
		int16_t                                  neuronID;
		int16_t                                  layerID;
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    refractoryPeriod;
        float                                    decaySynapticEfficacy;
        float                                    threshold;
        float                                    restingPotential;
        float                                    resetPotential;
        float                                    inputResistance;
        float                                    current;
        float                                    potential;
        bool                                     activity;
		float                                    synapticEfficacy;
		float                                    externalCurrent;
		float                                    currentBurnout;
		
		// ----- IMPLEMENTATION VARIABLES -----
		projection                               activeProjection;
        std::vector<std::unique_ptr<projection>> postProjections;
        std::vector<projection*>                 preProjections;
        projection                               initialProjection;
		int                                      fireCounter;
        float                                    timeStart;
        float                                    timeEnd;
        std::unique_ptr<std::ofstream>           counterLog;
        float                                    previousTimestamp;
        float                                    lastSpikeTime;
    };
}
