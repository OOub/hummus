/*
 * neuron.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: the neuron class defines a neuron and the learning rules dictating its behavior.
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

#include "networkDelegate.hpp"

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
        double      timestamp;
        projection* postProjection;
    };
	
    class Neuron
    {
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
    	Neuron(int16_t _neuronID, int16_t _layerID, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _alpha=1, float _lambda=1, float _threshold=-50, float _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1) :
			neuronID(_neuronID),
			layerID(_layerID),
			decayCurrent(_decayCurrent),
			decayPotential(_decayPotential),
			refractoryPeriod(_refractoryPeriod),
			synapticEfficacy(1),
			threshold(_threshold),
			restingPotential(_restingPotential),
			resetPotential(_resetPotential),
			inputResistance(_inputResistance),
			externalCurrent(_externalCurrent),
			current(0),
			potential(_restingPotential),
			supervisedPotential(_restingPotential),
			activity(false),
			initialProjection{nullptr, nullptr, 1000e-10, 0, true},
			fireCounter(0),
            timeStart(0),
            timeEnd(0),
            lastSpikeTime(0),
            alpha(_alpha),
            lambda(_lambda)
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
		void update(double timestamp, float timestep, spike s, Network* network)
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
			
			// current decay
			current *= std::exp(-timestep/decayCurrent);
			
			// potential decay
			potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential);
			
			// neuron inactive during refractory period
			if (!activity)
			{
				if (s.postProjection)
				{
					current += externalCurrent*s.postProjection->weight;
					activeProjection = *s.postProjection;
				}
				potential += (inputResistance*decayCurrent/(decayCurrent - decayPotential)) * current * (std::exp(-timestep/decayCurrent) - std::exp(-timestep/decayPotential));
				supervisedPotential = potential;
			}
			
			// impose spiking when using supervised learning
			if (network->getTeacher())
			{
				if (network->getTeacherIterator() < (*network->getTeacher())[0].size())
				{
					if ((*network->getTeacher())[1][network->getTeacherIterator()] == neuronID)
					{
						if (std::abs((*network->getTeacher())[0][network->getTeacherIterator()] - timestamp) < 1e-1)
						{
							current = 19e-10;
							potential = threshold;
							network->setTeacherIterator(network->getTeacherIterator()+1);
						}
					}
				}
				else
				{
					if (network->getTeachingProgress())
					{
						network->setTeachingProgress(false);
						std::cout << "learning stopped at t=" << timestamp << std::endl;
					}
				}
			}
			
			if (potentialLog)
			{
				*potentialLog << timestamp << " " << potential << "\n";
			}
			
			if (s.postProjection)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (s.postProjection->preNeuron ? s.postProjection->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << s.postProjection->weight << " d=" << s.postProjection->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> EMITTED" << std::endl;
				#endif
				for (auto delegate: network->getDelegates())
				{
					if (potential < threshold)
					{
						delegate->getArrivingSpike(timestamp, s.postProjection, false, false, network, this);
					}
				}
			}
			
			else
			{
				for (auto delegate: network->getDelegates())
				{
					if (delegate->getMode() == NetworkDelegate::Mode::display)
					{
						delegate->getArrivingSpike(timestamp, nullptr, false, true, network, this);
					}
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
				delayLearning(timestamp, network);
				lastSpikeTime = timestamp;
				potential = resetPotential;
				supervisedPotential = resetPotential;
				current = 0;
				activity = true;
			}
		}
		
		spike prepareInitialSpike(double timestamp)
        {
            if (!initialProjection.postNeuron)
            {
                initialProjection.postNeuron = this;
            }
            return spike{timestamp, &initialProjection};
        }
		
        void spikeCountLogger(double _timeStart, double _timeEnd, std::string filename)
        {
            timeStart = _timeStart;
            timeEnd = _timeEnd;
            counterLog.reset(new std::ofstream(filename));
        }
		
        void potentialLogger(std::string filename)
        {
        	std::cout << "logging the potential of neuron " << neuronID << std::endl;
        	potentialLog.reset(new std::ofstream(filename));
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
		
		void setCurrent(float newCurrent)
		{
			current = newCurrent;
		}
		
		float getExternalCurrent() const
		{
			return externalCurrent;
		}
		
		void setExternalCurrent(float newCurrent)
		{
			externalCurrent = newCurrent;
		}
	
	protected:
	
    	// ----- PROTECTED NEURON METHODS -----

    	template<typename Network>
		void delayLearning(double timestamp, Network* network)
		{
			if (network->getInputSpikeCounter() != 0)
			{
				if (layerID != 0)
				{
					#ifndef NDEBUG
					if (network->getTeachingProgress())
					{
						std::cout << "learning epoch" << std::endl;
					}
					#endif
					std::vector<double> timeDifferences;
					float tMax = 0;
					// if there is a supervised signal
					if (network->getTeacher())
					{
						tMax = timestamp;
					}
					
					// if there is no supervised signal
					else
					{
						tMax = *std::max_element(network->getPlasticTime().begin(), network->getPlasticTime().end());
					}
					#ifndef NDEBUG
					std::cout << "patterns contains " << network->getInputSpikeCounter() << " spikes" << std::endl;
					#endif
					// looping through presynaptic neurons belonging to the pattern
					int cpt = 0;
					for (auto& plasticNeurons: network->getPlasticNeurons())
					{
						// looping through each presynaptic neuron's projections
						for (auto& plasticProjections: plasticNeurons->postProjections)
						{
							// checking if it's the winner postNeuron
							if (plasticProjections->postNeuron->getNeuronID() == neuronID)
							{
								timeDifferences.push_back(tMax - network->getPlasticTime().at(cpt) - plasticProjections->delay);
								if (network->getTeacher()) // if supervised learning
								{
									if (network->getTeachingProgress())
									{
										// delay convergence equations
										if (timeDifferences.back() > 0)
										{
											float change = lambda*(inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(-alpha*timeDifferences.back()/decayCurrent) - std::exp(-alpha*timeDifferences.back()/decayPotential))*synapticEfficacy;
											plasticProjections->delay += change;
											#ifndef NDEBUG
											std::cout << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
											#endif
											if (network->getLearningLog())
											{
												*network->getLearningLog() << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " " << timeDifferences.back() << " " << change << " " << plasticProjections->weight << "\n";
											}
										}

										else if (timeDifferences.back() < 0)
										{
											float change = -lambda*((inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(alpha*timeDifferences.back()/decayCurrent) - std::exp(alpha*timeDifferences.back()/decayPotential)))*synapticEfficacy;
											plasticProjections->delay += change;
											#ifndef NDEBUG
											std::cout << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
											#endif
											if (network->getLearningLog())
											{
												*network->getLearningLog() << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " " << timeDifferences.back() << " " << change << " " << plasticProjections->weight << "\n";
											}
										}

										else
										{
											plasticProjections->delay += 0;
											#ifndef NDEBUG
											std::cout << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << 0 << std::endl;
											#endif
											if (network->getLearningLog())
											{
												*network->getLearningLog() << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " " << timeDifferences.back() << " " << 0 << " " << plasticProjections->weight << "\n";
											}
										}
									}
								}
								else // if unsupervised learning
								{
									// delay convergence equations
									if (timeDifferences.back() > 0)
									{
										float change = lambda*(inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(-alpha*timeDifferences.back()/decayCurrent) - std::exp(-alpha*timeDifferences.back()/decayPotential))*synapticEfficacy;
										plasticProjections->delay += change;
										#ifndef NDEBUG
										std::cout << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
										#endif
										if (network->getLearningLog())
										{
											*network->getLearningLog() << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " " << timeDifferences.back() << " " << change << " " << plasticProjections->weight << "\n";
										}
									}

									else if (timeDifferences.back() < 0)
									{
										float change = -lambda*((inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(alpha*timeDifferences.back()/decayCurrent) - std::exp(alpha*timeDifferences.back()/decayPotential)))*synapticEfficacy;
										plasticProjections->delay += change;
										#ifndef NDEBUG
										std::cout << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
										#endif
										if (network->getLearningLog())
										{
											*network->getLearningLog() << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " " << timeDifferences.back() << " " << change << " " << plasticProjections->weight << "\n";
										}
									}

									else
									{
										plasticProjections->delay += 0;
										#ifndef NDEBUG
										std::cout << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << 0 << std::endl;
										#endif
										if (network->getLearningLog())
										{
											*network->getLearningLog() << plasticProjections->preNeuron->getNeuronID() << " " << plasticProjections->postNeuron->getNeuronID() << " " << timeDifferences.back() << " " << 0 << " " << plasticProjections->weight << "\n";
										}
									}
								}
								synapticEfficacy = -std::exp(-std::pow(timeDifferences.back(),2))+1;
							}
						}
						cpt++;
					}
					weightReinforcement(network);
					resetAfterLearning(network);
				}
			}
		}
		
		template <typename Network>
        void resetAfterLearning(Network* network)
        {
			network->getPlasticTime().clear();
			network->getPlasticNeurons().clear();
//			network->getGeneratedSpikes().clear();
			network->setInputSpikeCounter(0);
			
			if (!network->getTeacher())
			{
				// lateral inhibition
				for (auto& projReset: preProjections[0]->preNeuron->postProjections)
				{
					if (projReset->postNeuron->neuronID != neuronID)// && projReset->postNeuron->layerID == layerID)
					{
						projReset->postNeuron->setPotential(restingPotential);
						projReset->postNeuron->setCurrent(0);
					}
				}
			}
        }
		
		template <typename Network>
        void weightReinforcement(Network* network) // this needs to be different OR we cannot simply align to the last value9
        {
            std::vector<int> plasticID;
            for (auto& plasticNeurons: network->getPlasticNeurons())
            {
                plasticID.push_back(plasticNeurons->getNeuronID());
            }
			
            // looping through all projections from the winner exclude the postProjections
            for (auto& allProjections: this->preProjections)
            {
                int16_t ID = allProjections->preNeuron->getNeuronID();
                // if the projection is plastic
                if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end())
                {
					// positive reinforcement
					if (supervisedPotential < threshold && allProjections->weight <= plasticID.size())
                    {
                     	allProjections->weight += plasticID.size()*0.01;
                    }
                }
                else
                {
                    if (allProjections->weight > 0)
                    {
                        // negative reinforcement
                        allProjections->weight -= plasticID.size()*0.01;
						if (allProjections->weight < 0)
						{
							allProjections->weight = 0;
						}
                    }
				}
            }
        }
		
		// ----- NEURON PARAMETERS -----
		int16_t                                  neuronID;
		int16_t                                  layerID;
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    refractoryPeriod;
        float                                    threshold;
        float                                    restingPotential;
        float                                    resetPotential;
        float                                    inputResistance;
        float                                    current;
        float                                    potential;
        bool                                     activity;
		float                                    synapticEfficacy;
		float                                    externalCurrent;
		float                                    alpha;
		float                                    lambda;
		
		// ----- IMPLEMENTATION VARIABLES -----
		projection                               activeProjection;
        std::vector<std::unique_ptr<projection>> postProjections;
        std::vector<projection*>                 preProjections;
        projection                               initialProjection;
		int                                      fireCounter;
        double                                   timeStart;
        double                                   timeEnd;
        std::unique_ptr<std::ofstream>           counterLog;
        std::unique_ptr<std::ofstream>           potentialLog;
        double                                   lastSpikeTime;
        float                                    supervisedPotential;
    };
}
