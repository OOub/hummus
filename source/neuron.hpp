/*
 * neuron.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: The Neuron class defines a neuron and the learning rules dictating its behavior. Any modifications to add new learning rules or neuron types are to be done at this stage.
 *
 * To add a new learning rule:
 * 1- create your learning method in the protected section of this class
 * 2- update the learningMode enum with your learning rule
 * 3- the new learning rule should be called via the learningRuleHandler method in the protected section of this class
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
#include <memory>
#include <cmath>
#include <deque>

#include "networkDelegate.hpp"

namespace adonis_c
{
	class Neuron;
	
	enum learningMode
	{
	    noLearning,
	    myelinPlasticityNoReinforcement,
	    myelinPlasticityReinforcement,
	};
	
	struct projection
	{
		Neuron*     preNeuron;
		Neuron*     postNeuron;
		float       weight;
		float       delay;
		double      lastInputTime;
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
    	Neuron(int16_t _neuronID, int16_t _layerID, int16_t _rfID=0, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _eligibilityDecay=100, float _alpha=1, float _lambda=1, float _threshold=-50, float _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, int16_t _zCoordinate=-1, learningMode _learningType=noLearning) :
			neuronID(_neuronID),
			layerID(_layerID),
			rfID(_rfID),
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
			active(true),
			initialProjection{nullptr, nullptr, 1000e-10, 0, -1},
            lastSpikeTime(0),
            alpha(_alpha),
            lambda(_lambda),
            eligibilityTrace(0),
            eligibilityDecay(_eligibilityDecay),
            xCoordinate(_xCoordinate),
            yCoordinate(_yCoordinate),
            zCoordinate(_zCoordinate),
            learningType(_learningType)
    	{    	    
    		// error handling
			if (decayCurrent == decayPotential)
            {
                throw std::logic_error("The current decay and the potential decay cannot be equal: a division by 0 occurs");
            }
			
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
		
		// ----- PUBLIC NEURON METHODS -----
		void addProjection(Neuron* postNeuron, float weight, float delay, bool redundantConnections=true)
        {
            if (postNeuron)
            {
                if (redundantConnections == false)
                {
                    int16_t ID = postNeuron->neuronID;
                    auto result = std::find_if(postProjections.begin(), postProjections.end(), [ID](const std::unique_ptr<projection>& p){return p->postNeuron->neuronID == ID;});
                    
                    if (result == postProjections.end()) 
                    {
                        postProjections.emplace_back(std::unique_ptr<projection>(new projection{this, postNeuron, weight, delay, -1}));
                        postNeuron->preProjections.push_back(postProjections.back().get());
                    }
                    else
                    {
                        #ifndef NDEBUG
                        std::cout << "projection " << neuronID << "->" << postNeuron->neuronID << " already exists" << std::endl;
                        #endif
                    }
                }
                else
                {
                    postProjections.emplace_back(std::unique_ptr<projection>(new projection{this, postNeuron, weight, delay, -1}));
                    postNeuron->preProjections.push_back(postProjections.back().get());
                }
            }
            else
            {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
		template<typename Network>
		void update(double timestamp, float timestep, spike s, Network* network)
		{			
            if (timestamp - lastSpikeTime >= refractoryPeriod)
            {
                active = true;
            }
			
			// current decay
			current *= std::exp(-timestep/decayCurrent);
			eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
			
			// potential decay
			potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential);
			
			// impose spiking when using supervised learning
			if (network->getTeachingProgress())
			{
				if (network->getTeacher()->front().neuronID == neuronID && std::abs(network->getTeacher()->front().timestamp - timestamp) < 1e-1)
				{
					potential = threshold;
					network->getTeacher()->pop_front();
				}
			}
			
			// neuron inactive during refractory period
			if (active)
			{
				if (s.postProjection)
				{
					current += externalCurrent*s.postProjection->weight;
					activeProjection = *s.postProjection;
					s.postProjection->lastInputTime = timestamp;
				}
				potential += (inputResistance*decayCurrent/(decayCurrent - decayPotential)) * current * (std::exp(-timestep/decayCurrent) - std::exp(-timestep/decayPotential));
				supervisedPotential = potential;
			}
			
			if (s.postProjection)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (s.postProjection->preNeuron ? s.postProjection->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << s.postProjection->weight << " d=" << s.postProjection->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> EMITTED" << std::endl;
				#endif
				for (auto delegate: network->getStandardDelegates())
				{
					if (potential < threshold)
					{
						delegate->incomingSpike(timestamp, s.postProjection, network);
					}
				}
				if (network->getMainThreadDelegate())
				{
					network->getMainThreadDelegate()->incomingSpike(timestamp, s.postProjection, network);
				}
			}
			
			else
			{
				for (auto delegate: network->getStandardDelegates())
				{
					delegate->timestep(timestamp, network, this);
				}
				if (network->getMainThreadDelegate())
				{
					network->getMainThreadDelegate()->timestep(timestamp, network, this);
				}
			}
			
			if (potential >= threshold)
			{
				eligibilityTrace = 1;
				
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (activeProjection.preNeuron ? activeProjection.preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeProjection.weight << " d=" << activeProjection.delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> SPIKED" << std::endl;
				#endif
				
				for (auto delegate: network->getStandardDelegates())
				{
					delegate->neuronFired(timestamp, &activeProjection, network);
				}
				if (network->getMainThreadDelegate())
				{
					network->getMainThreadDelegate()->neuronFired(timestamp, &activeProjection, network);
				}
				
				for (auto& p : postProjections)
				{
					network->injectGeneratedSpike(spike{timestamp + p->delay, p.get()});
				}
			
				learningRuleHandler(timestamp, network);
				lastSpikeTime = timestamp;
				potential = resetPotential;
				supervisedPotential = resetPotential;
				current = 0;
				active = false;
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
	
    	// ----- SETTERS AND GETTERS -----
		int16_t getNeuronID() const
        {
            return neuronID;
        }
		
		int16_t getLayerID() const
		{
			return layerID;
		}
		
	    int16_t getRFID() const
		{
			return rfID;
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
		
		int16_t getX() const
		{
		    return xCoordinate;
		}
		
		int16_t getY() const
		{
		    return yCoordinate;
		}
		
		int16_t getZ() const
		{
		    return zCoordinate;
		}
		
		std::vector<std::unique_ptr<projection>>& getPostProjections()
		{
			return postProjections;
		}
		
		float getEligibilityTrace() const
		{
			return eligibilityTrace;
		}
		
		projection* getInitialProjection()
		{
			return &initialProjection;
		}
		
	protected:
		
		// ----- THE LEARNING RULES -----
		
		// this is the method called by the update function. Any new learning rule methods should be called at this stage
		template<typename Network>
		void learningRuleHandler(double timestamp, Network* network)
		{
			if (network->getLearningStatus())
			{
				if (learningType == myelinPlasticityNoReinforcement || learningType == myelinPlasticityReinforcement)
				{
					myelinPlasticity(timestamp, network);
				}
			}
			lateralInhibition(network);
			resetLearning(network);
		}
		
		template<typename Network>
		void myelinPlasticity(double timestamp, Network* network)
		{
			std::vector<double> timeDifferences;
			std::vector<int16_t> plasticID;
			std::vector<std::vector<int16_t>> plasticCoordinates(3);
			
			#ifndef NDEBUG
			std::cout << "New learning epoch at t=" << timestamp << std::endl;
			#endif
			
			for (auto inputProjection: preProjections)
			{
				// selecting plastic neurons
				if (inputProjection->preNeuron->eligibilityTrace > 0.1)
				{
					plasticID.push_back(inputProjection->preNeuron->neuronID);
					plasticCoordinates[0].push_back(inputProjection->preNeuron->xCoordinate);
					plasticCoordinates[1].push_back(inputProjection->preNeuron->yCoordinate);
					plasticCoordinates[2].push_back(inputProjection->preNeuron->rfID);

					float change = 0;
					timeDifferences.push_back(timestamp - inputProjection->lastInputTime - inputProjection->delay);
					if (network->getTeacher()) // supervised learning
					{
						if (network->getTeachingProgress()) // stops learning when the teacher signal is over
						{
							if (timeDifferences.back() > 0)
							{
								change = lambda*(inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(-alpha*timeDifferences.back()/decayCurrent) - std::exp(-alpha*timeDifferences.back()/decayPotential))*synapticEfficacy;
								inputProjection->delay += change;
								#ifndef NDEBUG
								std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << " weight: " << inputProjection->weight << std::endl;
								#endif
							}

							else if (timeDifferences.back() < 0)
							{
								change = -lambda*((inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(alpha*timeDifferences.back()/decayCurrent) - std::exp(alpha*timeDifferences.back()/decayPotential)))*synapticEfficacy;
								inputProjection->delay += change;
								#ifndef NDEBUG
								std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << " weight: " << inputProjection->weight << std::endl;
								#endif
							}
						}
					}
					else // unsupervised learning
					{
						if (timeDifferences.back() > 0)
						{
							change = lambda*(inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(-alpha*timeDifferences.back()/decayCurrent) - std::exp(-alpha*timeDifferences.back()/decayPotential))*synapticEfficacy;
							inputProjection->delay += change;
							#ifndef NDEBUG
							std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
							#endif
						}

						else if (timeDifferences.back() < 0)
						{
							change = -lambda*((inputResistance/(decayCurrent-decayPotential)) * current * (std::exp(alpha*timeDifferences.back()/decayCurrent) - std::exp(alpha*timeDifferences.back()/decayPotential)))*synapticEfficacy;
							inputProjection->delay += change;
							#ifndef NDEBUG
							std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
							#endif
						}
					}
					synapticEfficacy = -std::exp(-std::pow(timeDifferences.back(),2))+1;
				}
			}
			
			for (auto delegate: network->getStandardDelegates())
			{
				delegate->learningEpoch(timestamp, network, this, timeDifferences, plasticCoordinates);
			}
			if (network->getMainThreadDelegate())
			{
				network->getMainThreadDelegate()->learningEpoch(timestamp, network, this, timeDifferences, plasticCoordinates);
			}
			
			if (learningType == myelinPlasticityReinforcement)
			{
				reinforcementLearning(plasticID, network);
			}
		}
		
		template <typename Network>
		void reinforcementLearning(std::vector<int16_t> plasticID, Network* network)
		{
			// looping through all projections from the winner
            for (auto& allProjections: preProjections)
            {
                int16_t ID = allProjections->preNeuron->neuronID;
                // if the projection is plastic
                if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end())
                {
					
					// positive reinforcement on winner projections
					if (supervisedPotential < threshold && allProjections->weight < 19e-10/plasticID.size())
                    {
                     	allProjections->weight += allProjections->weight*synapticEfficacy*plasticID.size()*0.01;
                    }
                }
                else
                {
                    if (allProjections->weight > 0)
                    {
                        // negative reinforcement on other projections going towards the winner to prevent other neurons from triggering it
                        allProjections->weight -= allProjections->weight*synapticEfficacy*plasticID.size()*0.01;
						if (allProjections->weight < 0)
						{
							allProjections->weight = 0;
						}
                    }
				}
            }
		}
		
		// problem with the lateral inhibition
		template<typename Network>
		void lateralInhibition(Network* network)
		{
			if (preProjections.size() > 0)
			{
				for (auto& projReset: preProjections[0]->preNeuron->postProjections)
				{
					if (projReset->postNeuron->neuronID != neuronID)
					{
						projReset->postNeuron->active = false;
						projReset->postNeuron->setCurrent(0);
						projReset->postNeuron->setPotential(restingPotential);
					}
				}
			}
		}
		
		template<typename Network>
		void resetLearning(Network* network)
		{
			// clearing generated spike list
			for (auto i=0; i<network->getGeneratedSpikes().size(); i++)
			{
				if (network->getGeneratedSpikes()[i].postProjection->postNeuron->getLayerID() == layerID && network->getGeneratedSpikes()[i].postProjection->postNeuron->getRFID() == rfID)
				{
					network->getGeneratedSpikes().erase(network->getGeneratedSpikes().begin()+i);
				}
			}
			
			// resetting plastic neurons
			for (auto& inputProjection: preProjections)
			{
				inputProjection->preNeuron->eligibilityTrace = 0;
			}
		}
		
		// ----- NEURON PARAMETERS -----
		int16_t                                  neuronID;
		int16_t                                  layerID;
		int16_t                                  rfID;
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    refractoryPeriod;
        float                                    threshold;
        float                                    restingPotential;
        float                                    resetPotential;
        float                                    inputResistance;
        float                                    current;
        float                                    potential;
		bool                                     active;
		float                                    synapticEfficacy;
		float                                    externalCurrent;
		float                                    alpha;
		float                                    lambda;
		float                                    eligibilityTrace;
		float                                    eligibilityDecay;
		int16_t                                  xCoordinate;
		int16_t                                  yCoordinate;
		int16_t                                  zCoordinate;
		
		// ----- IMPLEMENTATION VARIABLES -----
		projection                               activeProjection;
        std::vector<std::unique_ptr<projection>> postProjections;
        std::vector<projection*>                 preProjections;
        projection                               initialProjection;
        double                                   lastSpikeTime;
        float                                    supervisedPotential;
        learningMode                             learningType;
    };
}
