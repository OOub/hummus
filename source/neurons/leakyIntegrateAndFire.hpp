/*
 * leakyIntegrateAndFire.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: LIF neuron model
 */

#pragma once

#include "../core.hpp"

namespace adonis
{
	class LIF : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		LIF(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={},  bool _homeostasis=false, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, bool _wta=false, bool _burstingActivity=false, float _eligibilityDecay=20, float _decayHomeostasis=10, float _homeostasisBeta=1, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9, float _externalCurrent=100) :
			Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _threshold, _restingPotential, _membraneResistance),
    		refractoryPeriod(_refractoryPeriod),
			decayCurrent(_decayCurrent),
			decayPotential(_decayPotential),
			externalCurrent(_externalCurrent),
			current(0),
			active(true),
            eligibilityDecay(_eligibilityDecay),
			burstingActivity(_burstingActivity),
			homeostasis(_homeostasis),
			restingThreshold(-50),
			decayHomeostasis(_decayHomeostasis),
			homeostasisBeta(_homeostasisBeta),
			inhibited(false),
			inhibitionTime(0),
			wta(_wta)
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
		
		virtual ~LIF(){}
		
		// ----- PUBLIC LIF METHODS -----
		virtual void initialisation(Network* network) override
		{
			for (auto& rule: learningRuleHandler)
			{
				if(StandardAddOn* globalRule = dynamic_cast<StandardAddOn*>(rule))
				{
					if (std::find(network->getStandardAddOns().begin(), network->getStandardAddOns().end(), dynamic_cast<StandardAddOn*>(rule)) == network->getStandardAddOns().end())
					{
						network->getStandardAddOns().emplace_back(dynamic_cast<StandardAddOn*>(rule));
					}
				}
			}
		}
        
		virtual void update(double timestamp, axon* a, Network* network) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		virtual void updateSync(double timestamp, axon* a, Network* network, double timestep) override
		{
			if (inhibited && timestamp - inhibitionTime >= refractoryPeriod)
			{
				inhibited = false;
			}

            if (timestamp - previousSpikeTime >= refractoryPeriod)
            {
                active = true;
            }

			// current decay
			current *= std::exp(-timestep/decayCurrent);
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
            
			// potential decay
			potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential);

            
			// threshold decay
			if (homeostasis)
			{
				threshold = restingThreshold + (threshold-restingThreshold)*exp(-timestep/decayHomeostasis);
			}

			// neuron inactive during refractory period
			if (active && !inhibited)
			{
				if (a)
				{
					// increase the threshold
					if (homeostasis)
					{
						threshold += homeostasisBeta/decayHomeostasis;
					}
					current += externalCurrent*a->weight;
					activeAxon = *a;
					a->previousInputTime = timestamp;
				}
				potential += (membraneResistance*decayCurrent/(decayCurrent - decayPotential)) * current * (std::exp(-timestep/decayCurrent) - std::exp(-timestep/decayPotential));
			}

			if (a)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> EMITTED" << std::endl;
				#endif
				for (auto addon: network->getStandardAddOns())
				{
					if (potential < threshold)
					{
						addon->incomingSpike(timestamp, a, network);
					}
				}
				if (network->getMainThreadAddOn())
				{
					network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
				}
			}
			else
			{
				for (auto addon: network->getStandardAddOns())
				{
					addon->timestep(timestamp, network, this);
				}
				if (network->getMainThreadAddOn())
				{
					network->getMainThreadAddOn()->timestep(timestamp, network, this);
				}
			}

			if (potential >= threshold)
			{
				eligibilityTrace = 1;
				plasticityTrace += 1;

				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (activeAxon.preNeuron ? activeAxon.preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeAxon.weight << " d=" << activeAxon.delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << " --> SPIKED" << std::endl;
				#endif

				for (auto addon: network->getStandardAddOns())
				{
					addon->neuronFired(timestamp, &activeAxon, network);
				}
				if (network->getMainThreadAddOn())
				{
					network->getMainThreadAddOn()->neuronFired(timestamp, &activeAxon, network);
				}

				for (auto& p : postAxons)
				{
                    network->injectGeneratedSpike(spike{timestamp + p.delay, &p});
				}

				learn(timestamp, network);

				previousSpikeTime = timestamp;
				potential = restingPotential;
				if (!burstingActivity)
				{
					current = 0;
				}
				active = false;
			}
		}
		
        virtual void resetNeuron() override
        {
            previousSpikeTime = 0;
            current = 0;
            potential = restingPotential;
            eligibilityTrace = 0;
            inhibited = false;
            active = true;
            threshold = restingThreshold;
        }
        
		// ----- SETTERS AND GETTERS -----
		bool getActivity() const
		{
			return active;
		}
		
		float getDecayPotential() const
        {
            return decayPotential;
        }
		
        float getDecayCurrent() const
        {
            return decayCurrent;
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
		
		void setInhibition(double timestamp, bool inhibitionStatus)
		{
			inhibitionTime = timestamp;
			inhibited = inhibitionStatus;
		}
		
	protected:
		
        // winner-take-all algorithm
		virtual void WTA(double timestamp, Network* network) override
		{
			for (auto rf: network->getLayers()[layerID].sublayers[sublayerID].receptiveFields)
			{
				if (rf.row == rfRow && rf.col == rfCol)
				{
					for (auto n: rf.neurons)
					{
                        if (network->getNeurons()[n]->getNeuronID() != neuronID)
						{
                            network->getNeurons()[n]->setPotential(restingPotential);
                            
                            if (LIF* neuron = dynamic_cast<LIF*>(network->getNeurons()[n].get()))
                            {
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
        virtual void learn(double timestamp, Network* network) override
        {
            if (network->getLearningStatus())
            {
                if (!learningRuleHandler.empty())
                {
                    for (auto& learningRule: learningRuleHandler)
                    {
                        learningRule->learn(timestamp, this, network);
                    }
                }
            }
            if (wta)
            {
                WTA(timestamp, network);
            }
        }
        
		// ----- LIF PARAMETERS -----
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    current;
		bool                                     active;
		bool                                     inhibited;
		double                                   inhibitionTime;
		float                                    refractoryPeriod;
		float                                    externalCurrent;
		float                                    eligibilityDecay;
		bool                                     burstingActivity;
		bool                                     homeostasis;
		float                                    restingThreshold;
		float                                    decayHomeostasis;
		float                                    homeostasisBeta;
		bool                                     wta;
		axon                                     activeAxon;
	};
}
