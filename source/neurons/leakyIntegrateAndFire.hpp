/*
 * leakyIntegrateAndFire.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: LIF neuron model
 */

#pragma once

#include "../core.hpp"

namespace adonis
{
	class LeakyIntegrateAndFire : public PreNeuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		LeakyIntegrateAndFire(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={},  float _threshold=-50, float _restingPotential=-70, int _refractoryPeriod=3, float _decayCurrent=10, float _decayPotential=20, bool _burstingActivity=false, float _eligibilityDecay=20, float _inputResistance=50e9, float _externalCurrent=100 bool _homeostasis=false, float _decayHomeostasis=10, float _homeostasisBeta=1, bool _wta=false) :
			PreNeuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate),
    		refractoryPeriod(_refractoryPeriod),
    		potential(_restingPotential),
			learningRuleHandler(_learningRuleHandler),
			decayCurrent(_decayCurrent),
			decayPotential(_decayPotential),
			synapticEfficacy(1),
			threshold(_threshold),
			restingPotential(_restingPotential),
			inputResistance(_inputResistance),
			externalCurrent(_externalCurrent),
			current(0),
			active(true),
            lastSpikeTime(0),
            eligibilityTrace(0),
            eligibilityDecay(_eligibilityDecay),
			plasticityTrace(0),
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
		
		virtual ~LeakyIntegrateAndFire(){}
		
		// ----- PUBLIC LIF METHODS -----
		virtual void initialisation(Network* network) override
		{
			for (auto& rule: learningRuleHandler)
			{
				if(StandardAddOn* globalRule = dynamic_cast<StandardAddOn*>(rule))
				{
					if (std::find(network->getStandardAddOns().begin(), network->getStandardAddOns().end(), static_cast<StandardAddOn*>(rule)) == network->getStandardAddOns().end())
					{
						network->getStandardAddOns().emplace_back(static_cast<StandardAddOn*>(rule));
					}
				}
			}
		}
		
		virtual void update(double timestamp, axon* a, Network* network) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		virtual void updateSync(double timestamp, axon* a, Network* network) override
		{
			if (inhibited && timestamp - inhibitionTime >= refractoryPeriod)
			{
				inhibited = false;
			}

            if (timestamp - lastSpikeTime >= refractoryPeriod)
            {
                active = true;
            }

			// current decay
			current *= std::exp(-timestep/decayCurrent);
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
					a->lastInputTime = timestamp;
				}
				potential += (inputResistance*decayCurrent/(decayCurrent - decayPotential)) * current * (std::exp(-timestep/decayCurrent) - std::exp(-timestep/decayPotential));
			}

			if (a)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (a->preNeuron ? a->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << a->weight << " d=" << a->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> EMITTED" << std::endl;
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
				std::cout << "t=" << timestamp << " " << (activeAxon.preNeuron ? activeAxon.preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeAxon.weight << " d=" << activeAxon.delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> SPIKED" << std::endl;
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
					network->injectGeneratedSpike(spike{timestamp + p->delay, p.get()});
				}

				learn(timestamp, network);

				lastSpikeTime = timestamp;
				potential = resetPotential;
				if (!burstingActivity)
				{
					current = 0;
				}
				active = false;
			}
		}
		
		virtual void resetNeuron() override
		{
			lastSpikeTime = 0;
			current = 0;
			potential = restingPotential;
			eligibilityTrace = 0;
			inhibited = false;
			active = true;
			threshold = restingThreshold;
		}
		
		virtual void learn(double timestamp, Network* network)
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
				WTA(timestamp);
			}
			resetLearning();
		}
		
		void addAxon(Neuron* postNeuron, float weight=1., int delay=0, int probability=100, bool redundantConnections=true) override
        {
            if (postNeuron)
            {
            	if (connectionProbability(probability))
            	{
					if (redundantConnections == false)
					{
						int16_t ID = postNeuron->neuronID;
						auto result = std::find_if(postAxons.begin(), postAxons.end(), [ID](axon a){return a.postNeuron->neuronID == ID;});
						
						if (result == postAxons.end())
						{
							postAxons.emplace_back(this, postNeuron, weight*(1/inputResistance), delay, -1);
							postNeuron->preAxons.push_back(postAxons.back());
						}
						else
						{
							#ifndef NDEBUG
							std::cout << "axon " << neuronID << "->" << postNeuron->neuronID << " already exists" << std::endl;
							#endif
						}
					}
					else
					{
						postAxons.emplace_back(this, postNeuron, weight*(1/inputResistance), delay, -1);
						postNeuron->preAxons.push_back(postAxons.back());
					}
                }
            }
            else
            {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
		// ----- SETTERS AND GETTERS -----
		bool getActivity() const
		{
			return active;
		}
		
		float getThreshold() const
        {
            return threshold;
        }
		
        float setThreshold(float _threshold)
        {
            return threshold = _threshold;
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
		
		float getEligibilityTrace() const
		{
			return eligibilityTrace;
		}
		
		float getSynapticEfficacy() const
		{
			return synapticEfficacy;
		}
		
		float setSynapticEfficacy(float newEfficacy)
		{
			return synapticEfficacy = newEfficacy;
		}
		
		float getInputResistance() const
		{
			return inputResistance;
		}
		
		float getPlasticityTrace() const
		{
			return plasticityTrace;
		}
		
		void setPlasticityTrace(float newtrace)
		{
			plasticityTrace = newtrace;
		}
		
		double getLastSpikeTime() const
		{
			return lastSpikeTime;
		}
		
		void setInhibition(double timestamp, bool inhibitionStatus)
		{
			inhibitionTime = timestamp;
			inhibited = inhibitionStatus;
		}
		
		std::vector<LearningRuleHandler*> getLearningRuleHandler() const
		{
			return learningRuleHandler;
		}
		
		void addLearningRule(LearningRuleHandler* newRule)
		{
			learningRuleHandler.emplace_back(newRule);
		}
		
		float getPotential() const
        {
            return potential;
        }
		
        float setPotential(float newPotential)
        {
            return potential = newPotential;
        }
		
	protected:
		
		// ----- LIF BEHAVIOR -----
		void WTA(double timestamp, Network* network)
		{
			for (auto rf: network->getLayers()[layerID].sublayers[sublayerID].receptiveFields)
			{
				if (rf.row == rfRow && rf.col == rfCol)
				{
					for (auto n: rf.neurons)
					{
						if (network->getNeurons()[n].neuronID != neuronID)
						{
							network->getNeurons()[n].inhibited = true;
							network->getNeurons()[n].inhibitionTime = timestamp;
							network->getNeurons()[n].current = 0;
							network->getNeurons()[n].potential = restingPotential;
						}
					}
				}
			}
		}
		
		void resetLearning()
		{
			// resetting plastic neurons
			for (auto& inputAxon: preAxons)
			{
				inputAxon->preNeuron->eligibilityTrace = 0;
			}
		}
		
		// ----- LIF PARAMETERS -----
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    threshold;
        float                                    inputResistance;
        float                                    current;
		bool                                     active;
		bool                                     inhibited;
		double                                   inhibitionTime;
		float                                    potential;
		float                                    restingPotential;
		float                                    refractoryPeriod;
		std::vector<LearningRuleHandler*>        learningRuleHandler;
		float                                    synapticEfficacy;
		float                                    externalCurrent;
		float                                    eligibilityTrace;
		float                                    eligibilityDecay;
		bool                                     burstingActivity;
		bool                                     homeostasis;
		float                                    restingThreshold;
		float                                    decayHomeostasis;
		float                                    homeostasisBeta;
		bool                                     wta;
		axon                                     activeAxon;
        double                                   lastSpikeTime;
        float                                    plasticityTrace;
	};
}
