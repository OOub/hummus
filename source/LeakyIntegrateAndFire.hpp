/*
 * leakyIntegrateAndFire.hpp
 * Adonis - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: LIF neuron model
 */

#pragma once

#include "core.hpp"

namespace adonis
{
	struct spike;
	class LearningRuleHandler;
	
	class LeakyIntegrateAndFire : public BiologicalNeuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		LeakyIntegrateAndFire(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={},  float _threshold=-50, float _restingPotential=-70, int _refractoryPeriod=3, float _decayCurrent=10, float _decayPotential=20, bool _burstingActivity=false, float _eligibilityDecay=20, float _inputResistance=50e9, float _externalCurrent=100 bool _homeostasis=false, float _decayHomeostasis=10, float _homeostasisBeta=1, bool _wta=false){}
		
		virtual ~LeakyIntegrateAndFire(){}
		
		// ----- PUBLIC NEURON METHODS -----
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
	};
}
