/*
 * globalLearningRuleHandler.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: The GlobalLearningRuleHandler class inherits from both learningRuleHandler and addOn to be able to make use of learning rules that make changes to the network on a global scale as well as locally on a neuron.
 */

#pragma once

#include "addOn.hpp"
#include "learningRuleHandler.hpp"

namespace adonis
{
	class GlobalLearningRuleHandler : public LearningRuleHandler, public StandardAddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		GlobalLearningRuleHandler() = default;
		virtual ~GlobalLearningRuleHandler(){}
	};
}
