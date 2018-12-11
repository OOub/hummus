/*
 * globalLearningRuleHandler.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: The GlobalLearningRuleHandler class inherits from both learningRuleHandler and NetworkDelegate to be able to make use of learning rules that make changes to the network on a global scale as well as locally on a neuron.
 */

#pragma once

#include "networkAddOn.hpp"
#include "learningRuleHandler.hpp"

namespace adonis_c
{
	class GlobalLearningRuleHandler : public LearningRuleHandler, public StandardNetworkAddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		GlobalLearningRuleHandler() = default;
		virtual ~GlobalLearningRuleHandler(){}
	};
}
