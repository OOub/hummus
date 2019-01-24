/*
 * globalLearningRuleHandler.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: The GlobalLearningRuleHandler class inherits from both learningRuleHandler and addOn to be able to make use of learning rules that make changes to the network on a global scale as well as locally on a neuron. This class enables learning rules inheriting from it to have access to the addOn messages such as the onStart method which enables us to initialise a learning rule before the network starts running.
 */

#pragma once

#include "addOn.hpp"
#include "learningRuleHandler.hpp"

namespace adonis
{
	class GlobalLearningRuleHandler : public LearningRuleHandler, public AddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		GlobalLearningRuleHandler() = default;
		virtual ~GlobalLearningRuleHandler(){}
	};
}
