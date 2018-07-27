/*
 * spikeLogger.hpp
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on to the Network class, used to write the spiking neural network output into a log binary file. This can be read using the snnReader.m matlab function
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>


#include "network.hpp"

namespace adonis_t
{
    class SpikeLogger : public NetworkDelegate
    {
    public:
    	// ----- CONSTRUCTOR -----
        SpikeLogger(std::string filename)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC SPIKE LOGGER METHODS -----
		void incomingSpike(double timestamp, projection* p, Network* network) override
        {
			std::array<char, 32> bytes;
			copy_to(bytes.data() + 0, timestamp);
			copy_to(bytes.data() + 8, p->delay);
			copy_to(bytes.data() + 12, p->weight);
			copy_to(bytes.data() + 16, p->postNeuron->getPotential());
			copy_to(bytes.data() + 20, p->preNeuron ? p->preNeuron->getNeuronID() : -1);
			copy_to(bytes.data() + 22, p->postNeuron->getNeuronID());
			copy_to(bytes.data() + 24, p->postNeuron->getLayerID());
			copy_to(bytes.data() + 26, p->postNeuron->getRFID());
			copy_to(bytes.data() + 28, p->postNeuron->getX());
			copy_to(bytes.data() + 30, p->postNeuron->getY());
			saveFile.write(bytes.data(), bytes.size());
        }
		
		void neuronFired(double timestamp, projection* p, Network* network) override
        {
			std::array<char, 32> bytes;
			copy_to(bytes.data() + 0, timestamp);
			copy_to(bytes.data() + 8, p->delay);
			copy_to(bytes.data() + 12, p->weight);
			copy_to(bytes.data() + 16, p->postNeuron->getPotential());
			copy_to(bytes.data() + 20, p->preNeuron ? p->preNeuron->getNeuronID() : -1);
			copy_to(bytes.data() + 22, p->postNeuron->getNeuronID());
			copy_to(bytes.data() + 24, p->postNeuron->getLayerID());
			copy_to(bytes.data() + 26, p->postNeuron->getRFID());
			copy_to(bytes.data() + 28, p->postNeuron->getX());
			copy_to(bytes.data() + 30, p->postNeuron->getY());
			saveFile.write(bytes.data(), bytes.size());
        }
		
		template <typename T>
		static void copy_to(char* target, T t)
		{
		    *reinterpret_cast<T*>(target) = t;
		}
		
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
    };
}
