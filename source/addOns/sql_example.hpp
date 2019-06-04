/*
 * spikeLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on used to write the spiking neural network output into binary file. The logger is constrained to reduce file size
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>
#include <sqlite3.h>

#include "../core.hpp"

namespace hummus {
    class SqlSpikeLogger : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        SqlSpikeLogger(const std::string& filename) {
            sqlite3_config(SQLITE_CONFIG_SERIALIZED);
            if (sqlite3_open_v2(filename.c_str(), &spikeLog, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr)
            != SQLITE_OK) {
                throw std::runtime_error("error opening the spike log");
            }
            if (sqlite3_extended_result_codes(spikeLog, 1) != SQLITE_OK) {
                throw std::runtime_error("error enabling extended result codes");
            }
        }
        
        virtual ~SqlSpikeLogger() {}
    
		// ----- PUBLIC METHODS -----
        void onStart(Network* network) override {
        }
        
		void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
        }
        
		void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
        }
		
        /// check_sqlite_status throws an error if the given sqlite functions returns an error code.
        static void check_sqlite_status(int status, const std::string& message) {
            if (status != SQLITE_OK && status != SQLITE_DONE) {
                throw std::runtime_error(message + " failed with error " + std::to_string(status));
            }
        }
		
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        sqlite3* spikeLog;
    };
}
