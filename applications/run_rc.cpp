/*
 * rc.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 05/03/2019
 *
 * Information: Application that build a betwork from a previously saved JSON network file and loads data into the network;
 */

#include <iostream>
#include <string>

#include "../source/core.hpp"
#include "../source/builder.hpp"
#include "../source/dataParser.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"

int main(int argc, char** argv) {
    if (argc < 10) {
        
        std::cout << "The application received " << argc << " arguments" << std::endl;
        throw std::runtime_error(std::to_string(argc).append("received. Expecting at least 10 arguments."));
        
    } else {

        // ----- APPLICATION PARAMETERS -----
        // path to JSON network file
        std::string networkPath = argv[1];
		
        // verbose 0, 1 or 2
        int verbose = std::atoi(argv[9]);
        if (verbose!=0) std::cout<< "Verbosity level " << verbose << std::endl;
        if (verbose!=0) std::cout << "JSON path: " << argv[1] << std::endl;
        
        // path to data file
        std::string dataPath = argv[2];
        if (verbose!=0) std::cout << "Data path: " << argv[2] << std::endl;
        
        // gaussian noise on timestamps of the data mean of 0 standard deviation of 1.0
        bool timeJitter = std::atoi(argv[3]);
        if (verbose!=0) std::cout << "Additive Gaussian noise: " << argv[3] << std::endl;
        
        // percentage of additive noise
        int additiveNoise = std::atoi(argv[4]);
        if (verbose!=0) std::cout << "Percentage of additive noise: " << argv[4] << std::endl;
        
        // name of output spike file
        std::string spikeLogName = argv[5];
        if (verbose!=0) std::cout << "Data Log: " << argv[5] << std::endl;
        
        // name of output potential file
        std::string potentialLogName = argv[6];
        if (verbose!=0) std::cout << "Potential Log " << argv[6] << std::endl;
    
        // 0 to run without gui. 1 to run with gui
        bool gui = std::atoi(argv[7]);
        
        // 0 for event-based, > 0 for clock-based
        float timestep = std::atof(argv[8]);
        if (verbose!=0) std::cout << "Time step(0 for event based): " << argv[8] << std::endl;
		
		// ----- IMPORTING DATA -----
        hummus::DataParser parser;
        auto data = parser.readData(dataPath, timeJitter, additiveNoise);
        
        // neuron IDs to log
        std::vector<size_t> neuronIDs;
        if (argc > 10) {
            std::vector<std::string> fields;
			parser.split(fields, argv[10], " ,[]");
			
			for (auto f: fields) {
				neuronIDs.push_back(static_cast<size_t>(std::atoi(f.c_str())));
			}
        }
		
        //  ----- INITIALISING THE NETWORK -----
        hummus::Network network;
        network.makeAddon<hummus::SpikeLogger>(spikeLogName);
        auto& potentialLog = network.makeAddon<hummus::PotentialLogger>(potentialLogName);
        
        hummus::Builder builder(&network);
        
        network.verbosity(verbose);
        
        std::cout<<gui<<std::endl;
        hummus::QtDisplay qtDisplay;
        if (gui) {
            std::cout<<"I am here"<<std::endl;
            std::cout << "Starting GUI" << std::endl;
            network.setMainThreadAddon(&qtDisplay);
            qtDisplay.setTimeWindow(10000);
        }

        //  ----- CREATING THE NETWORK -----
        if (verbose!=0) std::cout << "importing network from JSON file..." << std::endl;
        builder.import(networkPath);

        // initialising the potentialLoggers
        network.turnOffLearning(0);
		
        if (argc > 10) {
            if (verbose!=0) std::cout << "logging the potential of the selected neurons" << std::endl;
            potentialLog.activate_for(neuronIDs);
        } else {
            if (verbose!=0) std::cout << "logging the potential of all the reservoir" << std::endl;
            potentialLog.activate_for(network.getLayers()[1].neurons);
        }
		
        //  ----- RUNNING THE NETWORK ASYNCHRONOUSLY-----
        network.run(&data, timestep);
    }
    //  ----- EXITING APPLICATION -----
    return 0;
}
