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
#include "../source/neurons/input.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"
#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/potentialLogger.hpp"
#include "../source/GUI/qtqtDisplay.hpp"

int main(int argc, char** argv) {
    if (argc < 9) {
        
        std::cout << "The application received " << argc << " arguments" << std::endl;
        throw std::runtime_error(std::to_string(argc).append("received. Expecting at least 9 arguments."));
        
    } else {

        // ----- APPLICATION PARAMETERS -----
        // path to JSON network file
        std::string networkPath = argv[1];
        std::cout << "JSON path: " << argv[1] << std::endl;
        
        // path to data file
        std::string dataPath = argv[2];
        std::cout << "Data path: " << argv[2] << std::endl;
        
        // gaussian noise on timestamps of the data mean of 0 standard deviation of 1.0
        bool timeJitter = std::atoi(argv[3]);
        std::cout << "Additive Gaussian noise: " << argv[3] << std::endl;
        
        // percentage of additive noise
        int additiveNoise = std::atoi(argv[4]);
        std::cout << "Percentage of additive noise: " << argv[4] << std::endl;
        
        // name of output spike file
        std::string spikeLogName = argv[5];
        std::cout << "Data Log: " << argv[5] << std::endl;
        
        // name of output potential file
        std::string potentialLogName = argv[6];
        std::cout << "Potential Log " << argv[6] << std::endl;
    
        // 0 to run without gui. 1 to run with gui
        bool gui = std::atoi(argv[7]);
        
        // 0 for event-based, > 0 for clock-based
        float timestep = std::atof(argv[8]);
        std::cout << "Time step(0 for event based): " << argv[8] << std::endl;
        
        // neuron IDs to log
        std::vector<std::string> arguments;
        std::vector<int> neuronIDs;
        if (argc > 9) {
            arguments.insert(arguments.end(), argv + 10, argv + argc);
            for (auto& arg: arguments) {
                neuronIDs.push_back(std::atoi(arg.c_str()));
            }
        }
        
        // ----- IMPORTING DATA -----
        hummus::DataParser parser;
        auto data = parser.readData(dataPath, timeJitter, additiveNoise);

        //  ----- INITIALISING THE NETWORK -----
        hummus::SpikeLogger spikeLog(spikeLogName);
        hummus::PotentialLogger potentialLog(potentialLogName);

        hummus::Network network({&spikeLog, &potentialLog});
        hummus::Builder builder(&network);
        
        hummus::QtDisplay qtDisplay;
        if (gui) {
            std::cout << "Starting GUI" << std::endl;
            network.setMainThreadAddOn(&qtDisplay);
            qtDisplay.useHardwareAcceleration(true);
            qtDisplay.setTimeWindow(10000);
        }

        //  ----- CREATING THE NETWORK -----
        std::cout << "importing network from JSON file..." << std::endl;
        builder.import(networkPath);

        // initialising the potentialLoggers
        network.turnOffLearning(0);
        
        if (argc >= 9) {
            std::cout << "logging the potential of the selected neurons" << std::endl;
            potentialLog.neuronSelection(neuronIDs);
        } else {
            std::cout << "logging the potential of all the reservoir" << std::endl;
            potentialLog.neuronSelection(network.getLayers()[1]);
        }
        
        //  ----- RUNNING THE NETWORK ASYNCHRONOUSLY-----
        network.run(&data, timestep);
    }
    //  ----- EXITING APPLICATION -----
    return 0;
}
