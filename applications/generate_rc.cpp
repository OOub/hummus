/*
 * generate_rc.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 07/03/2019
 *
 * Information: Application to generate a reservoir network and save it in a JSON file
 */

#include <iostream>
#include <string>
#include <iomanip>

#include "../source/core.hpp"
#include "../source/dataParser.hpp"
#include "../source/synapticKernels/step.hpp"
#include "../source/randomDistributions/cauchy.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/neurons/input.hpp"
#include "../source/neurons/LIF.hpp"

int main(int argc, char** argv) {
    if (argc < 19) {
        std::cout << "The application received " << argc << " arguments" << std::endl;
        throw std::runtime_error(std::to_string(argc).append(" received. Expecting 19 arguments"));
    }

    // ----- RESERVOIR PARAMETERS -----
    size_t clean_cout = std::string("Threshold Adaptation to firing rate: ").size()+10;
    
    // pixel grid width
    int gridWidth = std::atoi(argv[1]);
    std::cout << "Pixel width: " << std::setw(static_cast<int>(clean_cout-std::string("Pixel width: ").size())) << argv[1] << std::endl;
    
    // pixel grid height
    int gridHeight = std::atoi(argv[2]);
    std::cout << "Pixel height: " << std::setw(static_cast<int>(clean_cout-std::string("Pixel height: ").size())) << argv[2] << std::endl;
	
    // gaussian mean for reservoir weights
    float inputWeightMean = std::atof(argv[3]);
    std::cout << "Input Weight Mean: " << std::setw(static_cast<int>(clean_cout-std::string("Input Weight Mean: ").size())) << argv[3] << std::endl;
    
    // gaussian standard deviation for reservoir weights
    float inputWeightStdDev = std::atof(argv[4]);
    std::cout << "Input Weight Std: " << std::setw(static_cast<int>(clean_cout-std::string("Input Weight Std: ").size())) << argv[4] << std::endl;

    // number of neurons inside the reservoir
    int numberOfNeurons = std::atoi(argv[5]);
    std::cout << "Reservoir Neurons: " << std::setw(static_cast<int>(clean_cout-std::string("Reservoir Neurons: ").size())) << argv[5] << std::endl;
    
    // gaussian mean for weights
    float weightMean = std::atof(argv[6]);
    std::cout << "Weight mean: " << std::setw(static_cast<int>(clean_cout-std::string("Weight mean: ").size())) << argv[6] << std::endl;
    
    // gaussian standard deviation for weights
    float weightStdDev = std::atof(argv[7]);
    std::cout << "Weight std: " << std::setw(static_cast<int>(clean_cout-std::string("Weight std: ").size())) << argv[7] << std::endl;

    // percentage likelihood of feedforward connections
    int feedforwardProbability = std::atoi(argv[8]);
    std::cout << "Forward connection probability: " << std::setw(static_cast<int>(clean_cout-std::string("Forward connection probability: ").size())) << argv[8] << std::endl;
    
    // percentage likelihood of feedback connections
    int feedbackProbability = std::atoi(argv[9]);
    std::cout << "Back connection probability: " << std::setw(static_cast<int>(clean_cout-std::string("Back connection probability: ").size())) << argv[9] << std::endl;
    
    // percentage likelihood of self-excitation
    int selfExcitationProbability = std::atoi(argv[10]);
    std::cout << "Stay connection probability: " << std::setw(static_cast<int>(clean_cout-std::string("Stay connection probability: ").size())) << argv[10] << std::endl;
	
    // current step function reset value (integration time)
    float resetCurrent = std::atof(argv[11]);
    std::cout << "Reset current duration " << std::setw(static_cast<int>(clean_cout-std::string("Reset current duration ").size())) << argv[11] << std::endl;
	
    // time constant for membrane potential (decay)
    float decayPotential = std::atof(argv[12]);
    std::cout << "Potential decay time: " << std::setw(static_cast<int>(clean_cout-std::string("Potential decay time: ").size())) << argv[12] << std::endl;
    
    // neuron inactive for specified time after each spike
    int refractoryPeriod = std::atoi(argv[13]);
    std::cout << "Refractory Period: " << std::setw(static_cast<int>(clean_cout-std::string("Refractory Period: ").size())) << argv[13] << std::endl;
    
    // winner-takes-all algorithm
    bool wta = std::atoi(argv[14]);
    std::cout << "Winner takes all: " << std::setw(static_cast<int>(clean_cout-std::string("Winner takes all: ").size())) << argv[14] << std::endl;
    
    // threshold adaptation to firing rate
    bool homeostasis = std::atoi(argv[15]);
    std::cout << "Threshold Adaptation to firing rate: " << std::setw(10) << argv[15] << std::endl;
    
    // use weight matrix to build instead of a reservoir with probabilities
    bool useMatrix = std::atoi(argv[16]);
    std::cout << "use weight matrix: " << std::setw(static_cast<int>(clean_cout-std::string("use weight matrix: ").size())) << argv[16] << std::endl;
    
    // input weight matrix filename
    std::string inputWeightFile = argv[17];
    std::cout << "input weights file: " << std::setw(static_cast<int>(clean_cout-std::string("input weights file: ").size())) << argv[17] << std::endl;
    
    // reservoir weight matrix filename
    std::string reservoirWeightFile = argv[18];
    std::cout << "reservoir weights file: " << std::setw(static_cast<int>(clean_cout-std::string("reservoir weights file: ").size())) << argv[18] << std::endl;
    
    //  ----- READING WEIGHT MATRIX FOMR .TXT FILE -----
    hummus::DataParser parser;
    auto inputWeightMatrix = parser.readWeightMatrix(inputWeightFile);
    auto reservoirWeightMatrix = parser.readWeightMatrix(reservoirWeightFile);
    
    //  ----- CREATING THE NETWORK -----
    
    std::cout << "\nbuilding network..." << std::endl;
    
    // network initialisation
    hummus::Network network;
    
    // initialise synaptic kernel
    auto& step = network.makeSynapticKernel<hummus::Step>(resetCurrent);
    
    // pixel grid layer
    auto pixel_grid = network.make2dLayer<hummus::Input>(gridWidth, gridHeight, 1, {});
    
    if (useMatrix) {
        // create reservoir layer
        auto reservoir = network.makeLayer<hummus::LIF>(numberOfNeurons, {}, &step, homeostasis, decayPotential, refractoryPeriod, wta);
        
        // connecting input according to weight matrix
        network.weightMatrix(pixel_grid, reservoir, inputWeightMatrix, hummus::Normal(0, 0, 0, 0));
        
        // connecting reservoir according to weight matrix
        network.weightMatrix(reservoir, reservoir, reservoirWeightMatrix, hummus::Normal(0, 0, 0, 0));
        
    } else {
        // reservoir layer
        auto reservoir = network.makeReservoir<hummus::LIF>(numberOfNeurons, hummus::Normal(weightMean, weightStdDev), feedforwardProbability, feedbackProbability, selfExcitationProbability, &step, homeostasis, decayPotential, refractoryPeriod, wta);
        
        // connect pixel grid to the reservoir in an all to all fashion
        network.allToAll(pixel_grid, reservoir, hummus::Normal(inputWeightMean, inputWeightStdDev, 0, 0));
    }
    
    std::cout << "\nsaving network into rcNetwork.json file..." << std::endl;
    
    network.save("rcNetwork");
    
    std::cout << "done!" << std::endl;
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
