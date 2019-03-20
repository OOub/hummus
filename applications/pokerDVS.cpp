/*
 * pokerDVS.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"

#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/input.hpp"
#include "../source/neurons/decisionMaking.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/classificationLogger.hpp"
#include "../source/synapticKernels/exponential.hpp"

int main(int argc, char** argv) {
    
    bool networkType = 1; // choose between feedforward, deep spiking neural network or myelin plasticity network
    
    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;
    hummus::SpikeLogger spikeLog("pokerSpikeLog.bin");
    hummus::Network network({&spikeLog}, &qtDisplay);
    
    auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
    auto step = network.makeSynapticKernel<hummus::Step>(5); // step synaptic kernel
    network.setVerbose(0);
    
    if (networkType == 1) {
        //  ----- DEEP SPIKING NEURAL NETWORK -----
        
        /// parameters
        bool burst = true;
        bool homeostasis = true;
        bool conv_wta = true;
        bool pool_wta = false;

        /// creating the layers
        network.add2dLayer<hummus::Input>(40, 40, 1, {}, nullptr); // input layer
        network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.8, 0.1), 100, 4, {&ti_stdp}, &step, homeostasis, 20, 10, conv_wta, burst); // first convolution
        network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &step, homeostasis, 20, 0, pool_wta, false); // first pooling
        network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.8, 0.1), 100, 8, {&ti_stdp}, &step, homeostasis, 100, 10, conv_wta, burst); // second convolution
        network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &step, homeostasis, 20, 0, pool_wta, false); // second pooling
        network.addLayer<hummus::LIF>(2, {&ti_stdp}, &step, homeostasis, 500, 10, conv_wta, burst, 20, 0, 40, 1, -60, -70, 100); // output layer with 2 neurons
        
        /// connecting the layers
        network.allToAll(network.getLayers()[4], network.getLayers()[5], hummus::Normal(0.6, 0.1));
        network.allToAll(network.getLayers()[5], network.getLayers()[6], hummus::Normal(1, 0));
        
        qtDisplay.trackLayer(5);
        
    } else if (networkType == 0){
        // ----- SIMPLE FEEDFORWARD -----
        
        /// parameters
        bool homeostasis = true;
        bool wta = true;
        bool burst = true;
        
        /// creating the layers
        network.add2dLayer<hummus::Input>(34, 34, 1, {}, nullptr); // input layer
        network.addLayer<hummus::LIF>(100, {&ti_stdp}, &step, homeostasis, 20, 10, wta, burst, 20, 0, 40, 1, -50, -70, 100); // hidden layer with STDP
        network.addLayer<hummus::LIF>(2, {&ti_stdp}, &step, homeostasis, 500, 10, wta, burst, 20, 0, 40, 1, -60, -70, 100); // output layer with 2 neurons
        
        /// connecting the layers
        network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(0.8, 0.1));
        network.allToAll(network.getLayers()[1], network.getLayers()[2], hummus::Normal(0.8, 0.1));
        
        qtDisplay.trackLayer(2);
    }
    
	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
    auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");

	//  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(10000);
    
    std::cout << "output neuron IDs " << network.getNeurons().back()->getNeuronID() - 1 << " " << network.getNeurons().back()->getNeuronID() << std::endl;
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
    
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0, &testData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
