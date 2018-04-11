/*
 * sudoku.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 10/04/2018
 *
 * Information: spiking neural network trained to solve a 4x4 sudoku.
 *
 * Connectivity scheme:
 * 1- each domain is connected to the same domain on other layers for lateral inhibition
 * 2- horizontal inhibition within each layer
 * 3- vertical lateral inhibition within each layer
 * 4- lateral inhibition within each subgrid of each layer
 *
 *	  ----------- 			 -----------
 *	 |2 |  |  |1 |			|2 |4 |3 |1 |
 *	 |  |3 |  |  |			|1 |3 |4 |2 |
 *	 |  |  |1 |  |			|4 |2 |1 |3 |
 *	 |3 |  |  |4 |			|3 |1 |2 |4 |
 *	  -----------			 -----------
 *		SUDOKU                SOLUTION
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/display.hpp"
#include "../source/logger.hpp"

int main(int argc, char** argv)
{
	baal::DataParser dataParser;

//  ----- INITIALISING THE NETWORK -----
	baal::Network network;

//  ----- NETWORK PARAMETERS -----
	float runtime = 100;
	float timestep = 0.1;
	int   sudokuWidth = 4;
	int   neuronsPerDomain = 4;
    int   numberOfLayers = 5;
    
//  ----- CREATING THE LAYERS -----
    for (auto i=0; i<numberOfLayers; i++)
    {
        int x = 0;
        int y = 0;
        
        for (auto j=0; j<std::pow(sudokuWidth,2); j++)
        {
            if (j % sudokuWidth == 0 && j != 0)
		    {
			    x++;
			    y = 0;
		    } 
            network.addNeurons(neuronsPerDomain,i+1, x, y);
            y++;
        }    
    }
    
//  ----- CONNECTING THE LAYERS -----
    for (auto i=0; i<std::pow(sudokuWidth,2); i++)
    {
        // 1->2 and 2->1
        network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[i+std::pow(sudokuWidth,2)], true, -1, false, 5);
        network.allToallConnectivity(&network.getNeuronPopulations()[i+std::pow(sudokuWidth,2)], &network.getNeuronPopulations()[i], true, -1, false, 5);
		
		// 1->3 and 3->1
		network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*2)], true, -1, false, 5);
		network.allToallConnectivity(&network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*2)], &network.getNeuronPopulations()[i], true, -1, false, 5);
		
		// 1->4 and 4->1
		network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*3)], true, -1, false, 5);
		network.allToallConnectivity(&network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*3)], &network.getNeuronPopulations()[i], true, -1, false, 5);
		
		// 2->3 and 3->2
		network.allToallConnectivity(&network.getNeuronPopulations()[i+std::pow(sudokuWidth,2)], &network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*2)], true, -1, false, 5);
		network.allToallConnectivity(&network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*2)], &network.getNeuronPopulations()[i+std::pow(sudokuWidth,2)], true, -1, false, 5);
		
		// 2->4 and 4->2
		network.allToallConnectivity(&network.getNeuronPopulations()[i+std::pow(sudokuWidth,2)], &network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*3)], true, -1, false, 5);
		network.allToallConnectivity(&network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*3)], &network.getNeuronPopulations()[i+std::pow(sudokuWidth,2)], true, -1, false, 5);
		
		// 3->4 and 4->3
		network.allToallConnectivity(&network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*2)], &network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*3)], true, -1, false, 5);
		network.allToallConnectivity(&network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*3)], &network.getNeuronPopulations()[i+(std::pow(sudokuWidth,2)*2)], true, -1, false, 5);
    }
    
    // loop for horizontal / vertical / grid on each layer
    for (auto i=0; i<(numberOfLayers-1)*std::pow(sudokuWidth,2); i+=std::pow(sudokuWidth,2))
    {
		// horizontal connections
		int x = 0;
		for (auto j=i; j<i+std::pow(sudokuWidth,2); j++)
		{   
			if (j % sudokuWidth == 0 && j != i)
			{
				x++;
			}
			 
			for (auto k=i; k<i+std::pow(sudokuWidth,2); k++)
			{
				if (network.getNeuronPopulations()[k][0].getX() == x && j != k)
				{
					network.allToallConnectivity(&network.getNeuronPopulations()[j], &network.getNeuronPopulations()[k], true, -1, false, 5);
				}
			}
		}
    }


//  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return 0;
}
