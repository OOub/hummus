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

//  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return 0;
}
