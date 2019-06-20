/*
 * tools.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 20/06/2019
 *
 * Information: Contains a bunch of functions used for speed optimisation
 */

#pragma once

namespace hummus {
    inline double fast_exp(double x) {
        x = 1.0 + x / 256.0;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        return x;
    }
}
