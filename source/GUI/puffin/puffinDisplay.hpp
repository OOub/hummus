/*
 * puffinDisplay.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 19/03/2019
 *
 * Information: Add-on used to display a GUI of the spiking neural network output using the puffin websocket server (no dependencies). Adds the ability to view the network using a browser independently from the computer actually running the network - useful for running on clusters.
 */

#include <vector>
#include <string>

#include "../../core.hpp"
#include "../../dependencies/puffin.hpp"
#include "../../dependencies/json.hpp"

namespace hummus {
    class PuffinDisplay : public MainThreadAddon {

    public:

        // ----- CONSTRUCTOR -----
        PuffinDisplay() = default;

        // ----- PUBLIC DISPLAY METHODS -----
        void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // server->broadcast(puffin::string_to_message("s,0,0,2,3.2"));
        }

        void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // server->broadcast(puffin::string_to_message("neuronFired"));
        }

        void timestep(double timestamp, Neuron* postsynapticNeuron, Network* network) override {
            // server->broadcast(puffin::string_to_message("timestep"));
        }

        void statusUpdate(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // server->broadcast(puffin::string_to_message("statusUpdate"));
        }

        // Method to start the server
        void begin(Network* network, std::mutex* sync) override {
          server = puffin::make_server(
              8080,
              [network](std::size_t id, const std::string& url) {
                  std::cout << id << " connected" << std::endl;
                  return puffin::string_to_message("welcome");
              },
              [this](std::size_t id, const puffin::message& message) {},
              [](std::size_t id) { std::cout << id << " disconnected" << std::endl; });

            sync->unlock();
        }

    protected:

        // ----- IMPLEMENTATION VARIABLES -----
        std::unique_ptr<puffin::server> server;
    };
}
