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
#include <sstream>

#include "../../core.hpp"
#include "../../dependencies/puffin.hpp"
#include "../../dependencies/json.hpp"

#include <chrono> // @DEV (remove once done)

namespace hummus {
    class PuffinDisplay : public MainThreadAddon {

    public:

        // ----- CONSTRUCTOR -----
        PuffinDisplay() = default;

        // ----- PUBLIC DISPLAY METHODS -----
       void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {

           std::this_thread::sleep_for(std::chrono::seconds(1)); // @DEV (remove once done)

           std::stringstream stream;
           stream << "{\"type\":\"incomingSpike\",\"timestamp\":" << timestamp << ",\"pre\":" << s->getPresynapticNeuronID() << ",\"post\":" << s->getPostsynapticNeuronID() << ", \"postPotential\":" << postsynapticNeuron->getPotential() << "}";
           server->broadcast(puffin::string_to_message(stream.str()));
       }

       void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {

           std::this_thread::sleep_for(std::chrono::seconds(1)); // @DEV (remove once done)

           std::stringstream stream;
           stream << "{\"type\":\"neuronFired\",\"timestamp\":" << timestamp << ",\"pre\":" << s->getPresynapticNeuronID() << ",\"post\":" << s->getPostsynapticNeuronID() << ", \"postPotential\":" << postsynapticNeuron->getPotential() << "}";
           server->broadcast(puffin::string_to_message(stream.str()));
       }

        void timestep(double timestamp, Neuron* postsynapticNeuron, Network* network) override {
            std::string message_passed = std::to_string(timestamp) + std::to_string(postsynapticNeuron->getPotential());
            std::stringstream stream;
            stream << "{\"type\":\"timestep\",\"timestamp\":" << timestamp << ", \"postPotential\":" << postsynapticNeuron->getPotential() << "}";
            server->broadcast(puffin::string_to_message(stream.str()));
        }

       void statusUpdate(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
           // server->broadcast(puffin::string_to_message("statusUpdate"));
       }

        // Method to start the server
        void begin(Network* network, std::mutex* sync) override {
          server = puffin::make_server(
              8080,
              [network](std::size_t id, const std::string& url) {
                  std::stringstream stream;
                  stream << "{\"type\":\"state\"}";
                  return puffin::string_to_message(stream.str());
              },
              [this](std::size_t id, const puffin::message& message) {},
              [](std::size_t id) {});

            sync->unlock();
        }

    protected:

        // ----- IMPLEMENTATION VARIABLES -----
        std::unique_ptr<puffin::server> server;
    };
}
