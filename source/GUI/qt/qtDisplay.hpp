/*
 * qtDisplay.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Add-on used to display a GUI of the spiking neural network output using Qt (Qt5 dependency)
 */

#pragma once

#include <numeric>
#include <vector>
#include <thread>
#include <chrono>
#include <string>

// QT5 Dependency
#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QtQuick/QQuickView>

#include "../../core.hpp"
#include "inputViewer.hpp"
#include "outputViewer.hpp"
#include "potentialViewer.hpp"

namespace hummus {
    class QtDisplay : public MainThreadAddon {
        
    public:

    	// ----- CONSTRUCTOR AND DESTRUCTOR-----
        QtDisplay() :
                neuronToTrack(-1),
                inputSublayerToTrack(0),
                outputLayerToTrack(1),
                outputSublayerToTrack(0) {
        	static int argc = 1;
			static char* argv[1] = {NULL};

			app.reset(new QApplication(argc, argv));

            qmlRegisterType<InputViewer>("InputViewer", 1, 0, "InputViewer");
            qmlRegisterType<OutputViewer>("OutputViewer", 1, 0, "OutputViewer");
            qmlRegisterType<PotentialViewer>("PotentialViewer", 1, 0, "PotentialViewer");
			
			engine = new QQmlApplicationEngine();
			
			engine->rootContext()->setContextProperty("layers", 1);
			engine->rootContext()->setContextProperty("inputSublayer", 1);
			engine->rootContext()->setContextProperty("sublayers", 1);
			engine->rootContext()->setContextProperty("numberOfNeurons", 1);
			
            engine->loadData(
				#include "gui.qml"
            );
            auto window = qobject_cast<QQuickWindow*>(engine->rootObjects().first());

			QSurfaceFormat format;
            format.setDepthBufferSize(24);
            format.setStencilBufferSize(8);
            format.setVersion(3, 3);
            format.setProfile(QSurfaceFormat::CompatibilityProfile);
            window->setFormat(format);
            window->show();

            inputviewer = window->findChild<InputViewer*>("inputViewer");
            outputviewer = window->findChild<OutputViewer*>("outputViewer");
            potentialviewer = window->findChild<PotentialViewer*>("potentialViewer");
        }
        
        virtual ~QtDisplay(){}
        
    	// ----- PUBLIC DISPLAY METHODS -----
		void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
			potentialviewer->handleData(timestamp, s, postsynapticNeuron, network);
		}

        void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
			inputviewer->handleData(timestamp, s, postsynapticNeuron, network);
			outputviewer->handleData(timestamp, s, postsynapticNeuron, network);
			potentialviewer->handleData(timestamp, s, postsynapticNeuron, network);
		}

		void timestep(double timestamp, Neuron* postsynapticNeuron, Network* network) override {
			inputviewer->handleTimestep(timestamp);
			outputviewer->handleTimestep(timestamp);
			potentialviewer->handleTimestep(timestamp, postsynapticNeuron, network);
		}

        void statusUpdate(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            potentialviewer->handleData(timestamp, s, postsynapticNeuron, network);
        }
        
		void begin(Network* network, std::mutex* sync) override {
            // finding the number of layers in the network
            int numberOfLayers = static_cast<int>(network->getLayers().size());

            // number of sublayers in each layer
            std::vector<int> sublayerInLayers;
            for (auto& l: network->getLayers()) {
                sublayerInLayers.emplace_back(l.sublayers.size());
            }

            // number of neurons in each layer
            std::vector<int> neuronsInLayers;
            for (auto& l: network->getLayers()) {
                neuronsInLayers.emplace_back(l.neurons.size());
            }

            // number of neurons in each sublayer
            std::vector<std::vector<int>> neuronsInSublayers(numberOfLayers);
            int idx = 0;
            for (auto& l: network->getLayers()) {
                for (auto& s: l.sublayers) {
                    neuronsInSublayers[idx].emplace_back(s.neurons.size());
                }
                idx += 1;
            }

            int neuronNumber = static_cast<int>(network->getNeurons().size());
            
            engine->rootContext()->setContextProperty("numberOfNeurons", neuronNumber);
            engine->rootContext()->setContextProperty("inputSublayer", sublayerInLayers[0]-1);
            engine->rootContext()->setContextProperty("layers", numberOfLayers-1);

            inputviewer->setYLookup(neuronsInSublayers[0]);
            outputviewer->setEngine(engine);
            outputviewer->setYLookup(neuronsInSublayers, neuronsInLayers);

            inputviewer->changeSublayer(inputSublayerToTrack);
            outputviewer->changeLayer(outputLayerToTrack);
            outputviewer->changeSublayer(outputSublayerToTrack);
            potentialviewer->trackNeuron(neuronToTrack);
			
            sync->unlock();

			app->exec();
		}
		
		// ----- SETTERS -----
		void useHardwareAcceleration(bool accelerate) {
            inputviewer->useHardwareAcceleration(accelerate);
            outputviewer->useHardwareAcceleration(accelerate);
            potentialviewer->useHardwareAcceleration(accelerate);
        }

		void trackLayer(int layerToTrack) {
			outputLayerToTrack = layerToTrack;
		}
		
		void trackInputSublayer(int sublayerToTrack) {
			inputSublayerToTrack = sublayerToTrack;
		}
		
		void trackOutputSublayer(int sublayerToTrack) {
			outputSublayerToTrack = sublayerToTrack;
		}
		
        void trackNeuron(size_t _neuronToTrack) {
        	neuronToTrack = _neuronToTrack;
        }

		void setTimeWindow(double newWindow) {
            inputviewer->setTimeWindow(newWindow);
            outputviewer->setTimeWindow(newWindow);
            potentialviewer->setTimeWindow(newWindow);
        }
		
    protected:

		// ----- IMPLEMENTATION VARIABLES -----
        std::unique_ptr<QApplication>          app;
        QQmlApplicationEngine*                 engine;
        InputViewer*                           inputviewer;
        OutputViewer*                          outputviewer;
        PotentialViewer*                       potentialviewer;
        size_t                                 neuronToTrack;
        int                                    inputSublayerToTrack;
        int                                    outputLayerToTrack;
        int                                    outputSublayerToTrack;
    };
}
