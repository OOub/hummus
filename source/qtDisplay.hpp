/*
 * qtDisplay.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Add-on to the Network class, used to display a GUI of the spiking neural network output. Depends on Qt5
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
#include <QQmlContext>

#include "network.hpp"
#include "inputViewer.hpp"
#include "outputViewer.hpp"
#include "potentialViewer.hpp"

namespace adonis_c
{
    class QtDisplay : public MainThreadNetworkDelegate
    {
    public:

    	// ----- CONSTRUCTOR -----
        QtDisplay()
        {
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

    	// ----- PUBLIC DISPLAY METHODS -----
		void incomingSpike(double timestamp, projection* p, Network* network) override
		{
			potentialviewer->handleData(timestamp, p, network);
		}

        void neuronFired(double timestamp, projection* p, Network* network) override
        {
			inputviewer->handleData(timestamp, p, network);
			outputviewer->handleData(timestamp, p, network);
			potentialviewer->handleData(timestamp, p, network);
		}

		void timestep(double timestamp, Network* network, Neuron* postNeuron) override
        {
			inputviewer->handleTimestep(timestamp);
			outputviewer->handleTimestep(timestamp);
			potentialviewer->handleTimestep(timestamp, network, postNeuron);
		}

		void begin(int numberOfLayers, std::vector<int> sublayerInLayers, std::vector<int> neuronsInLayers) override
		{
			int neuronNumber = std::accumulate(neuronsInLayers.begin(), neuronsInLayers.end(), 0);
			engine->rootContext()->setContextProperty("numberOfNeurons", neuronNumber);
			engine->rootContext()->setContextProperty("inputSublayer", sublayerInLayers[0]);
			engine->rootContext()->setContextProperty("layers", numberOfLayers);
        	outputviewer->setYLookup(neuronsInLayers);
			outputviewer->setSublayer(sublayerInLayers, engine);
			app->exec();
		}
		
		// ----- SETTERS -----
		void useHardwareAcceleration(bool accelerate)
        {
            inputviewer->useHardwareAcceleration(accelerate);
            outputviewer->useHardwareAcceleration(accelerate);
            potentialviewer->useHardwareAcceleration(accelerate);
        }

		void trackLayer(int layerToTrack)
		{
			outputviewer->changeLayer(layerToTrack);
		}

		void trackSublayer(int sublayerToTrack)
		{
			outputviewer->changeSublayer(sublayerToTrack);
		}
		
        void trackNeuron(int neuronToTrack)
        {
            potentialviewer->trackNeuron(neuronToTrack);
        }

		void setTimeWindow(double newWindow)
        {
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
    };
}
