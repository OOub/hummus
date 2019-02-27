/*
 * outputViewer.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 02/03/2018
 *
 * Information: The OutputViewer class is used by the Display class to show the output neurons. Depends on Qt5
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <atomic>

// QT5 and QT Charts Dependency
#include <QtCore/QObject>
#include <QtCore/QtMath>

#include <QtQuick/QQuickView>
#include <QtQuick/QQuickItem>
#include <QtQuick/QtQuick>
#include <QtCharts/QAbstractSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QXYSeries>
#include <QtCharts/QChart>
#include <QtWidgets/QSpinBox>

#include "../core.hpp"

namespace hummus {
    class OutputViewer : public QObject {
        
    Q_OBJECT
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        OutputViewer(QObject *parent = 0) :
                QObject(parent),
                timeWindow(100),
                openGL(false),
                isClosed(false),
                input(0),
                minY(0),
                maxY(1),
                layerTracker(1),
                sublayerTracker(0) {
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~OutputViewer(){}
		
    	// ----- PUBLIC OUTPUTVIEWER METHODS -----		
		void handleData(double timestamp, axon* a, Network* network) {
            input = timestamp;
			if (a->postNeuron->getLayerID() == layerTracker) {
				if (a->postNeuron->getSublayerID() == sublayerTracker) {
					while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
					if (!isClosed) {
						points.append(QPointF(timestamp, a->postNeuron->getNeuronID()));
						maxY = std::max(static_cast<float>(maxY), static_cast<float>(a->postNeuron->getNeuronID()));
					} else {
						points.clear();
					}
					atomicGuard.clear(std::memory_order_release);
				}
			}
        }
		
		void handleTimestep(double timestamp) {
			input = timestamp;
        }
		
		// ----- SETTERS -----
		void setEngine(QQmlApplicationEngine* _engine) {
			engine = _engine;
		}
		
		void setTimeWindow(float newWindow) {
            timeWindow = newWindow;
        }
		
		void useHardwareAcceleration(bool accelerate) {
            openGL = accelerate;
        }
		
		void setYLookup(std::vector<std::vector<int>> newLookup, std::vector<int> _neuronsInLayers) {
		    yLookupTable = newLookup;
		    neuronsInLayers = _neuronsInLayers;
		}
		
    Q_SIGNALS:
    public slots:
		
    	// ----- QT-RELATED METHODS -----
		void changeLayer(int newLayer) {
			layerTracker = newLayer;
			sublayerTracker = 0;
			engine->rootContext()->setContextProperty("sublayers", static_cast<int>(yLookupTable[layerTracker].size()-1));
			int previousLayerNeurons = std::accumulate(neuronsInLayers.begin(), neuronsInLayers.begin()+layerTracker, 0);
			
			minY = previousLayerNeurons;
			maxY = minY+1;
		}
		
		void changeSublayer(int newSublayer) {
			sublayerTracker = newSublayer;
			int previousLayerNeurons = std::accumulate(neuronsInLayers.begin(), neuronsInLayers.begin()+layerTracker, 0);
			int previousSublayerNeurons = std::accumulate(yLookupTable[layerTracker].begin(), yLookupTable[layerTracker].begin()+sublayerTracker, 0);
			minY = previousLayerNeurons+previousSublayerNeurons;
			maxY = minY+1;
		}
		
        void disable() {
            while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
            isClosed = true;
            atomicGuard.clear(std::memory_order_release);
        }
        
        void update(QtCharts::QValueAxis *axisX, QtCharts::QValueAxis *axisY, QtCharts::QAbstractSeries *series) {
            if (!isClosed) {
                if (series) {
                    while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
                    if (openGL) {
                        series->setUseOpenGL(true);
                    }
                    axisX->setRange(input - timeWindow, input+1);
                    if (!points.isEmpty()) {
                        auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - timeWindow, [](double timestamp, const QPointF& point) {
                            return timestamp < point.x();
                        });
                        points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
            
                        static_cast<QtCharts::QXYSeries *>(series)->replace(points);
                        axisY->setRange(minY,maxY);
                    }
                }
                atomicGuard.clear(std::memory_order_release);
            }
        }
		
    protected:
		
    	// ----- IMPLEMENTATION VARIABLES -----
        bool                          openGL;
        bool                          isClosed;
        double                        timeWindow;
        QVector<QPointF>              points;
        float                         input;
        int                           minY;
        int                           maxY;
        std::atomic_flag              atomicGuard;
        int                           layerTracker;
        int                           sublayerTracker;
        std::vector<std::vector<int>> yLookupTable;
        std::vector<int>              neuronsInLayers;
        QQmlApplicationEngine*        engine;
    };
}
