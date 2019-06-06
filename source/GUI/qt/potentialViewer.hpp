/*
 * potentialViewer.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The PotentialViewer class is used by the Display class to show a specified neuron's potential. Depends on Qt5
 */

#pragma once

#include <algorithm>
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

#include "../../core.hpp"

namespace hummus {
    class PotentialViewer : public QObject {
        
    Q_OBJECT
    public:
		
        // ----- CONSTRUCTOR AND DESTRUCTOR
        PotentialViewer(QObject *parent = 0) :
                QObject(parent),
                timeWindow(100),
                openGL(true),
                isClosed(false),
                maxX(0),
                minY(20),
                maxY(-70),
                potential(0),
                threshold(-50),
                neuronTracker(-1) {
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~PotentialViewer(){}
		
    	// ----- PUBLIC POTENTIALVIEWER METHODS -----
		void handleData(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) {
			if (postsynapticNeuron->getNeuronID() == neuronTracker) {
                while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
				if (!isClosed) {
					potential = postsynapticNeuron->getPotential();
					threshold = postsynapticNeuron->getThreshold();
					points.append(QPointF(timestamp, potential));
					thresPoints.append(QPointF(timestamp, threshold));
					minY = std::min(minY, static_cast<float>(potential));
					maxY = std::max(maxY, static_cast<float>(potential));
                    maxX = timestamp;
				} else {
					points.clear();
				}
				atomicGuard.clear(std::memory_order_release);
			}
		}
		
		void handleTimestep(double timestamp, Neuron* postsynapticNeuron, Network* network) {
			if (postsynapticNeuron->getNeuronID() == neuronTracker) {
				while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
				if (!isClosed) {
					potential = postsynapticNeuron->getPotential();
					points.append(QPointF(timestamp, potential));
					thresPoints.append(QPointF(timestamp, threshold));
					minY = std::min(minY, static_cast<float>(potential));
					maxY = std::max(maxY, static_cast<float>(potential));
				} else {
					points.clear();
					thresPoints.clear();
				}
				atomicGuard.clear(std::memory_order_release);
			}
			maxX = timestamp;
		}
		
		// ----- SETTERS -----
		void setTimeWindow(double newWindow) {
            timeWindow = newWindow;
        }
		
		void useHardwareAcceleration(bool accelerate) {
            openGL = accelerate;
        }
		
        void trackNeuron(size_t neuronToTrack) {
            neuronTracker = neuronToTrack;
        }
		
    Q_SIGNALS:
    public slots:
		
        // ----- QT-RELATED METHODS -----
        void changeTrackedNeuron(size_t newNeuron) {
            if (neuronTracker != newNeuron) {
                neuronTracker = newNeuron;
                minY = 20;
                maxY = -70;
            }
        }
    
        void disable() {
            while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
            isClosed = true;
            atomicGuard.clear(std::memory_order_release);
        }
    
        void update(QtCharts::QValueAxis *axisX, QtCharts::QValueAxis *axisY, QtCharts::QAbstractSeries *series,  int seriesType) {
            if (!isClosed) {
                if (series) {
                    while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
                    if (openGL) {
                        series->setUseOpenGL(true);
                    }
					
                    if (seriesType == 0) {
						axisX->setRange(maxX - timeWindow, maxX+1);
						if (!points.isEmpty()) {
							auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - timeWindow, [](double timestamp, const QPointF& point) {
								return timestamp < point.x();
							});
							points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
				
							static_cast<QtCharts::QXYSeries *>(series)->replace(points);
							axisY->setRange(minY-1,maxY+1);
						}
                    }
                    else if (seriesType == 1) {
						if (!points.isEmpty()) {
							auto firstToKeep = std::upper_bound(thresPoints.begin(), thresPoints.end(), thresPoints.back().x() - timeWindow, [](double timestamp, const QPointF& thresPoints) {
								return timestamp < thresPoints.x();
							});
							thresPoints.remove(0, static_cast<int>(std::distance(thresPoints.begin(), firstToKeep)));
				
							static_cast<QtCharts::QXYSeries *>(series)->replace(thresPoints);
						}
					}
					
                    atomicGuard.clear(std::memory_order_release);
                }
            }
        }
        
    protected:
		
        // ----- IMPLEMENTATION VARIABLES -----
        bool                  isClosed;
        bool                  openGL;
        double                timeWindow;
        QVector<QPointF>      points;
        QVector<QPointF>      thresPoints;
        double                maxX;
        float                 minY;
        float                 maxY;
        std::atomic_flag      atomicGuard;
        float                 potential;
        float                 threshold;
        size_t                neuronTracker;
    };
}
