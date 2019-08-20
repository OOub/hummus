/*
 * inputViewer.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The InputViewer class is used by the Display class to show the input neurons. Depends on Qt5
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

#include "../core.hpp"

Q_DECLARE_METATYPE(QtCharts::QAbstractSeries *)
Q_DECLARE_METATYPE(QtCharts::QValueAxis *)

namespace hummus {
    class InputViewer : public QObject {
        
    Q_OBJECT
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        InputViewer(QObject *parent = 0) :
                QObject(parent),
                timeWindow(100),
                openGL(true),
                isClosed(false),
                maxX(1),
                minY(0),
                maxY(1),
                sublayerTracker(0) {
            qRegisterMetaType<QtCharts::QAbstractSeries*>();
            qRegisterMetaType<QtCharts::QValueAxis*>();
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~InputViewer(){}
		
    	// ----- PUBLIC INPUTVIEWER METHODS -----
		void handleData(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) {
            maxX = timestamp;
            if (s && s->getPresynapticNeuronID() == -1) {
                if (postsynapticNeuron->getSublayerID() == sublayerTracker) {
    
                    while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
                    if (!isClosed) {
                        points.append(QPointF(timestamp, postsynapticNeuron->getNeuronID()));
                        maxY = std::max(static_cast<float>(maxY), static_cast<float>(postsynapticNeuron->getNeuronID()));
                    } else {
                        points.clear();
                    }
                    atomicGuard.clear(std::memory_order_release);
                }
            }
        }
		
		void handleTimestep(double timestamp) {
			maxX = timestamp;
        }
		
		// ----- SETTERS -----
        void setTimeWindow(double newWindow) {
            timeWindow = newWindow;
        }
		
        void setYLookup(std::vector<int> newLookup) {
		    yLookupTable = newLookup;
		}
		
		void useHardwareAcceleration(bool accelerate) {
            openGL = accelerate;
        }
        
        void reset() {
            points.clear();
        }
        
    Q_SIGNALS:
    public slots:
		
    	// ----- QT-RELATED METHODS -----
    	void changeSublayer(int newSublayer) {
			sublayerTracker = newSublayer;
			
			if (newSublayer > 0) {
				minY = std::accumulate(yLookupTable.begin(), yLookupTable.begin()+sublayerTracker, 0);
			} else {
				minY = 0;
			}
			maxY = minY+1;
		}
		
        void disable() {
            while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
            isClosed = true;
            atomicGuard.clear(std::memory_order_release);
        }
			
        void update(QtCharts::QValueAxis *axisX, QtCharts::QValueAxis *axisY, QtCharts::QAbstractSeries *series) {
            if (!isClosed) {
                if (series)
                {
                    while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
                    if (openGL) {
                        series->setUseOpenGL(true);
                    }
                    axisX->setRange(maxX - timeWindow, maxX+1);
                    if (!points.isEmpty()) {
                        auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - timeWindow, [](double timestamp, const QPointF& point) {
                            return timestamp < point.x();
                        });
                        points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
            
                        static_cast<QtCharts::QXYSeries *>(series)->replace(points);
                        axisY->setRange(minY,maxY);
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
        double                maxX;
        int                   minY;
        int                   maxY;
        std::atomic_flag      atomicGuard;
        int                   sublayerTracker;
        std::vector<int>      yLookupTable;
    };
}
