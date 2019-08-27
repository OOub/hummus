/*
 * dynamicsViewer.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The DynamicsViewer class is used by the Display class to show a specified neuron's potential and current. Depends on Qt5
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

namespace hummus {
    
    class DynamicsViewer : public QObject {
        
    Q_OBJECT
    public:
		
        // ----- CONSTRUCTOR AND DESTRUCTOR
        DynamicsViewer(QObject *parent = 0) :
                QObject(parent),
                timeWindow(100),
                openGL(true),
                isClosed(false),
                maxX(0),
                minY(20),
                maxY(-70),
                min_y_right(0),
                max_y_right(1),
                current_plot(false),
                neuronTracker(-1) {
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~DynamicsViewer(){}
		
    	// ----- PUBLIC DYNAMICSVIEWER METHODS -----
		void handleData(double timestamp, int postsynapticNeuronID, float _potential, float _current, float _threshold) {
			if (postsynapticNeuronID == neuronTracker) {
                while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
				if (!isClosed) {
                    // saving data points to plot
                    if (current_plot) {
                        currentPoints.append(QPointF(timestamp,_current));
                    }
					points.append(QPointF(timestamp, _potential));
					thresPoints.append(QPointF(timestamp, _threshold));
                    // membrane potential axis
					minY = std::min(minY, static_cast<float>(_potential));
					maxY = std::max(maxY, static_cast<float>(_potential));
                    // injected current axis
                    min_y_right = std::min(min_y_right, static_cast<float>(_current));
                    max_y_right = std::max(max_y_right, static_cast<float>(_current));
				} else {
					points.clear();
                    thresPoints.clear();
                    currentPoints.clear();
				}
				atomicGuard.clear(std::memory_order_release);
			}
            
            // time axis
            maxX = timestamp;
		}
		
		// ----- SETTERS -----
		void setTimeWindow(double newWindow) {
            timeWindow = newWindow;
        }
		
		void useHardwareAcceleration(bool accelerate) {
            openGL = accelerate;
        }
		
        void trackNeuron(int neuronToTrack) {
            neuronTracker = neuronToTrack;
        }
		
        void plotCurrents(bool _current_plot) {
            current_plot = _current_plot;
        }
        
        void reset() {
            points.clear();
            thresPoints.clear();
            currentPoints.clear();
        }
        
    Q_SIGNALS:
    public slots:
		
        // ----- QT-RELATED METHODS -----
        void changeTrackedNeuron(int newNeuron) {
            if (neuronTracker != newNeuron) {
                neuronTracker = newNeuron;
                minY = -70;
                maxY = -50;
                min_y_right = 0;
                max_y_right = 1;
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
					
                    switch (seriesType) {
                        case 0:
                            axisX->setRange(maxX - timeWindow, maxX+1);
                            if (!points.isEmpty()) {
                                auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - timeWindow, [](double timestamp, const QPointF& point) {
                                    return timestamp < point.x();
                                });
                                points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
                    
                                static_cast<QtCharts::QXYSeries *>(series)->replace(points);
                                axisY->setRange(minY-1,maxY+1);
                            }
                            break;
                        case 1:
                            if (!thresPoints.isEmpty()) {
                                auto firstToKeep = std::upper_bound(thresPoints.begin(), thresPoints.end(), thresPoints.back().x() - timeWindow, [](double timestamp, const QPointF& thresPoints) {
                                    return timestamp < thresPoints.x();
                                });
                                thresPoints.remove(0, static_cast<int>(std::distance(thresPoints.begin(), firstToKeep)));
                    
                                static_cast<QtCharts::QXYSeries *>(series)->replace(thresPoints);
                            }
                            break;
                        case 2:
                            if (current_plot) {
                                if (!currentPoints.isEmpty()) {
                                    auto firstToKeep = std::upper_bound(currentPoints.begin(), currentPoints.end(), currentPoints.back().x() - timeWindow, [](double timestamp, const QPointF& currentPoints) {
                                        return timestamp < currentPoints.x();
                                    });
                                    currentPoints.remove(0, static_cast<int>(std::distance(currentPoints.begin(), firstToKeep)));
                                    
                                    static_cast<QtCharts::QXYSeries *>(series)->replace(currentPoints);
                                    axisY->setRange(min_y_right-1,max_y_right+1);
                                }
                                break;
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
        QVector<QPointF>      currentPoints;
        double                maxX;
        float                 minY;
        float                 maxY;
        float                 min_y_right;
        float                 max_y_right;
        std::atomic_flag      atomicGuard;
        int                   neuronTracker;
        bool                  current_plot;
    };
}
