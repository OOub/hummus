/*
 * outputViewer.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 18/10/2019
 *
 * Information: The OutputViewer class is used by the Display class to show the output neurons. Depends on Qt5
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <atomic>

// QT5 and QT Charts Dependency
#include <QtCore/QObject>

#include <QtQuick/QQuickView>
#include <QtQuick/QQuickItem>
#include <QtQuick/QtQuick>
#include <QtCharts/QAbstractSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QXYSeries>
#include <QtCharts/QChart>
#include <QtWidgets/QSpinBox>

namespace hummus {
    
    class OutputViewer : public QObject {
        
    Q_OBJECT
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        OutputViewer(QObject *parent = 0) :
                QObject(parent),
                openGL(true),
                isClosed(false),
                timeWindow(100),
                input(0),
                minY(0),
                maxY(1),
                layerTracker(1),
                sublayerTracker(0),
                layer_changed(false) {
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~OutputViewer(){}
		
    	// ----- PUBLIC OUTPUTVIEWER METHODS -----		
		void handle_data(double timestamp, int postsynapticNeuronID, int postsynapticLayerID, int postsynapticSublayerID) {
            input = timestamp;
			if (postsynapticLayerID == layerTracker) {
				if (postsynapticSublayerID == sublayerTracker) {
					while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
					if (!isClosed) {
						points.append(QPointF(timestamp, postsynapticNeuronID));
						maxY = std::max(maxY, postsynapticNeuronID);
					} else {
						points.clear();
					}
					atomicGuard.clear(std::memory_order_release);
				}
			}
        }
		
		void handle_update(double timestamp) {
			input = timestamp;
        }
		
		// ----- SETTERS -----
        bool get_layer_changed() const {
            return layer_changed;
        }
        
        void set_layer_changed(bool new_layer) {
            layer_changed = new_layer;
        }
		
        int get_layer_tracker() const {
            return layerTracker;
        }
        
		void set_time_window(float newWindow) {
            timeWindow = newWindow;
        }
		
		void hardware_acceleration(bool accelerate) {
            openGL = accelerate;
        }
		
        std::vector<std::vector<int>> get_y_lookup() const {
            return yLookupTable;
        }
        
		void set_y_lookup(std::vector<std::vector<int>> newLookup, std::vector<int> _neuronsInLayers) {
		    yLookupTable = newLookup;
		    neuronsInLayers = _neuronsInLayers;
		}
		
        void reset() {
            points.clear();
        }
        
    Q_SIGNALS:
    public Q_SLOTS:
		
    	// ----- QT-RELATED METHODS -----
		void change_layer(int newLayer) {
			layerTracker = newLayer;
			sublayerTracker = 0;
            layer_changed = true;
			int previousLayerNeurons = std::accumulate(neuronsInLayers.begin(), neuronsInLayers.begin()+layerTracker, 0);
			minY = previousLayerNeurons;
			maxY = minY+1;
		}
		
		void change_sublayer(int newSublayer) {
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
        float                         timeWindow;
        QVector<QPointF>              points;
        float                         input;
        int                           minY;
        int                           maxY;
        std::atomic_flag              atomicGuard;
        int                           layerTracker;
        int                           sublayerTracker;
        std::vector<std::vector<int>> yLookupTable;
        std::vector<int>              neuronsInLayers;
        bool                          layer_changed;
    };
}
