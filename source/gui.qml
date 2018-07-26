R""(

/*
 * gui.qml
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 16/01/2018
 *
 * Information: QML file that defines the GUI.
 */

import QtQuick 2.7
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.3
import QtQuick.Window 2.2
import QtCharts 2.2

import InputViewer 1.0
import OutputViewer 1.0
import PotentialViewer 1.0

ApplicationWindow
{
	id: mainWindow
	title: qsTr("Spiking Neural Network")
	height: 900
	width: 900
	minimumHeight: 650
	minimumWidth: 450
	property int refresh: 20

	property int a: 0
	property int b: 1

	ColumnLayout
	{
		id: mainGrid
		anchors.fill: parent

		Rectangle
		{
			id: inputRec
			color: '#FFFAEF'
			Layout.alignment: Qt.AlignCenter
			Layout.topMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-10
			radius: 10

			ChartView
			{
				id: inputChart
				title: "Input Neurons"
				anchors.fill: parent
				antialiasing: true
				backgroundColor: '#FFFAEF'
				legend.visible: false
				dropShadowEnabled: false

				ValueAxis
				{
					id:inputX
					tickCount: inputRec.width/75
				}

				ValueAxis
				{
					id:inputY
					tickCount: inputRec.height/50
				}

				ScatterSeries
				{
					id: input
					name: "Input Neurons"
					markerSize: 5
					markerShape: ScatterSeries.MarkerShapeCircle
					axisX: inputX
					axisY: inputY
					borderColor: 'transparent'
				}

				Timer
				{
					id: refreshTimer
					interval: refresh
					running: true
					repeat: true
					onTriggered:
					{
						inputViewer.update(inputX, inputY, inputChart.series(0));
					}
				}

				InputViewer
				{
					objectName: "inputViewer"
					id: inputViewer
				}
			}
		}

		Rectangle
		{
			id: outputRec
			color: '#FFFAEF'
			Layout.alignment: Qt.AlignCenter
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-10
			radius: 10

			ChartView
			{
				id: outputChart
				title: "Output Neurons"
				anchors.fill: parent
				antialiasing: true
				backgroundColor: '#FFFAEF'
				legend.visible: false
				dropShadowEnabled: false

				ValueAxis
				{
					id:outputX
					tickCount: outputRec.width/75
				}

				ValueAxis
				{
					id:outputY
					tickCount: outputRec.height/50
				}

				SpinBox
				{
					id: layerbox
					minimumValue: 1
					maximumValue: layers
					onEditingFinished:
					{
						outputViewer.changeLayer(value)
					}
				}

				ScatterSeries
				{
					id: output
					name: "Output Neurons"
					markerSize: 5
					markerShape: ScatterSeries.MarkerShapeCircle
					axisX: outputX
					axisY: outputY
					borderColor: 'transparent'
				}

				Timer
				{
					id: refreshTimer2
					interval: refresh
					running: true
					repeat: true
					onTriggered:
					{
						outputViewer.update(outputX, outputY, outputChart.series(0));
					}
				}

				OutputViewer
				{
					objectName: "outputViewer"
					id: outputViewer
				}

			}
		}

		Rectangle
		{
			id: potentialRec
			color: '#FFFAEF'
			Layout.alignment: Qt.AlignCenter
			Layout.bottomMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-10
			radius: 10

			ChartView
			{
				id: membraneChart
				title: "Membrane Potential"
				anchors.fill: parent
				antialiasing: true
				backgroundColor: '#FFFAEF'
				legend.visible: false
				dropShadowEnabled: false

				ValueAxis
				{
					id:mX
					tickCount: potentialRec.width/75
				}

				ValueAxis
				{
					id:mY
					tickCount: potentialRec.height/50
				}

				SpinBox
				{
					id: spinbox
                    maximumValue: 1000000000000
					onEditingFinished:
					{
						potentialViewer.changeTrackedNeuron(value)
					}
				}

				LineSeries
				{
					id: membranePotential
					name: "Approximate Membrane Potential - Discrete Values"
					axisX: mX
					axisY: mY
				}

				LineSeries 
				{
					id: threshold
					axisX: mX
					axisY: mY
				}

				Timer
				{
					id: refreshTimer3
					interval: refresh
					running: true
					repeat: true
					onTriggered:
					{
						potentialViewer.update(mX, mY, membraneChart.series(0),a);
						potentialViewer.update(mX, mY, membraneChart.series(1),b);
					}
				}

				PotentialViewer
				{
					objectName: "potentialViewer"
					id: potentialViewer
				}
			}
		}
	}
	onClosing:
	{
		inputViewer.disable();
    	outputViewer.disable();
    	potentialViewer.disable();
	}
}
)""
