R""(

/*
 * gui.qml
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 16/01/2018
 *
 * Information: QML file that defines the GUI.
 */

import QtQuick 2.1
import QtQuick.Controls 2.1
import QtQuick.Controls 1.4 as OldCtrl
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import QtCharts 2.1

import InputViewer 1.0
import OutputViewer 1.0
import DynamicsViewer 1.0

ApplicationWindow {
	id: mainWindow
	title: qsTr("Hummus")
	height: 900
	width: 900
	minimumHeight: 650
	minimumWidth: 450
	property int refresh: 100
	property int a: 0
	property int b: 1
	property int c: 2
	property bool pp: true
	visible: true
	color: "#FFFFFF"
	flags: Qt.Window | Qt.WindowFullscreenButtonHint
    
	ColumnLayout {
		id: mainGrid
		anchors.fill: parent

		Rectangle {
			id: menu
    		color: "#FFFFFF"
    		Layout.alignment: Qt.AlignCenter
			Layout.topMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: 27
    		radius: 2

			RoundButton {
				id: play
				text: qsTr("\u2759\u2759")
				anchors.centerIn: parent

				contentItem: Text {
					text: play.text
					color: play.down ? "#000000" : "#363636"
					horizontalAlignment: Text.AlignHCenter
        			verticalAlignment: Text.AlignVCenter
					elide: Text.ElideRight
				}

				background: Rectangle {
					color: play.down ? "#bdbebf" : "#FFFFFF"
					implicitWidth: 35
					implicitHeight: 35
					border.width: 1
					border.color: "#bdbebf"
					radius: 17.5
				}

				onClicked: {
					if (pp == true) {
						pp = false
						play.text = qsTr("\u25B6")
					}
					else {
						pp = true
						play.text = qsTr("\u2759\u2759")
					}
				}
			}

		}

		Rectangle {
			color: "#bdbebf"
			Layout.minimumWidth: mainGrid.width
			Layout.minimumHeight: 1
			Layout.alignment: Qt.AlignCenter
		}

		Rectangle {
			id: inputRec
			color: '#FFFFFF'
			Layout.alignment: Qt.AlignCenter
			Layout.topMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-19
			radius: 2

			ChartView {
				id: inputChart
				title: "Input Neurons"
				titleFont : Qt.font({bold: true})
				anchors.fill: parent
				antialiasing: true
				backgroundColor: '#FFFFFF'
				legend.visible: false
				dropShadowEnabled: false

				ValueAxis {
					id:inputX
					tickCount: inputRec.width/75
					titleText: "Time (ms)"
					labelsFont:Qt.font({pointSize: 11})
				}

				ValueAxis {
					id:inputY
					tickCount: inputRec.height/50
					titleText: "Input Neurons"
					labelsFont:Qt.font({pointSize: 11})
				}

				Text {
					id: sublayerLegend1
					text: "sublayers"
				}

				OldCtrl.SpinBox {
					id: sublayerbox1
					minimumValue: 0
					maximumValue: inputSublayer
					anchors.left: sublayerLegend1.right
					anchors.leftMargin: 5

					onEditingFinished: {
						inputViewer.changeSublayer(value)
					}
				}

				ScatterSeries {
					id: input
					name: "Input Neurons"
					markerSize: 5
					markerShape: ScatterSeries.MarkerShapeCircle
					axisX: inputX
					axisY: inputY
					borderColor: 'transparent'
				}

				Timer {
					id: refreshTimer
					interval: refresh
					running: pp
					repeat: true
					onTriggered: {
						inputViewer.update(inputX, inputY, inputChart.series(0));
					}
				}

				InputViewer {
					objectName: "inputViewer"
					id: inputViewer
				}
			}
		}

		Rectangle {
			color: "#bdbebf"
			Layout.minimumWidth: mainGrid.width
			Layout.minimumHeight: 1
			Layout.alignment: Qt.AlignCenter
		}

		Rectangle {
			id: outputRec
			color: '#FFFFFF'
			Layout.alignment: Qt.AlignCenter
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-19
			radius: 2

			ChartView {
				id: outputChart
				title: "Downstream Neurons"
				titleFont : Qt.font({bold: true})
				anchors.fill: parent
				antialiasing: true
				backgroundColor: '#FFFFFF'
				legend.visible: false
				dropShadowEnabled: false

				ValueAxis {
					id:outputX
					tickCount: outputRec.width/75
					titleText: "Time (ms)"
					labelsFont:Qt.font({pointSize: 11})
				}

				ValueAxis {
					id:outputY
					tickCount: outputRec.height/50
					titleText: "Downstream Neurons"
					labelsFont:Qt.font({pointSize: 11})

				}

				Text {
					id: layerLegend
					text: "layer"
				}

				OldCtrl.SpinBox {
					id: layerbox
					minimumValue: 1
					maximumValue: layers
					anchors.left: layerLegend.right
					anchors.leftMargin: 5
					onEditingFinished: {
						outputViewer.changeLayer(value)
					}
				}

				Text {
					id: sublayerLegend
					text: "sublayers"
					anchors.left: layerbox.right
					anchors.leftMargin: 20
				}

				OldCtrl.SpinBox {
					id: sublayerbox
					minimumValue: 0
					maximumValue: sublayers
					anchors.left: sublayerLegend.right
					anchors.leftMargin: 5
					onEditingFinished: {
						outputViewer.changeSublayer(value)
					}
				}

				ScatterSeries {
					id: output
					name: "Output Neurons"
					markerSize: 5
					markerShape: ScatterSeries.MarkerShapeCircle
					axisX: outputX
					axisY: outputY
					borderColor: 'transparent'
				}

				Timer {
					id: refreshTimer2
					interval: refresh
					running: pp
					repeat: true
					onTriggered: {
						outputViewer.update(outputX, outputY, outputChart.series(0));
					}
				}

				OutputViewer {
					objectName: "outputViewer"
					id: outputViewer
				}

			}
		}

		Rectangle {
			color: "#bdbebf"
			Layout.minimumWidth: mainGrid.width
			Layout.minimumHeight: 1
			Layout.alignment: Qt.AlignCenter
		}

		Rectangle {
			id: potentialRec
			color: '#FFFFFF'
			Layout.alignment: Qt.AlignCenter
			Layout.bottomMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-19
			radius: 2

			ChartView {
				id: membraneChart
				title: "Neuron Dynamics"
				titleFont : Qt.font({bold: true})
				anchors.fill: parent
				antialiasing: true
				backgroundColor: '#FFFFFF'
				legend.visible: false
				dropShadowEnabled: false

				ValueAxis {
					id:mX
					tickCount: potentialRec.width/75
					titleText: "Time (ms)"
					labelsFont:Qt.font({pointSize: 11})
				}

				ValueAxis {
					id:mY
					tickCount: potentialRec.height/50
					titleText: "Membrane Potential (mV)"
					labelsFont:Qt.font({pointSize: 11})
				}

				ValueAxis {
					id:mY_right
					tickCount: potentialRec.height/50
					titleText: "Injected Current (A)"
					labelsFont:Qt.font({pointSize: 11})
				}

				Text {
					id: neuronLegend
					text: "neuron"
				}

				OldCtrl.SpinBox {
					id: spinbox
                    maximumValue: numberOfNeurons
                    anchors.left: neuronLegend.right
					anchors.leftMargin: 5
					onEditingFinished: {
						dynamicsViewer.changeTrackedNeuron(value)
					}
				}

				LineSeries {
					id: membranePotential
					name: "Neuron Dynamics"
					axisX: mX
					axisY: mY
				}

				LineSeries {
					id: threshold
					axisX: mX
					axisY: mY
					color: "#ED6A56"
				}

				LineSeries {
					id: injectedCurrent
					axisX: mX
					axisYRight: mY_right
				}

				Timer {
					id: refreshTimer3
					interval: refresh
					running: pp
					repeat: true
					onTriggered: {
						dynamicsViewer.update(mX, mY, membraneChart.series(0),a);
						dynamicsViewer.update(mX, mY, membraneChart.series(1),b);
						dynamicsViewer.update(mX, mY_right, membraneChart.series(2),c);
					}
				}

				DynamicsViewer {
					objectName: "dynamicsViewer"
					id: dynamicsViewer
				}
			}
		}
	}
	onClosing: {
		inputViewer.disable();
    	outputViewer.disable();
    	dynamicsViewer.disable();
	}
}
)""
