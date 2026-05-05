import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from merlin_track_position.server import MotorServer


class _MainWindowGUI(QtWidgets.QMainWindow):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self.setWindowTitle("Track Positions")


class MainWindow(_MainWindowGUI):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self._server = MotorServer(self)
        self._server.sigMoveDetected.connect(self._on_move_detected)
        self._server.start()

        self.plot_widget = pg.PlotWidget()

    @QtCore.Slot(int)
    def _on_move_detected(self, target: int) -> None:
        self._server.set_result(True, "")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.server.stop()
        self.server.wait()

        super().closeEvent(event)
