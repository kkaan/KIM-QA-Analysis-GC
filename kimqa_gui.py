import sys
from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets

"""Simple PyQt GUI wrapper for the KIM QA analysis utilities."""

from analyse_kimqa import load_coordinates, analyse_static


class KimQaApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KIM QA Analysis")
        self.resize(700, 500)
        QtWidgets.QApplication.setStyle("Fusion")

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Analysis type
        analysis_group = QtWidgets.QGroupBox("Analysis type")
        analysis_layout = QtWidgets.QHBoxLayout()
        self.static_radio = QtWidgets.QRadioButton("Static")
        self.dynamic_radio = QtWidgets.QRadioButton("Dynamic")
        self.interrupt_radio = QtWidgets.QRadioButton("Treatment Interrupt")
        self.static_radio.setChecked(True)
        for w in (self.static_radio, self.dynamic_radio, self.interrupt_radio):
            analysis_layout.addWidget(w)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Linac vendor
        vendor_group = QtWidgets.QGroupBox("Linac Vendor")
        vendor_layout = QtWidgets.QHBoxLayout()
        self.varian_radio = QtWidgets.QRadioButton("Varian")
        self.varian_radio.setChecked(True)
        self.elekta_radio = QtWidgets.QRadioButton("Elekta/Varian with ADI")
        for w in (self.varian_radio, self.elekta_radio):
            vendor_layout.addWidget(w)
        vendor_group.setLayout(vendor_layout)
        layout.addWidget(vendor_group)

        # KIM log folder
        self.kim_folder_edit = QtWidgets.QLineEdit()
        kim_button = QtWidgets.QPushButton("Select KIM Log Folder")
        kim_button.clicked.connect(self._choose_kim_folder)
        layout.addLayout(self._hbox([kim_button, self.kim_folder_edit]))

        # Motion trace file
        self.motion_file_edit = QtWidgets.QLineEdit()
        motion_button = QtWidgets.QPushButton("Select Motion Trace")
        motion_button.clicked.connect(self._choose_motion_file)
        layout.addLayout(self._hbox([motion_button, self.motion_file_edit]))

        # Coordinate file
        self.coord_file_edit = QtWidgets.QLineEdit()
        coord_button = QtWidgets.QPushButton("Select Coordinate File")
        coord_button.clicked.connect(self._choose_coord_file)
        layout.addLayout(self._hbox([coord_button, self.coord_file_edit]))

        # Static shifts
        shift_group = QtWidgets.QGroupBox("Static Shifts (mm)")
        shift_layout = QtWidgets.QHBoxLayout()
        self.shift_lr = QtWidgets.QLineEdit("0")
        self.shift_si = QtWidgets.QLineEdit("0")
        self.shift_ap = QtWidgets.QLineEdit("0")
        for label, widget in [
            ("Lateral (LR):", self.shift_lr),
            ("Long (SI):", self.shift_si),
            ("Vert (AP):", self.shift_ap),
        ]:
            shift_layout.addWidget(QtWidgets.QLabel(label))
            shift_layout.addWidget(widget)
        shift_group.setLayout(shift_layout)
        layout.addWidget(shift_group)

        # Output folder
        self.output_folder_edit = QtWidgets.QLineEdit()
        out_button = QtWidgets.QPushButton("Select Output Folder")
        out_button.clicked.connect(self._choose_output_folder)
        layout.addLayout(self._hbox([out_button, self.output_folder_edit]))

        # Analyse button
        analyse_button = QtWidgets.QPushButton("Analyse")
        analyse_button.clicked.connect(self._run_analysis)
        layout.addWidget(analyse_button)

    def _hbox(self, widgets):
        box = QtWidgets.QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        return box

    def _choose_kim_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select KIM log folder")
        if folder:
            self.kim_folder_edit.setText(folder)
            self.output_folder_edit.setText(folder)

    def _choose_motion_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select motion trace", filter="Text files (*.txt)")
        if file_name:
            self.motion_file_edit.setText(file_name)

    def _choose_coord_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select coordinate file", filter="Text files (*.txt)")
        if file_name:
            self.coord_file_edit.setText(file_name)

    def _choose_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def _run_analysis(self):
        coord_path = self.coord_file_edit.text()
        motion_path = self.motion_file_edit.text()
        out_folder = self.output_folder_edit.text()
        if not all([coord_path, motion_path, out_folder]):
            QtWidgets.QMessageBox.warning(self, "Missing Data", "Please select required files/folders")
            return

        coords = load_coordinates(coord_path)
        motion = pd.read_csv(motion_path, sep="\s+", header=None).values[:, :3]
        mean, std, pct = analyse_static(coords[:-1, :], motion[: coords.shape[0] - 1, :])

        result = (
            f"Mean: {mean}\n"
            f"Std: {std}\n"
            f"Percentiles (5,95): {pct}"
        )
        QtWidgets.QMessageBox.information(self, "Static Analysis Result", result)
        out_file = Path(out_folder) / "analysis_result.txt"
        out_file.write_text(result)


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = KimQaApp()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
