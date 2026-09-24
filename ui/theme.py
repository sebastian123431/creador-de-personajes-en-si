DARK_THEME_QSS = """
QMainWindow, QDialog {
    background-color: #121418;
    color: #e2e8f0;
}

QWidget {
    font-family: 'Segoe UI', 'Roboto', 'Inter', sans-serif;
    font-size: 13px;
    color: #cbd5e1;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #181b20;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #262b34;
    background: #181b22;
    border-radius: 6px;
}
QTabBar::tab {
    background: #181b22;
    color: #94a3b8;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #232833;
    color: #38bdf8;
    font-weight: bold;
    border-bottom: 2px solid #38bdf8;
}
QTabBar::tab:hover {
    background: #1e232d;
    color: #f1f5f9;
}

/* Lists and Trees */
QListWidget, QTreeWidget, QTableWidget {
    background-color: #161920;
    border: 1px solid #262c38;
    border-radius: 6px;
    padding: 4px;
    color: #e2e8f0;
}
QListWidget::item {
    padding: 8px 10px;
    border-radius: 4px;
    margin-bottom: 2px;
}
QListWidget::item:selected {
    background: #0284c7;
    color: #ffffff;
    font-weight: 600;
}
QListWidget::item:hover:!selected {
    background: #1e2430;
}

/* Buttons */
QPushButton {
    background-color: #242b38;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 7px 15px;
    color: #f8fafc;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #1e293b;
}
QPushButton:disabled {
    background-color: #1a1e24;
    border-color: #242930;
    color: #64748b;
}

QPushButton#primaryButton {
    background-color: #0284c7;
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #0369a1;
}

QPushButton#successButton {
    background-color: #16a34a;
    border: 1px solid #22c55e;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#successButton:hover {
    background-color: #15803d;
}

QPushButton#dangerButton {
    background-color: #dc2626;
    border: 1px solid #ef4444;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#dangerButton:hover {
    background-color: #b91c1c;
}

/* GroupBox and Frames */
QGroupBox {
    border: 1px solid #262c38;
    border-radius: 6px;
    margin-top: 20px;
    padding-top: 10px;
    font-weight: bold;
    color: #94a3b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #121418;
    color: #38bdf8;
}

/* Labels and Inputs */
QLabel {
    color: #cbd5e1;
}
QLineEdit, QComboBox {
    background-color: #1a1e26;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px 10px;
    color: #f1f5f9;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
}

/* ToolBar & Menu */
QMenuBar {
    background-color: #121418;
    border-bottom: 1px solid #242932;
    color: #cbd5e1;
}
QMenuBar::item:selected {
    background-color: #242b38;
}
QMenu {
    background-color: #181b22;
    border: 1px solid #2b3240;
    color: #e2e8f0;
}
QMenu::item:selected {
    background-color: #0284c7;
}

QStatusBar {
    background-color: #0e1014;
    border-top: 1px solid #20242c;
    color: #94a3b8;
}
"""
