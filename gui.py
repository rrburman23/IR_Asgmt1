"""
File Name: gui.py
Description: A PyQt6 graphical interface for the Art Gallery Hybrid Search Engine.
"""

# pylint: disable=no-name-in-module

import sys
import base64
import requests

import torch  # pylint: disable=unused-import # noqa: F401
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QLabel,
)
from PyQt6.QtCore import QThread, pyqtSignal

from hybrid_search import ArtGallerySearchEngine
from ingest_data import OUTPUT_FILE


# ==========================================
# 1. Background Worker Thread
# ==========================================
class EngineLoadThread(QThread):
    """Loads the search engine in the background to keep the UI responsive."""

    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        """Executes the engine initialization."""
        try:
            # pylint: disable=broad-exception-caught
            engine = ArtGallerySearchEngine(OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ==========================================
# 2. Main GUI Application
# ==========================================
class ArtSearchGUI(QMainWindow):
    """Main window class for the Art Gallery Search GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")
        self.resize(950, 750)
        self.apply_modern_styling()

        self.engine = None
        self.load_thread = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        self._setup_ui()
        self._start_engine_thread()

    def apply_modern_styling(self):
        """Injects QSS (Qt Style Sheets) for a dark-mode aesthetic."""
        self.setStyleSheet(
            """
            QMainWindow { background-color: #121212; }
            QWidget { font-family: 'Segoe UI', sans-serif; }
            QLineEdit {
                padding: 12px; border: 1px solid #333333;
                border-radius: 8px; background-color: #1e1e1e;
                color: #ffffff; font-size: 15px;
            }
            QLineEdit:focus { border: 1px solid #3498db; }
            QPushButton {
                padding: 12px 24px; background-color: #2980b9;
                color: white; border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:disabled { background-color: #2c3e50; color: #7f8c8d; }
            QTextBrowser {
                background-color: #1e1e1e; color: #e0e0e0;
                border: 1px solid #333333; border-radius: 8px; padding: 15px;
            }
        """
        )

    def _setup_ui(self):
        """Sets up the search bar, buttons, and results display area."""
        self.top_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for artwork...")
        self.search_input.returnPressed.connect(self.perform_search)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.perform_search)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_results)
        self.clear_button.setStyleSheet(
            """
            QPushButton { background-color: #7f8c8d; }
            QPushButton:hover { background-color: #95a5a6; }
        """
        )

        self.top_layout.addWidget(self.search_input)
        self.top_layout.addWidget(self.search_button)
        self.top_layout.addWidget(self.clear_button)

        self.status_label = QLabel("Loading AI models... This may take a moment.")
        self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")

        self.results_display = QTextBrowser()
        self.results_display.setOpenExternalLinks(True)

        self.layout.addLayout(self.top_layout)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.results_display)

        self.search_input.setEnabled(False)
        self.search_button.setEnabled(False)
        self.clear_button.setEnabled(False)

    def _start_engine_thread(self):
        """Initializes the background thread to load the search engine."""
        self.load_thread = EngineLoadThread()
        self.load_thread.engine_ready.connect(self.on_engine_ready)
        self.load_thread.error_occurred.connect(self.on_engine_error)
        self.load_thread.start()

    def on_engine_ready(self, loaded_engine):
        """Slot called when the background thread finishes successfully."""
        self.engine = loaded_engine
        self.status_label.setText("Engine Online. Ready for queries.")
        self.status_label.setStyleSheet("color: #2ecc71;")

        self.search_input.setEnabled(True)
        self.search_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.search_input.setFocus()

    def on_engine_error(self, error_msg):
        """Slot called if the background thread crashes."""
        self.status_label.setText(f"Critical Error: {error_msg}")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def perform_search(self):
        """Executes the hybrid search and formats the output."""
        query = self.search_input.text().strip()
        if not query:
            return

        self.status_label.setText(f"Searching for: '{query}'...")
        self.status_label.setStyleSheet("color: #3498db;")
        QApplication.processEvents()

        try:
            # pylint: disable=broad-exception-caught
            results = self.engine.hybrid_search(query, top_k=10)
            self.display_results(results, query)
            self.status_label.setText(f"Found {len(results)} results.")
            self.status_label.setStyleSheet("color: #2ecc71;")
        except Exception as e:
            self.results_display.setHtml(f"<p style='color:#e74c3c;'>Error: {e}</p>")
            self.status_label.setText("Error during search.")

    def clear_results(self):
        """Clears the search bar and the results display."""
        self.search_input.clear()
        self.results_display.clear()
        self.status_label.setText("Engine Online. Ready for queries.")
        self.status_label.setStyleSheet("color: #2ecc71;")
        self.search_input.setFocus()

    def display_results(self, results, query):
        """Formats the dictionary results into readable HTML with base64 images."""
        if not results:
            self.results_display.setHtml(
                f"<h3 style='color:#e0e0e0;'>No results found for '{query}'.</h3>"
            )
            return

        html_output = (
            f"<h2 style='color: #ffffff; margin-bottom: 20px;'>"
            f"Top Results for '{query}'</h2>"
        )

        for res in results:
            title = res.get("Title", "Unknown Title").title()
            artist = res.get("Artist", "Unknown Artist").title()
            medium = res.get("Description", "No description available.")
            score = res.get("Score", 0.0)
            thumbnail_url = res.get("Thumbnail", "")

            # Default placeholder broken into multiple lines
            img_html = (
                "<div style='width:140px; height:100px; background-color:#2c3e50; "
                "text-align:center; line-height:100px; color:#7f8c8d; "
                "border-radius:4px;'>No Image</div>"
            )

            if thumbnail_url and str(thumbnail_url).startswith("http"):
                try:
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36"
                        )
                    }
                    img_res = requests.get(thumbnail_url, headers=headers, timeout=5)

                    if img_res.status_code == 200:
                        b64_data = base64.b64encode(img_res.content).decode("utf-8")
                        img_html = (
                            f"<img src='data:image/jpeg;base64,{b64_data}' "
                            f"width='140' style='border-radius:4px;'/>"
                        )
                    else:
                        print(f"Server rejected image: HTTP {img_res.status_code}")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Failed to load image for {title}: {e}")

            # Build the HTML table using multiple string concatenations
            html_output += (
                f"<table width='100%' style='margin-bottom: 15px;'>"
                f"<tr><td width='150' valign='top'>{img_html}</td>"
                f"<td valign='top'>"
                f"<h3 style='margin: 0; color: #3498db; font-size: 18px;'>"
                f"{res['Rank']}. {title}</h3>"
                f"<p style='margin: 4px 0 2px 0; color: #ecf0f1; font-size: 15px;'>"
                f"<b>Artist:</b> {artist}</p>"
                f"<p style='margin: 2px 0; color: #bdc3c7; font-size: 14px;'>"
                f"<i>Medium: {medium}</i></p>"
                f"<p style='margin: 4px 0 0 0; font-size: 12px; color: #7f8c8d;'>"
                f"Relevance Score: {score}</p></td></tr></table>"
                f"<hr style='border:none; border-top:1px solid #333333; margin:15px 0;'>"
            )

        self.results_display.setHtml(html_output)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ArtSearchGUI()
    window.show()
    sys.exit(app.exec())
