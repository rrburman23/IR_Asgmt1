"""
File Name: gui.py
Description: Optimized PyQt6 GUI with image caching and instant text rendering.
             Uses an internal dictionary to prevent redundant web requests.
"""

# pylint: disable=no-name-in-module, broad-exception-caught

import sys
import base64
import urllib.parse
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
    """Initializes the AI backend without locking the GUI window."""

    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        """Background execution logic."""
        try:
            engine = ArtGallerySearchEngine(OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ==========================================
# 2. Main GUI Application
# ==========================================
class ArtSearchGUI(QMainWindow):
    """Optimized interface for high-speed semantic art discovery."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tate Gallery | High-Speed Semantic Search")
        self.resize(950, 750)

        # Memory Cache: Prevents re-downloading images during a single session
        self.image_cache = {}

        self.apply_modern_styling()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        self._setup_ui()

        # Trigger the background AI engine loader
        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.error_occurred.connect(self._on_error)
        self.engine_thread.start()

    def apply_modern_styling(self):
        """Sets the dark-mode aesthetic via QSS."""
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QWidget { font-family: 'Segoe UI', sans-serif; }
            QLineEdit { padding: 12px; border-radius: 8px; background: #1a1a1a; color: white; border: 1px solid #333; }
            QPushButton { padding: 12px; border-radius: 8px; background: #2980b9; color: white; font-weight: bold; }
            QPushButton:hover { background: #3498db; }
            QTextBrowser { background: #141414; color: #ccc; border: 1px solid #222; border-radius: 8px; padding: 10px; }
        """)

    def _setup_ui(self):
        """Constructs the search bar and results display."""
        nav = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search the collection (e.g., 'sunset over the sea')..."
        )
        self.search_input.returnPressed.connect(self.perform_search)

        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.perform_search)

        nav.addWidget(self.search_input)
        nav.addWidget(btn_search)

        self.status = QLabel("Initializing AI... (Loading Cache)")
        self.status.setStyleSheet("color: #f39c12; font-style: italic;")

        self.results_area = QTextBrowser()

        self.layout.addLayout(nav)
        self.layout.addWidget(self.status)
        self.layout.addWidget(self.results_area)

    def _on_engine_ready(self, engine):
        """Enables search interface once the engine is hot."""
        self.engine = engine
        self.status.setText("Engine Online. Ready.")
        self.status.setStyleSheet("color: #2ecc71;")

    def _on_error(self, err):
        """Displays error if initialization fails."""
        self.status.setText(f"Critical Error: {err}")
        self.status.setStyleSheet("color: #e74c3c;")

    def _get_img_b64(self, url, title):
        """Fetches, patches, and encodes images with session caching."""
        if url in self.image_cache:
            return self.image_cache[url]

        # Patch dead Tate URLs to the new CDN on-the-fly
        target_url = (
            url.replace("http://www.tate.org.uk", "https://media.tate.org.uk")
            if "tate.org.uk" in url
            else url
        )

        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(target_url, headers=headers, timeout=3)
            if r.status_code != 200:
                raise Exception("Dead Link")
            b64 = base64.b64encode(r.content).decode()
        except Exception:
            # Graceful Fallback: Generate a placeholder image via API
            safe_t = urllib.parse.quote(title[:12] + ".." if len(title) > 12 else title)
            f_url = f"https://placehold.co/140x100/2c3e50/ecf0f1?text={safe_t}"
            r = requests.get(f_url, timeout=3)
            b64 = base64.b64encode(r.content).decode()

        self.image_cache[url] = b64
        return b64

    def perform_search(self):
        """Handles the search event and renders results."""
        query = self.search_input.text().strip()
        if not query or not hasattr(self, "engine"):
            return

        self.status.setText("Retrieving matches...")
        QApplication.processEvents()  # Refresh label

        results = self.engine.hybrid_search(query)
        self.display_results(results, query)
        self.status.setText(f"Completed. Found {len(results)} items.")

    def display_results(self, results, query):
        """Compiles search results into rich HTML."""
        html = f"<h2 style='color:white'>Search Results: '{query}'</h2><hr>"

        for res in results:
            # Fetch image (will pull from cache if repeated)
            b64 = self._get_img_b64(res["Thumbnail"], res["Title"])

            html += f"""
            <table width='100%' style='margin-bottom:15px'>
                <tr>
                    <td width='150' valign='top'>
                        <img src='data:image/jpeg;base64,{b64}' width='140' style='border-radius:4px;'/>
                    </td>
                    <td valign='top'>
                        <b style='color:#3498db; font-size:17px'>{res["Rank"]}. {res["Title"].title()}</b><br>
                        <span style='color:#ecf0f1; font-size:14px'><b>Artist:</b> {res["Artist"].title()}</span><br>
                        <i style='color:#999; font-size:13px'>Medium: {res["Description"]}</i><br>
                        <span style='color:#666; font-size:11px'>Relevance: {res["Score"]}</span>
                    </td>
                </tr>
            </table>
            <hr style='border:none; border-top:1px solid #222;'>
            """
        self.results_area.setHtml(html)

    def clear_results(self):
        """Resets the UI view."""
        self.search_input.clear()
        self.results_area.clear()
        self.status.setText("Engine Online. Ready.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ArtSearchGUI()
    win.show()
    sys.exit(app.exec())
