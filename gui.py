"""
File Name: gui.py
Description: Optimized GUI with a Search History sidebar.
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
    QListWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal
from hybrid_search import ArtGallerySearchEngine
from ingest_data import OUTPUT_FILE


class EngineLoadThread(QThread):
    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            engine = ArtGallerySearchEngine(OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ArtSearchGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")
        self.resize(1100, 750)  # Wider for sidebar

        self.engine = None
        self.image_cache = {}
        self.apply_modern_styling()

        # Main Layout (Horizontal: Sidebar + Content)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_sidebar()
        self._setup_main_area()

        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.start()

    def apply_modern_styling(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QWidget { font-family: 'Segoe UI', sans-serif; color: #ccc; }
            QLineEdit { padding: 12px; border-radius: 8px; background: #1a1a1a; color: white; border: 1px solid #333; }
            QPushButton { padding: 12px; border-radius: 8px; background: #2980b9; color: white; font-weight: bold; }
            QTextBrowser { background: #141414; border: 1px solid #222; border-radius: 8px; padding: 10px; }
            QListWidget { background: #1a1a1a; border: none; border-radius: 8px; padding: 5px; outline: none; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { background: #2980b9; color: white; border-radius: 4px; }
        """)

    def _setup_sidebar(self):
        """Creates the search history sidebar."""
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 10, 0)

        lbl = QLabel("SEARCH HISTORY")
        lbl.setStyleSheet(
            "font-weight: bold; color: #555; font-size: 11px; margin-bottom: 5px;"
        )

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setFixedWidth(200)

        sidebar.addWidget(lbl)
        sidebar.addWidget(self.history_list)
        self.main_layout.addLayout(sidebar)

    def _setup_main_area(self):
        """Standard search area."""
        content = QVBoxLayout()

        nav = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search the collection...")
        self.search_input.returnPressed.connect(self.perform_search)

        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.perform_search)

        nav.addWidget(self.search_input)
        nav.addWidget(btn_search)

        self.status = QLabel("Initializing AI...")
        self.results_area = QTextBrowser()

        content.addLayout(nav)
        content.addWidget(self.status)
        content.addWidget(self.results_area)
        self.main_layout.addLayout(content)

    def _on_engine_ready(self, engine):
        self.engine = engine
        self.status.setText("Engine Online.")
        self.status.setStyleSheet("color: #2ecc71;")

    def _on_history_clicked(self, item):
        """Re-runs a search when clicking a history item."""
        self.search_input.setText(item.text())
        self.perform_search()

    def _get_img_b64(self, url, title):
        if url in self.image_cache:
            return self.image_cache[url]
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            b64 = base64.b64encode(r.content).decode() if r.status_code == 200 else None
        except Exception:
            b64 = None

        if not b64:
            safe_t = urllib.parse.quote(title[:12])
            r = requests.get(
                f"https://placehold.co/140x100/2c3e50/ecf0f1?text={safe_t}", timeout=3
            )
            b64 = base64.b64encode(r.content).decode()

        self.image_cache[url] = b64
        return b64

    def perform_search(self):
        query = self.search_input.text().strip()
        if not query or self.engine is None:
            return

        # Add to history if it's a new query
        existing_items = [
            self.history_list.item(i).text() for i in range(self.history_list.count())
        ]
        if query not in existing_items:
            self.history_list.insertItem(0, query)

        results = self.engine.hybrid_search(query)
        self.display_results(results, query)
        self.status.setText(f"Found {len(results)} items.")

    def display_results(self, results, query):
        html = f"<h2 style='color:white'>Results: '{query}'</h2><hr>"
        for res in results:
            b64 = self._get_img_b64(res["Thumbnail"], res["Title"])
            html += (
                f"<table width='100%' style='margin-bottom:15px'><tr>"
                f"<td width='150' valign='top'>"
                f"<img src='data:image/jpeg;base64,{b64}' width='140' style='border-radius:4px;'/></td>"
                f"<td valign='top'>"
                f"<b style='color:#3498db; font-size:17px'>{res['Rank']}. {res['Title']}</b><br>"
                f"<span style='color:#ecf0f1;'>Artist: {res['Artist']}</span><br>"
                f"<i style='color:#666; font-size:13px'>{res['Description']}</i></td>"
                f"</tr></table><hr style='border:none; border-top:1px solid #222;'>"
            )
        self.results_area.setHtml(html)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ArtSearchGUI()
    win.show()
    sys.exit(app.exec())
