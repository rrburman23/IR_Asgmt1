"""
File Name: gui.py
Description: Provides the PyQt6 graphical user interface for the Tate Gallery
             Search Engine. Implements background loading, memory caching for
             images, transparency reporting, and clickable spelling suggestions.
"""

import time
import base64
import urllib.parse
import requests

# pylint: disable=no-name-in-module
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

# ==========================================
# 1. Background Worker Thread
# ==========================================
class EngineLoadThread(QThread):
    """Initializes the AI search engine on a separate background thread."""

    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        """Standard QThread execution method."""
        try:
            engine = ArtGallerySearchEngine(OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:  # pylint: disable=broad-except
            self.error_occurred.emit(str(e))


# ==========================================
# 2. Main GUI Application
# ==========================================
class ArtSearchGUI(QMainWindow):
    """The main window class. Renders rich HTML results with transparency."""

    def __init__(self):
        """Initializes the GUI and triggers background model loading."""
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")
        self.resize(1150, 800)

        self.engine = None
        self.image_cache = {}

        self.apply_modern_styling()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_sidebar()
        self._setup_main_area()

        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.start()

    def apply_modern_styling(self):
        """Defines the CSS-like Dark Mode aesthetic for all widgets."""
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QWidget { font-family: 'Segoe UI', sans-serif; color: #ccc; }
            QLineEdit { 
                padding: 12px; border-radius: 8px; 
                background: #1a1a1a; color: white; border: 1px solid #333; 
            }
            QPushButton { 
                padding: 12px; border-radius: 8px; 
                background: #2980b9; color: white; font-weight: bold; 
            }
            QPushButton:hover { background: #3498db; }
            QPushButton#clearBtn { background: #333; color: #888; font-size: 11px; }
            QPushButton#clearBtn:hover { background: #e74c3c; color: white; }
            QTextBrowser { 
                background: #141414; border: 1px solid #222; 
                border-radius: 8px; padding: 10px; 
            }
            QListWidget { 
                background: #1a1a1a; border: none; 
                border-radius: 8px; padding: 5px; outline: none; 
            }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { 
                background: #2980b9; color: white; border-radius: 4px; 
            }
        """)

    def _setup_sidebar(self):
        """Initializes the left-hand sidebar for search history tracking."""
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 10, 0)
        lbl = QLabel("SEARCH HISTORY")
        lbl.setStyleSheet(
            "font-weight: bold; color: #555; font-size: 11px; margin-bottom: 5px;"
        )
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setFixedWidth(220)
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.setObjectName("clearBtn")
        self.clear_history_btn.clicked.connect(self.clear_history)
        sidebar.addWidget(lbl)
        sidebar.addWidget(self.history_list)
        sidebar.addWidget(self.clear_history_btn)
        self.main_layout.addLayout(sidebar)

    def _setup_main_area(self):
        """Initializes the search input and HTML result viewing area."""
        content = QVBoxLayout()
        nav = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search the collection...")
        self.search_input.returnPressed.connect(self.perform_search)
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.perform_search)
        nav.addWidget(self.search_input)
        nav.addWidget(btn_search)
        self.status = QLabel("Initializing...")
        self.status.setStyleSheet("color: #f39c12;")

        self.results_area = QTextBrowser()
        self.results_area.setOpenLinks(False)
        self.results_area.anchorClicked.connect(self._on_suggestion_clicked)

        content.addLayout(nav)
        content.addWidget(self.status)
        content.addWidget(self.results_area)
        self.main_layout.addLayout(content)

    def _on_engine_ready(self, engine):
        """Slot for handling engine initialization completion."""
        self.engine = engine
        self.status.setText("Engine Online - Ready for Queries!")
        self.status.setStyleSheet("color: #2ecc71;")

    def _on_history_clicked(self, item):
        """Populates the search bar with historical text and re-searches."""
        self.search_input.setText(item.text())
        self.perform_search()

    def _on_suggestion_clicked(self, qurl):
        """Slot to intercept clicks on 'Did you mean' links."""
        suggested_query = qurl.toString()
        self.search_input.setText(suggested_query)
        self.perform_search()

    def clear_history(self):
        """Wipes the interaction history sidebar."""
        self.history_list.clear()

    def _get_img_b64(self, url: str, title: str):
        """Fetches and encodes images to Base64 to bypass local file restrictions."""
        if url in self.image_cache:
            return self.image_cache[url]
        b64 = None
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
            if r.status_code == 200:
                b64 = base64.b64encode(r.content).decode()
        except Exception:  # pylint: disable=broad-except
            b64 = None
        if not b64:
            try:
                safe_t = urllib.parse.quote(title[:15])
                f_url = f"https://placehold.co/140x100/2c3e50/ecf0f1?text={safe_t}"
                r = requests.get(f_url, timeout=3)
                b64 = base64.b64encode(r.content).decode()
            except Exception:  # pylint: disable=broad-except
                b64 = None
        self.image_cache[url] = b64
        return b64

    def perform_search(self):
        """Routes search execution and handles administrative slash commands."""
        query = self.search_input.text().strip()
        if not query or self.engine is None:
            return

        if query.lower() in ["/test", "test", "/evaluate", "evaluate", "/eval"]:
            self.status.setText("Running System Task... Check Terminal")
            self.status.setStyleSheet("color: #3498db;")
            QApplication.processEvents()
            # pylint: disable=import-outside-toplevel
            from main import launch_tests, launch_evaluation

            if "test" in query.lower():
                launch_tests()
            else:
                launch_evaluation()
            self.status.setText("Task Completed. Engine Online.")
            self.status.setStyleSheet("color: #2ecc71;")
            self.search_input.clear()
            return

        # Safely check for None when accessing list items
        existing = []
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            if item is not None:
                existing.append(item.text())

        if query not in existing:
            self.history_list.insertItem(0, query)

        # Execute Search with a local timer to measure performance and provide user feedback
        search_start = time.perf_counter()
        results = self.engine.hybrid_search(query)
        search_end = time.perf_counter()
        search_time = search_end - search_start

        # Pass the time to the display function
        self.display_results(results, query, search_time)

        # Update Status Bar
        total_items = len(self.engine.df)

        self.status.setText(
            f"Complete | {total_items} items indexed on {self.engine.device.upper()}"
        )
        self.status.setStyleSheet("color: #2ecc71;")

    def display_results(self, results: list, query: str, search_time: float = 0.0):
        """Renders the HTML result cards and clickable suggestions into the display area."""
        suggestion_html = ""
        if results and results[0].get("Suggestion"):
            sug = results[0]["Suggestion"]
            suggestion_html = (
                f"<div style='color:#f39c12; margin-bottom:10px;'>"
                f"<i>Did you mean: <a href='{sug}' style='color:#3498db; "
                f"text-decoration:none;'><b>{sug}</b></a>?</i></div>"
            )
        # Inject the search time into the header for user feedback on performance
        html = (
            f"{suggestion_html}"
            f"<h2 style='color:white; margin-bottom:10px;'>"
            f"Results: '{query}' "
            f"<span style='color:#888; font-size:14px; font-weight:normal;'>"
            f"({search_time:.3f} seconds)</span></h2><hr>"
        )

        for res in results:
            b64 = self._get_img_b64(res["Thumbnail"], res["Title"])

            # Break long string to satisfy character limit
            img_part = f"src='data:image/jpeg;base64,{b64}'" if b64 else ""
            img_html = (
                f"<img {img_part} width='140' style='border-radius:4px;'/>"
                if b64
                else "[No Image]"
            )

            # Assemble rows with shorter concatenation to keep lines under 100
            row = "<table width='100%' style='margin-bottom:15px'><tr>"
            row += f"<td width='160' valign='top'>{img_html}</td>"
            row += "<td valign='top'>"
            row += f"<b style='color:#3498db; font-size:18px'>{res['Rank']}. {res['Title']}</b><br>"
            row += f"<span style='color:#ecf0f1; font-size:14px'>Artist: {res['Artist']}</span><br>"
            row += (
                f"<i style='color:#666; font-size:13px;'>{res['Description']}</i><br>"
            )
            row += "<div style='color:#2ecc71; font-size:11px; margin-top:5px;'>"
            row += f"<b>Match Logic:</b> {res['Reasons']}</div>"
            row += f"<span style='color:#444; font-size:11px;'>Score: {res['Score']}</span></td>"
            row += "</tr></table><hr style='border:none; border-top:1px solid #222;'>"
            html += row

        self.results_area.setHtml(html)
