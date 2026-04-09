"""
File Name: gui.py

Description:
PyQt6 GUI for the Tate Art Search Engine.
- Async background loading/searching using QThread.
- Centered pagination, clickable browser links, search history sidebar.
- Stream redirection is handled robustly to prevent leaking file handles.
- User-friendly/robust for PyInstaller EXE builds on Windows.
"""

import sys
import os
import time
import base64
import requests
import webbrowser
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QApplication,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QLabel,
    QListWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QIcon


# --- Native Stream Redirection (Safe, PEP8-compliant) ---
def redirect_streams():
    """
    In PyInstaller GUI mode, sys.stdout/stderr may be None.
    This helper makes sure they are safely redirected and closed at program exit.
    """
    import atexit

    streams = []
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
        streams.append(sys.stdout)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
        streams.append(sys.stderr)
    for s in streams:
        atexit.register(s.close)


redirect_streams()
# --------------------------------------------------------


class EngineLoadThread(QThread):
    """Loads the heavy DataFrame and Models in the background on startup."""

    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            from hybrid_search import ArtGallerySearchEngine
            import ingest_data

            engine = ArtGallerySearchEngine(ingest_data.OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SearchWorker(QThread):
    """
    Runs the heavy hybrid search in a background thread to keep GUI responsive.
    """

    results_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, engine, query: str, page: int, per_page: int):
        super().__init__()
        self.engine = engine
        self.query = query
        self.page = page
        self.per_page = per_page

    def run(self):
        try:
            data = self.engine.hybrid_search(
                self.query, page=self.page, per_page=self.per_page
            )
            self.results_ready.emit(data)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ArtSearchGUI(QMainWindow):
    """
    Main window: search box, history, result browser, and navigation controls.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")

        # Use icon if present, fallback to no icon if not
        if os.path.exists("icon.ico"):
            self.setWindowIcon(QIcon("icon.ico"))
        self.setMinimumSize(900, 600)
        self.resize(1150, 800)

        self.engine = None
        self.search_worker = None
        self.search_start_time = 0.0
        self.image_cache: Dict[str, str] = {}  # img URL -> base64

        # Pagination state
        self.current_query = ""
        self.current_page = 1
        self.per_page = 10
        self.total_pages = 1

        self.apply_styling()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_sidebar()
        self._setup_main_area()

        # Start-up: disable input until model/data loaded
        self.search_input.setDisabled(True)
        self.btn_search.setDisabled(True)

        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.error_occurred.connect(self._on_engine_error)
        self.engine_thread.start()

    def apply_styling(self):
        """Sets unified stylesheet for all widgets."""
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QWidget { font-family: 'Segoe UI', sans-serif; color: #ccc; }
            QLineEdit {
                padding: 12px; border-radius: 8px;
                background: #1a1a1a; color: white; border: 1px solid #333;
                font-size: 14px;
            }
            QPushButton {
                padding: 12px; border-radius: 8px;
                background: #2980b9; color: white; font-weight: bold;
            }
            QPushButton:hover { background: #3498db; }
            QPushButton:disabled { background: #2c3e50; color: #7f8c8d; }
            QPushButton#clearBtn {
                background: #222; color: #888; font-size: 11px; margin-top: 5px;
            }
            QPushButton#clearBtn:hover { background: #e74c3c; color: white; }
            QPushButton#navBtn {
                background: #1e1e1e; color: #3498db; font-size: 12px; padding: 6px 16px;
                border: 1px solid #333;
            }
            QPushButton#navBtn:hover { background: #252525; border-color: #3498db; }
            QPushButton#navBtn:disabled { color: #444; border-color: #222; }
            QTextBrowser { background: #141414; border: 1px solid #222; border-radius: 8px; padding: 10px; }
            QListWidget { background: #1a1a1a; border: none; border-radius: 8px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { background: #2980b9; color: white; }
        """)

    def _setup_sidebar(self):
        """Builds the left sidebar (search history and clear button)."""
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 10, 0)
        lbl = QLabel("SEARCH HISTORY")
        lbl.setStyleSheet(
            "font-weight: bold; color: #555; font-size: 11px; margin-bottom: 5px;"
        )
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setFixedWidth(220)
        self.btn_clear = QPushButton("Clear History")
        self.btn_clear.setObjectName("clearBtn")
        self.btn_clear.clicked.connect(self.history_list.clear)
        sidebar.addWidget(lbl)
        sidebar.addWidget(self.history_list)
        sidebar.addWidget(self.btn_clear)
        self.main_layout.addLayout(sidebar)

    def _setup_main_area(self):
        """Creates search input, results area, and bottom pagination/controls."""
        content = QVBoxLayout()

        nav_top = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search the archives or type /help...")
        self.search_input.returnPressed.connect(self.perform_search)
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.perform_search)
        nav_top.addWidget(self.search_input)
        nav_top.addWidget(self.btn_search)

        self.status = QLabel("Initializing Search Engine...")
        self.status.setStyleSheet("color: #f39c12; font-weight: bold;")

        self.results_area = QTextBrowser()
        self.results_area.setOpenLinks(False)
        self.results_area.anchorClicked.connect(self._handle_link_click)

        self.nav_bottom = QHBoxLayout()
        self.nav_bottom.addStretch(1)
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.setObjectName("navBtn")
        self.btn_prev.clicked.connect(self.go_to_prev_page)
        self.label_page = QLabel("Page 1 of 1")
        self.label_page.setStyleSheet(
            "color:#8ecaf7; font-size:13px; font-weight:bold; margin:0 15px;"
        )
        self.btn_next = QPushButton("Next")
        self.btn_next.setObjectName("navBtn")
        self.btn_next.clicked.connect(self.go_to_next_page)
        self.nav_bottom.addWidget(self.btn_prev)
        self.nav_bottom.addWidget(self.label_page)
        self.nav_bottom.addWidget(self.btn_next)
        self.nav_bottom.addStretch(1)

        content.addLayout(nav_top)
        content.addWidget(self.status)
        content.addWidget(self.results_area)
        content.addLayout(self.nav_bottom)

        self.main_layout.addLayout(content)
        self._update_nav_ui()

    def _on_engine_ready(self, engine):
        """Slot: Called when model/engine is loaded asynchronously."""
        self.engine = engine
        self.search_input.setDisabled(False)
        self.btn_search.setDisabled(False)
        self.status.setText(f"Online | {len(engine.df)} Artworks Indexed")
        self.status.setStyleSheet("color: #2ecc71; font-weight: bold;")

    def _on_engine_error(self, err: str):
        """Slot: Model failed to load."""
        self.status.setText(f"Failed to load engine: {err}")
        self.status.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def _on_history_clicked(self, item):
        """Triggers a search when a query history item is clicked."""
        if item is not None:
            self.search_input.setText(item.text())
            self.perform_search()

    def _handle_link_click(self, qurl: QUrl):
        """Handles anchor link clicks in the result view."""
        url_str = qurl.toString()
        if url_str.startswith("http"):
            webbrowser.open(url_str)
        else:
            self.search_input.setText(url_str)
            self.perform_search()

    def _update_nav_ui(self):
        """Shows/hides and updates navigation controls as needed."""
        visible = self.total_pages > 1
        self.btn_prev.setVisible(visible)
        self.btn_next.setVisible(visible)
        self.label_page.setVisible(visible)
        self.btn_prev.setDisabled(self.current_page <= 1)
        self.btn_next.setDisabled(self.current_page >= self.total_pages)
        self.label_page.setText(f"Page {self.current_page} of {self.total_pages}")

    def go_to_next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._run_search_and_display()

    def go_to_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._run_search_and_display()

    def _get_img_b64(self, url: str) -> Optional[str]:
        """
        Fetches image from url, encodes as base64. Uses a simple memory cache.
        """
        if not url or len(url) < 10:
            return None
        if url in self.image_cache:
            return self.image_cache[url]
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            if r.status_code == 200:
                b64 = base64.b64encode(r.content).decode()
                self.image_cache[url] = b64
                return b64
        except Exception:
            pass
        return None

    def perform_search(self):
        """
        Initiates the search from search bar or search history.
        Handles commands, manages search history.
        """
        query = self.search_input.text().strip()
        if not query or not self.engine:
            return

        # System commands: /help, /exit
        if query.startswith("/"):
            cmd = query.lower()
            if cmd == "/exit":
                QApplication.quit()
                return
            if cmd == "/help":
                html = (
                    "<h2 style='color:white;'>System Guide</h2><hr><ul>"
                    "<li><b>/help</b> - Show this manual.</li>"
                    "<li><b>/exit</b> - Quit the application.</li></ul>"
                )
                self.results_area.setHtml(html)
                self.search_input.clear()
                return

        # Search history logic
        existing = []
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            if item is not None:
                existing.append(item.text())
        if query not in existing:
            self.history_list.insertItem(0, query)

        self.current_query = query
        self.current_page = 1
        self._run_search_and_display()

    def _run_search_and_display(self):
        """
        Begins the search in a background thread, updates loading status.
        """
        if not self.engine or not self.current_query:
            return

        self.btn_search.setDisabled(True)
        self.search_input.setDisabled(True)
        self.status.setText(f"Searching archives for '{self.current_query}'...")
        self.status.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.results_area.setHtml(
            f"<h3 style='color: #888; text-align: center; margin-top: 50px;'>Loading search results for page {self.current_page}...</h3>"
        )

        self.search_start_time = time.perf_counter()
        self.search_worker = SearchWorker(
            self.engine, self.current_query, self.current_page, self.per_page
        )
        self.search_worker.results_ready.connect(self._on_search_complete)
        self.search_worker.error_occurred.connect(self._on_search_error)
        self.search_worker.start()

    def _on_search_complete(self, data: Dict[str, Any]):
        """
        Slot: Fired when background search completes. Unlocks UI, updates pagination and results.
        """
        if self.engine is None:
            self._on_search_error("Search engine is unavailable.")
            return

        duration = time.perf_counter() - self.search_start_time
        self.btn_search.setDisabled(False)
        self.search_input.setDisabled(False)
        self.search_input.setFocus()
        self.status.setText(f"Online | {len(self.engine.df)} Artworks Indexed")
        self.status.setStyleSheet("color: #2ecc71; font-weight: bold;")

        self.total_pages = data.get("total_pages", 1)
        self._update_nav_ui()
        scroll_bar = self.results_area.verticalScrollBar()
        if scroll_bar is not None:
            scroll_bar.setValue(0)

        self.display_results(
            data.get("results", []),
            self.current_query,
            duration,
            data.get("suggestion"),
        )

    def _on_search_error(self, err: str):
        """
        Slot: Fired when hybrid search fails (exception in thread).
        """
        self.btn_search.setDisabled(False)
        self.search_input.setDisabled(False)
        self.status.setText(f"Search Failed: {err}")
        self.status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.results_area.setHtml(
            f"<h3 style='color:#e74c3c;'>An error occurred during retrieval.</h3><p>{err}</p>"
        )

    def display_results(self, results, query, duration, suggestion):
        """
        Simple HTML card view for paginated Tate results.
        """
        sug_html = ""
        if suggestion:
            # Allow user to click suggestion to search or right-click to copy
            sug_html = (
                f"<div style='color:#f39c12; margin-bottom:10px;'>Did you mean: "
                f"<a href='{suggestion}' style='color:#3498db; text-decoration:none;'><b>{suggestion}</b></a>?"
                f"</div>"
            )
        html = f"{sug_html}<h2 style='color:white;'>Results: '{query}' ({duration:.3f}s)</h2><hr>"

        if not results:
            html += "<h3 style='color:#888;'>No matches found.</h3>"

        for res in results:
            thumb = res["Thumbnail"]
            b64 = self._get_img_b64(thumb)
            img_tag = (
                f"<a href='{thumb}'><img src='data:image/jpeg;base64,{b64}' width='140' style='border-radius:4px;'/></a>"
                if b64
                else "<div style='width:140px; height:100px; background:#222; border-radius:4px; text-align:center; padding-top:40px; color:#555;'>No Image</div>"
            )

            row = f"<table width='100%' style='margin-bottom:20px;'><tr><td width='160' valign='top'>{img_tag}</td><td valign='top'>"
            row += (
                f"<div style='margin-bottom:2px;'><a href='{thumb}' style='color:#3498db; text-decoration:none; font-size:18px; font-weight:bold;'>{res['Title']}</a> "
                f"<span style='color:#7f8c8d; font-size:14px;'>({res['Year']})</span></div>"
            )
            row += f"<div style='color:#ecf0f1; font-size:15px; font-weight:bold; margin-bottom:4px;'>{res['Artist']}</div>"
            row += f"<div style='color:#f1c40f; font-size:13px; font-style:italic; margin-bottom:6px;'>{res['Medium']}</div>"
            row += f"<div style='color:#cccccc; font-size:13px; margin-bottom:6px;'>{res['Description']}</div>"
            meta = [res["Dimensions"]] if res["Dimensions"] else []
            if res["CreditLine"]:
                meta.append(f"Credit: {res['CreditLine']}")
            if meta:
                row += f"<div style='color:#7f8c8d; font-size:11px; margin-bottom:8px;'>{' | '.join(meta)}</div>"

            row += f"<div style='color:#2ecc71; font-size:11px;'><b>Score:</b> {res['Score']} &nbsp;|&nbsp; <b>Rank:</b> {res['Rank']}</div>"
            row += (
                "</td></tr></table><hr style='border:none; border-top:1px solid #222;'>"
            )
            html += row

        self.results_area.setHtml(html)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ArtSearchGUI()
    window.show()
    sys.exit(app.exec())
