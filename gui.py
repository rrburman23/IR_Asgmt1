"""
File Name: gui.py
Description: PyQt6 GUI for Art Search.
- Layout: Clean Div-based Hierarchy (Name -> Artist -> Medium -> Description).
"""

import sys
import time
import base64
import requests
from typing import List

# pylint: disable=no-name-in-module
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
from PyQt6.QtGui import QDesktopServices, QIcon


class EngineLoadThread(QThread):
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


class ArtSearchGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")
        self.setWindowIcon(QIcon("icon.png"))
        self.setStyleSheet("background-color: #0f0f0f; color: #ccc;")
        self.setMinimumSize(900, 600)
        self.resize(1150, 800)
        self.engine = None
        self.image_cache: dict[str, str] = {}
        self.apply_styling()
        self.setWindowIcon(QIcon("icon.png"))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_sidebar()
        self._setup_main_area()

        self.search_input.setDisabled(True)
        self.btn_search.setDisabled(True)

        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.error_occurred.connect(self._on_engine_error)
        self.engine_thread.start()

    def apply_styling(self):
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
            QPushButton#clearBtn { 
                background: #222; color: #888; font-size: 11px; margin-top: 5px;
            }
            QPushButton#clearBtn:hover { background: #e74c3c; color: white; }
            QTextBrowser { background: #141414; border: 1px solid #222; border-radius: 8px; }
            QListWidget { background: #1a1a1a; border: none; border-radius: 8px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { background: #2980b9; color: white; }
        """)

    def _setup_sidebar(self):
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
        self.btn_clear.clicked.connect(self.clear_history)

        sidebar.addWidget(lbl)
        sidebar.addWidget(self.history_list)
        sidebar.addWidget(self.btn_clear)
        self.main_layout.addLayout(sidebar)

    def _setup_main_area(self):
        content = QVBoxLayout()
        nav = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search the archives or type /help...")
        self.search_input.returnPressed.connect(self.perform_search)
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.perform_search)

        self.status = QLabel("Initializing Search Engine...")
        self.status.setStyleSheet("color: #f39c12;")
        self.results_area = QTextBrowser()
        self.results_area.setOpenLinks(False)
        self.results_area.anchorClicked.connect(self._on_suggestion_clicked)

        nav.addWidget(self.search_input)
        nav.addWidget(self.btn_search)
        content.addLayout(nav)
        content.addWidget(self.status)
        content.addWidget(self.results_area)
        self.main_layout.addLayout(content)

    def _on_engine_ready(self, engine):
        self.engine = engine
        self.search_input.setDisabled(False)
        self.btn_search.setDisabled(False)
        self.status.setText(f"Online | {len(engine.df)} Artworks Indexed")
        self.status.setStyleSheet("color: #2ecc71;")

    def _on_engine_error(self, err: str):
        self.status.setText(f"Failed: {err}")
        self.status.setStyleSheet("color: #e74c3c;")

    def _on_history_clicked(self, item):
        if item is not None:
            self.search_input.setText(item.text())
            self.perform_search()

    def _on_suggestion_clicked(self, qurl: QUrl):
        url_str = qurl.toString()
        if url_str.startswith("img::"):
            # Open thumbnail in browser
            image_url = url_str[5:]  # strip the "img::" prefix
            QDesktopServices.openUrl(QUrl(image_url))
        else:
            # Spelling suggestion — re-run search
            self.search_input.setText(url_str)
            self.perform_search()

    def clear_history(self):
        self.history_list.clear()

    def _get_img_b64(self, url: str) -> str | None:
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

    def _display_help(self):
        html = "<h2 style='color:white;'>System Guide</h2><hr>"
        html += (
            "<p style='color:#3498db; font-size:16px;'>Available Slash Commands:</p>"
        )
        html += "<ul>"
        html += "<li style='margin-bottom:8px;'><b>/help</b> - Show this manual.</li>"
        html += (
            "<li style='margin-bottom:8px;'><b>/exit</b> - Quit the application.</li>"
        )
        html += "<li style='margin-bottom:8px;'><b>/test</b> - Run unit tests (terminal).</li>"
        html += "<li style='margin-bottom:8px;'><b>/evaluate</b> - Run performance eval (terminal).</li>"
        html += "</ul>"
        self.results_area.setHtml(html)

    def perform_search(self):
        query = self.search_input.text().strip()
        if not query or not self.engine:
            return

        if query.startswith("/"):
            cmd = query.lower()
            if cmd == "/exit":
                QApplication.quit()
                return
            if cmd == "/help":
                self._display_help()
                self.search_input.clear()
                return
            if cmd in ["/test", "/evaluate", "/eval"]:
                self.status.setText("Running System Task... Check Terminal Console")
                self.status.setStyleSheet("color: #3498db;")
                QApplication.processEvents()
                main_mod = sys.modules.get("__main__")
                if main_mod and hasattr(main_mod, "launch_tests"):
                    if cmd == "/test":
                        main_mod.launch_tests()  # type: ignore
                    else:
                        main_mod.launch_evaluation()  # type: ignore
                self.status.setText("Task Completed. Engine Ready.")
                self.status.setStyleSheet("color: #2ecc71;")
                self.search_input.clear()
                return

            self.results_area.setHtml(
                f"<div style='color:#e74c3c;'>[ERROR] Unknown command: {query}. Valid commands are /help, /exit, /test, /evaluate.</div>"
            )
            return

        existing: List[str] = []
        for i in range(self.history_list.count()):
            hist_item = self.history_list.item(i)
            if hist_item is not None:
                existing.append(hist_item.text())

        if query not in existing:
            self.history_list.insertItem(0, query)

        start = time.perf_counter()
        results = self.engine.hybrid_search(query)
        duration = time.perf_counter() - start
        self.display_results(results, query, duration)

    def display_results(self, results: list, query: str, duration: float):
        suggestion_html = ""
        if results and results[0].get("Suggestion"):
            sug = results[0]["Suggestion"]
            suggestion_html = (
                f"<div style='color:#f39c12; margin-bottom:10px;'>"
                f"Did you mean: <a href='{sug}' style='color:#3498db; text-decoration:none;'><b>{sug}</b></a>?</div>"
            )

        html = f"{suggestion_html}<h2 style='color:white;'>Results: '{query}' ({duration:.3f}s)</h2><hr>"

        for res in results:
            thumb_url = res["Thumbnail"]
            b64 = self._get_img_b64(thumb_url)
            if b64:
                img_html = f"<a href='img::{thumb_url}'><img src='data:image/jpeg;base64,{b64}' width='140' style='border-radius:4px; cursor:pointer;'/></a>"
            else:
                img_html = "<div style='width:140px; height:100px; background:#222; border-radius:4px; text-align:center; padding-top:40px; color:#555;'>No Image</div>"

            # DIV-BASED HIERARCHY (Clean Spacing)
            row = "<table width='100%' style='margin-bottom:20px;'><tr>"
            row += f"<td width='160' valign='top'>{img_html}</td>"
            row += "<td valign='top'>"

            # Title & Year
            year_str = (
                f" ({res['Year']})"
                if res["Year"] and res["Year"] != "Unknown Date"
                else ""
            )

            row += f"<div style='font-size:18px; font-weight:bold; margin-bottom:2px;'><a href='img::{thumb_url}' style='color:#3498db; text-decoration:none;'>{res['Title']}</a><span style='color:#7f8c8d; font-size:14px; font-weight:normal;'>{year_str}</span></div>"

            # Artist
            row += f"<div style='color:#ecf0f1; font-size:15px; font-weight:bold; margin-bottom:4px;'>{res['Artist']}</div>"

            # Medium
            row += f"<div style='color:#f1c40f; font-size:13px; font-style:italic; margin-bottom:6px;'>{res['Medium']}</div>"

            # Description
            row += f"<div style='color:#cccccc; font-size:13px; margin-bottom:6px;'>{res['Description']}</div>"

            # Dimensions & Credit
            meta = []
            if res.get("Dimensions") and "Unavailable" not in res["Dimensions"]:
                meta.append(res["Dimensions"])
            if res.get("CreditLine"):
                meta.append(f"Credit: {res['CreditLine']}")
            if meta:
                row += f"<div style='color:#7f8c8d; font-size:11px; margin-bottom:8px;'>{' | '.join(meta)}</div>"

            # Match Logic
            row += f"<div style='color:#2ecc71; font-size:11px;'><b>Match:</b> {res['Reasons']} &nbsp;|&nbsp; <b>RRF Score:</b> {res['Score']}</div>"
            row += (
                "</td></tr></table><hr style='border:none; border-top:1px solid #222;'>"
            )
            html += row

        self.results_area.setHtml(html)
