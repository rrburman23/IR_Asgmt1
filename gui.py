"""
File Name: gui.py
Description: Provides the PyQt6 graphical user interface for the Tate Gallery
             Search Engine. Implements background loading, memory caching for
             images, URL patching, and persistent search history management.
"""

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
    """
    Initializes the AI search engine on a separate background thread.
    This prevents the main PyQt6 GUI window from freezing or showing
    "Not Responding" during the model loading or embedding computation.
    """

    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        """
        Standard QThread execution method. Initializes the engine and
        emits the finished object back to the main thread.
        """
        try:
            engine = ArtGallerySearchEngine(OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:  # pylint: disable=broad-except
            self.error_occurred.emit(str(e))


# ==========================================
# 2. Main GUI Application
# ==========================================
class ArtSearchGUI(QMainWindow):
    """
    The main window class. Handles the layout, user input, search history,
    and rendering of the hybrid search results into HTML.
    """

    def __init__(self):
        """
        Initializes the GUI, sets up the multi-pane layout, and fires
        off the background thread to load the AI models.
        """
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")
        self.resize(1150, 800)

        # Internal State Management
        self.engine = None
        self.image_cache = {}

        self.apply_modern_styling()

        # Setup the core horizontal layout (Sidebar on left, Main on right)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_sidebar()
        self._setup_main_area()

        # Fire off the background engine loader
        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.start()

    def apply_modern_styling(self):
        """
        Defines the 'Dark Mode' aesthetic using Qt Style Sheets (QSS).
        """
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
            
            /* Specific styling for the Clear History button */
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
        """
        Initializes the left-hand sidebar containing the interactive
        search history list and the clear button.
        """
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 10, 0)

        # Sidebar Header
        lbl = QLabel("SEARCH HISTORY")
        lbl.setStyleSheet(
            "font-weight: bold; color: #555; font-size: 11px; margin-bottom: 5px;"
        )

        # History List Widget
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setFixedWidth(220)

        # Clear History Button
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.setObjectName("clearBtn")
        self.clear_history_btn.clicked.connect(self.clear_history)

        # Assemble the sidebar
        sidebar.addWidget(lbl)
        sidebar.addWidget(self.history_list)
        sidebar.addWidget(self.clear_history_btn)
        self.main_layout.addLayout(sidebar)

    def _setup_main_area(self):
        """
        Initializes the right-hand area for user input (search bar)
        and the rich text browser for rendering results.
        """
        content = QVBoxLayout()
        nav = QHBoxLayout()

        # Search Input Field
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search the collection (e.g., 'ships in a storm')..."
        )
        self.search_input.returnPressed.connect(self.perform_search)

        # Search Button
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.perform_search)

        nav.addWidget(self.search_input)
        nav.addWidget(btn_search)

        # Status Label (Displays loading/hardware info)
        self.status = QLabel("Initializing...")
        self.status.setStyleSheet("color: #f39c12;")

        # Results Display Area
        self.results_area = QTextBrowser()

        # Assemble the main area
        content.addLayout(nav)
        content.addWidget(self.status)
        content.addWidget(self.results_area)
        self.main_layout.addLayout(content)

    def _on_engine_ready(self, engine):
        """
        Slot triggered when the background thread finishes loading the AI model.
        """
        self.engine = engine
        self.status.setText("Engine Online - Ready for Queries!")
        self.status.setStyleSheet("color: #2ecc71;")

    def _on_history_clicked(self, item):
        """
        Allows users to re-run previous searches by clicking them in the sidebar.
        """
        self.search_input.setText(item.text())
        self.perform_search()

    def clear_history(self):
        """
        Wipes the history list widget clean.
        """
        self.history_list.clear()

    def _get_img_b64(self, url: str, title: str):
        """
        Fetches an image from a URL and converts it to a Base64 string.
        Includes error handling and dynamically generates a placeholder
        image using an API if the original Tate link is completely dead.
        """
        # Return immediately if we've already fetched it this session
        if url in self.image_cache:
            return self.image_cache[url]

        b64 = None
        try:
            # Request image with a standard browser User-Agent to bypass blocks
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
            if r.status_code == 200:
                b64 = base64.b64encode(r.content).decode()
        except Exception:  # pylint: disable=broad-except
            b64 = None

        # Fallback: Generate a placeholder image using placehold.co
        if not b64:
            try:
                safe_t = urllib.parse.quote(title[:15])
                f_url = f"https://placehold.co/140x100/2c3e50/ecf0f1?text={safe_t}"
                r = requests.get(f_url, timeout=3)
                b64 = base64.b64encode(r.content).decode()
            except Exception:  # pylint: disable=broad-except
                b64 = None

        # Cache the result (even if it's just a placeholder) to save bandwidth
        self.image_cache[url] = b64
        return b64

    def perform_search(self):
        """
        Main search execution logic. Grabs input, manages the history list,
        calls the hybrid engine, and passes results to the display function.
        """
        query = self.search_input.text().strip()
        if not query or self.engine is None:
            return

        # ==========================================
        # Intercept Slash Commands
        # ==========================================
        if query.lower() in ["/test", "test"]:
            self.status.setText("Running Unit Tests... Check your terminal!")
            self.status.setStyleSheet("color: #3498db;")  # Blue for system task
            QApplication.processEvents()  # Force the GUI to update the text immediately

            # Import and run the test router from main.py
            # pylint: disable=import-outside-toplevel
            from main import launch_tests

            launch_tests()

            self.status.setText("Tests Completed. Engine Online.")
            self.status.setStyleSheet("color: #2ecc71;")
            self.search_input.clear()
            return  # Stop here so we don't actually search the art database

        if query.lower() in ["/evaluate", "/eval", "evaluate"]:
            self.status.setText("Running Evaluation Suite... Check your terminal!")
            self.status.setStyleSheet("color: #3498db;")
            QApplication.processEvents()

            # pylint: disable=import-outside-toplevel
            from main import launch_evaluation

            launch_evaluation()

            self.status.setText("Evaluation Completed. Engine Online.")
            self.status.setStyleSheet("color: #2ecc71;")
            self.search_input.clear()
            return
        # ==========================================

        # Manage Search History Sidebar
        # Check if the query is already in history to avoid duplicates
        existing = [
            self.history_list.item(i).text()  # type: ignore
            for i in range(self.history_list.count())
        ]
        if query not in existing:
            self.history_list.insertItem(0, query)

        # Execute the Hybrid Search
        results = self.engine.hybrid_search(query)
        self.display_results(results, query)
        self.status.setText(f"Completed. Found {len(results)} matches.")

    def display_results(self, results: list, query: str):
        """
        Converts the raw dictionary results list into formatted HTML tables
        and renders them in the QTextBrowser.
        """
        html = (
            f"<h2 style='color:white; margin-bottom:10px;'>Results: '{query}'</h2><hr>"
        )

        for res in results:
            # Convert URL to Base64 (either from web or session cache)
            b64 = self._get_img_b64(res["Thumbnail"], res["Title"])

            # Create the image tag
            img_html = (
                f"<img src='data:image/jpeg;base64,{b64}' width='140' style='border-radius:4px;'/>"
                if b64
                else "[No Image]"
            )

            # Construct the HTML table row for this artwork
            html += (
                "<table width='100%' style='margin-bottom:15px'><tr>"
                + f"<td width='160' valign='top'>{img_html}</td>"
                + "<td valign='top'>"
                + f"<b style='color:#3498db; font-size:18px'>{res['Rank']}. {res['Title']}</b><br>"
                + f"<span style='color:#ecf0f1; font-size:14px'>Artist: {res['Artist']}</span><br>"
                + "<i style='color:#666; font-size:13px; margin-top:5px;'>"
                + f"{res['Description']}</i><br>"
                + "<span style='color:#444; font-size:11px;'>Relevance Score: "
                + f"{res['Score']}</span></td>"
                + "</tr></table><hr style='border:none; border-top:1px solid #222;'>"
            )

        # Push the finalized HTML to the screen
        self.results_area.setHtml(html)
