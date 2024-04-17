import sys
from cx_Freeze import setup, Executable

sys.setrecursionlimit(10000)  # Set the recursion limit to a higher value

base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [Executable("sentiment_analysis_app.py", base=base)]

setup(
    name="SentimentAnalysisApp",
    version="1.0",
    description="Your description",
    executables=executables,
)
