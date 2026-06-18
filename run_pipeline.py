"""
Entry point — runs the full pipeline from project root.
Usage: python run_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pricing_engine import run_pipeline

if __name__ == "__main__":
    run_pipeline()
    print("\nDone! Now launch the dashboard:")
    print("  streamlit run dashboard/app.py")
