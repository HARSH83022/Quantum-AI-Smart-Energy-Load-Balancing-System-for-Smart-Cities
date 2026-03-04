"""
Simple tests that don't require heavy dependencies
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_python_version():
    """Test Python version compatibility"""
    assert sys.version_info >= (3, 11)


def test_imports():
    """Test basic imports work"""
    try:
        import fastapi
        import sqlalchemy
        import numpy
        import pandas
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_project_structure():
    """Test project structure exists"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Check main directories exist
    assert os.path.exists(os.path.join(project_root, 'src'))
    assert os.path.exists(os.path.join(project_root, 'tests'))
    assert os.path.exists(os.path.join(project_root, 'requirements.txt'))
    assert os.path.exists(os.path.join(project_root, 'README.md'))


def test_main_module():
    """Test main module can be imported"""
    try:
        from src.main import app
        assert app is not None
        assert hasattr(app, 'title')
        assert app.title == "Quantum-AI Smart Energy Load Balancing System"
    except ImportError as e:
        pytest.fail(f"Cannot import main module: {e}")


def test_api_routes():
    """Test API routes module can be imported"""
    try:
        from src.api.routes import router
        assert router is not None
    except ImportError as e:
        pytest.fail(f"Cannot import API routes: {e}")


def test_database_models():
    """Test database models can be imported"""
    try:
        from src.database import models
        assert models is not None
    except ImportError as e:
        pytest.fail(f"Cannot import database models: {e}")


if __name__ == "__main__":
    pytest.main([__file__])