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


def test_src_structure():
    """Test src directory structure"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    src_dir = os.path.join(project_root, 'src')
    
    # Check main src modules exist
    assert os.path.exists(os.path.join(src_dir, 'main.py'))
    assert os.path.exists(os.path.join(src_dir, 'api'))
    assert os.path.exists(os.path.join(src_dir, 'database'))
    assert os.path.exists(os.path.join(src_dir, 'data_sources'))


def test_api_structure():
    """Test API directory structure"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    api_dir = os.path.join(project_root, 'src', 'api')
    
    # Check API files exist
    assert os.path.exists(os.path.join(api_dir, 'routes.py'))
    assert os.path.exists(os.path.join(api_dir, '__init__.py'))


def test_requirements_file():
    """Test requirements.txt has expected content"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    req_file = os.path.join(project_root, 'requirements.txt')
    
    with open(req_file, 'r') as f:
        content = f.read()
        
    # Check for key dependencies
    assert 'fastapi' in content
    assert 'sqlalchemy' in content
    assert 'numpy' in content
    assert 'pandas' in content


if __name__ == "__main__":
    pytest.main([__file__])