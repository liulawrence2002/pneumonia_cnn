@echo off
echo ========================================
echo Pneumonia CNN - Virtual Environment Setup
echo ========================================
echo.

:: Check if Python 3.11 is available
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 not found!
    echo.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Found Python 3.11!
echo.

:: Create virtual environment with Python 3.11
echo Creating virtual environment...
py -3.11 -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully!
echo.

:: Activate and upgrade pip
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run 'install_dependencies.bat' to install packages
echo 2. Or manually run: pip install -r requirements.txt
echo.
pause
