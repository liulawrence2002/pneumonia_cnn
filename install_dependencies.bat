@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.
echo Installation complete!
echo.
pause
