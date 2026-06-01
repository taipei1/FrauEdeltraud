@echo off
cd /d "%~dp0"
echo === FrauEdeltraud: полный прогон автотестов ===
echo.
echo Использование:
echo   run_tests.bat                    - все тесты
echo   run_tests.bat unit               - только быстрые
echo   run_tests.bat integration        - только интеграция
echo   run_tests.bat "not slow"         - все кроме долгих
echo.
python -m pytest tests/ -v --tb=short -m %*
exit /b %ERRORLEVEL%
