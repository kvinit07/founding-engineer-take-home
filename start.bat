@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 bootstrap.py %*
    goto done
)
where python >nul 2>nul
if not errorlevel 1 (
    python bootstrap.py %*
    goto done
)
echo Install Python 3.11 or later, then run this launcher again.
:done
if errorlevel 1 pause
