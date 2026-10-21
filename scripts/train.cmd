@echo off
setlocal
where conda >nul 2>&1
if errorlevel 1 (
    echo Conda was not found. Run this script from an Anaconda Prompt or an activated Conda terminal.
    exit /b 1
)
call conda run --no-capture-output -n python_31208 python -u "%~dp0train.py" %*
exit /b %errorlevel%
