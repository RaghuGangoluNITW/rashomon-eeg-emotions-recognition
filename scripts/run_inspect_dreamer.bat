@echo off
REM Runs the DREAMER inspection script and writes output to repo root
"%~dp0\..\C:\Users\rgangolu\AppData\Local\Programs\Python\Python311\python.exe" "%~dp0inspect_dreamer_fields.py" > "%~dp0\..\inspect_dreamer_output.txt" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Inspector exited with error. See scripts\inspect_dreamer_output.txt for details.
) else (
    echo Inspector completed. Output written to inspect_dreamer_output.txt
)