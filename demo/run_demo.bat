@echo off
REM ==============================================================================
REM VFI 80% Milestone Review Demo Execution Script (Windows)
REM ==============================================================================

cd /d "%~dp0\.."
node demo\run_demo.js %*
