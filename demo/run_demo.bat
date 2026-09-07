@echo off
REM ==============================================================================
REM VFI Phase 3 Review Demo Execution Script (Windows)
REM ==============================================================================

cd /d "%~dp0\.."
node demo\run_demo.js %*
