@echo off
REM Fresh-ish PATH then MVP build (avoids "input line is too long").
REM Keep Git (+ GNU patch from Git usr\bin) on PATH — setuptools-scm,
REM CMake FetchContent, and vllm_flash_attn PATCH_COMMAND need them.
set "GIT_HOME="
if exist "%LOCALAPPDATA%\hermes\git\cmd\git.exe" set "GIT_HOME=%LOCALAPPDATA%\hermes\git"
if not defined GIT_HOME if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT_HOME=%ProgramFiles%\Git"
if not defined GIT_HOME if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "GIT_HOME=%LOCALAPPDATA%\Programs\Git"
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0;%GIT_HOME%\cmd;%GIT_HOME%\usr\bin;C:\Python312;C:\Python312\Scripts"
cd /d "%~dp0..\.."
if not defined BUILD_MVP_LOG set "BUILD_MVP_LOG=%~dp0..\..\.superpowers\sdd\task-8-build.log"
set "LOG=%BUILD_MVP_LOG%"
if exist "%LOG%" del /f /q "%LOG%" 2>nul
call "%~dp0build_mvp.cmd" > "%LOG%" 2>&1
set ERR=%ERRORLEVEL%
echo EXIT=%ERR%>> "%LOG%"
exit /b %ERR%
