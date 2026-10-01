@echo off
setlocal enabledelayedexpansion
rem ============================================================
rem  train_track.bat  --  the confirmed v26k recipe, end to end,
rem  for any circuit already in Neural_Network_NEAT-master/new/
rem  f1tenth_racetracks-main/<CIRCUIT>/.
rem
rem  Usage:
rem    train_track.bat CIRCUIT [OUT_NAME] [EXTRA_FLAGS]
rem
rem    train_track.bat --attach CIRCUIT OUT_NAME
rem
rem  --attach: the kernel is already running on Kaggle (this window/terminal
rem  died, or was closed, after step 2). Skips the parity check and the push
rem  and picks up at step 3 -- polling, then fetch, trace, deploy prompt,
rem  ghost prompt, filing. OUT_NAME must be the one the run was pushed with
rem  (the Kaggle kernel name is derived from it). Safe to run while the
rem  kernel is still RUNNING or already COMPLETE.
rem
rem  Example:
rem    train_track.bat Spa
rem    train_track.bat Monza Monza_v26k_retry
rem    train_track.bat Monza Monza_v26k_startwidened "--ddqn --line-k 0 --stop-after-stale 600 --eval-backlog 1 --n-step 6 --start-widened"
rem
rem  OUT_NAME defaults to <CIRCUIT>_v26k. Pass your own to avoid
rem  clobbering a previous attempt's checkpoint under the same name.
rem
rem  EXTRA_FLAGS, if given, REPLACES kernel_run.py's own default recipe
rem  flags entirely rather than adding to them (same as kaggle_push.sh's own
rem  EXTRA_FLAGS argument, which this passes straight through) -- so a one-
rem  off experiment needs to repeat the recipe flags it still wants, same as
rem  in the example above. Omit for the standard recipe.
rem
rem  What it does, in order:
rem    1. A quick CPU/GPU parity check -- refuses to spend Kaggle
rem       GPU quota on a broken local edit.
rem    2. Pushes the training kernel (kaggle_push.sh) with no extra
rem       flags: config.py's RL_EDGE_MARGIN=0 / off-track-cost ramp /
rem       lap-bonus settings and rlpolicy.py's 42-action table (3
rem       brake levels + half throttle) ARE the v26k recipe now, and
rem       kernel_run.py's own default EXTRA flags (--ddqn --line-k 0
rem       --stop-after-stale 600) are unchanged from what produced
rem       Spa's 121.30 s. Nothing circuit-specific to set.
rem    3. Polls Kaggle every 2 minutes until the run finishes on its
rem       own (--stop-after-stale stops it once it has genuinely
rem       plateaued -- this can take anywhere from ~30 min to a
rem       couple of hours; the window can be left alone meanwhile).
rem    4. Fetches the checkpoint (kaggle_fetch.sh).
rem    5. Traces it (tools/trace_rl.py) and reports the flying lap
rem       time, saving a PNG next to this script.
rem    6. Asks before touching the deployed assets/policies/
rem       <CIRCUIT>.npz -- never overwrites it silently. If it was
rem       replaced, then asks whether to re-record that circuit's
rem       qualifying ghost (tools/record_ghost.py). Either way,
rem       the raw checkpoint is filed under ai_sw/policy/<CIRCUIT>/
rem       and the Kaggle log under ai_sw/logs/<CIRCUIT>/, matching
rem       how every prior run this session was organised.
rem ============================================================

set ATTACH=
if /I "%~1"=="--attach" (
  set ATTACH=1
  shift
)

if "%~1"=="" (
  echo Usage: train_track.bat [--attach] CIRCUIT [OUT_NAME] [EXTRA_FLAGS]
  echo Example: train_track.bat Monza
  echo          train_track.bat --attach Melbourne Melbourne_v26_best
  exit /b 1
)

set CIRCUIT=%~1
set OUT_NAME=%~2
if "%OUT_NAME%"=="" set OUT_NAME=%CIRCUIT%_v26k
set EXTRA_FLAGS=%~3

set KUSER=mskimdev
set REPO=C:\Users\user\Desktop\WorkSpace\Dongari\STEM2026
set PY=%REPO%\.venv312\Scripts\python.exe
set POLICIES=%REPO%\ai_sw\assets\policies
set POLICY_DIR=%REPO%\ai_sw\policy\%CIRCUIT%
set LOG_DIR=%REPO%\ai_sw\logs\%CIRCUIT%
set DEPLOYED=

if not exist "%POLICY_DIR%" mkdir "%POLICY_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if defined ATTACH (
  echo.
  echo === attach mode: skipping parity check and push; joining the already-
  echo === running kernel aisw-%OUT_NAME% ^(same OUT_NAME the run was pushed with^)
  goto :step3
)

echo.
echo === [1/6] CPU/GPU parity check ===
"%PY%" "%REPO%\ai_sw_gpu\tests\parity.py" --mode wall
if errorlevel 1 (
  echo.
  echo Parity FAILED -- something in game/ or gpuenv/ disagrees between
  echo the CPU and GPU environments. Fix that before spending GPU quota.
  exit /b 1
)
"%PY%" "%REPO%\ai_sw_gpu\tests\parity.py" --mode random
if errorlevel 1 (
  echo.
  echo Parity FAILED on random-action mode -- see above.
  exit /b 1
)
echo Parity OK.

echo.
echo === [2/6] pushing the Kaggle kernel (circuit=%CIRCUIT%, out=%OUT_NAME%) ===
wsl -d kali-linux bash -c "export PATH=\"$HOME/.local/bin:/usr/bin:/bin\"; cd /mnt/c/Users/user/Desktop/WorkSpace/Dongari/STEM2026/ai_sw_gpu/kaggle && bash kaggle_push.sh %KUSER% %CIRCUIT% %OUT_NAME% '' '%EXTRA_FLAGS%'"
if errorlevel 1 (
  echo.
  echo Kernel push failed -- see the output above ^(a stale dataset,
  echo an expired kaggle.json, or a name collision are the usual causes^).
  exit /b 1
)

:step3
echo.
echo === [3/6] waiting for training to finish ===
echo ^(polling every 2 minutes; this window can be left alone^)
:pollloop
wsl -d kali-linux bash -c "export PATH=\"$HOME/.local/bin:/usr/bin:/bin\"; kaggle kernels status %KUSER%/aisw-$(echo %OUT_NAME% | tr '[:upper:]_' '[:lower:]-')" > "%TEMP%\aisw_kstatus.txt" 2>&1
set /p KSTATUS=<"%TEMP%\aisw_kstatus.txt"
echo   !KSTATUS!
echo !KSTATUS! | findstr /C:"COMPLETE" >nul
if not errorlevel 1 goto polldone
rem A session stopped by hand (Kaggle "Stop Session") ends as
rem CANCEL_ACKNOWLEDGED, not COMPLETE; treat it as finished so the checkpoint
rem it saved before being stopped still gets fetched and traced.
echo !KSTATUS! | findstr /C:"CANCEL_ACKNOWLEDGED" >nul
if not errorlevel 1 (
  echo   Kernel was cancelled -- fetching whatever it saved before that.
  goto polldone
)
echo !KSTATUS! | findstr /C:"ERROR" >nul
if not errorlevel 1 (
  echo.
  echo Kernel ERRORED. Check it on kaggle.com or via kaggle_watch.sh
  echo before retrying.
  exit /b 1
)
timeout /t 120 /nobreak >nul
goto pollloop
:polldone
echo Training finished.

echo.
echo === [4/6] fetching the checkpoint ===
wsl -d kali-linux bash -c "export PATH=\"$HOME/.local/bin:/usr/bin:/bin\"; cd /mnt/c/Users/user/Desktop/WorkSpace/Dongari/STEM2026/ai_sw_gpu/kaggle && bash kaggle_fetch.sh %KUSER%/aisw-$(echo %OUT_NAME% | tr '[:upper:]_' '[:lower:]-') %OUT_NAME%"
if not exist "%POLICIES%\%OUT_NAME%_best.npz" (
  echo.
  echo Fetch did not produce %OUT_NAME%_best.npz -- check the kaggle_fetch.sh
  echo output above.
  exit /b 1
)

echo.
echo === [5/6] tracing ===
set TRACE_PNG=%REPO%\%OUT_NAME%_trace.png
set DEPLOY_SRC=%OUT_NAME%_best.npz
set LAPTIME_BEST=%TEMP%\aisw_laptime_best.txt
set LAPTIME_WIDE=%TEMP%\aisw_laptime_wide.txt
set LAPTIME_LAP=%TEMP%\aisw_laptime_bestlap.txt
del /Q "%LAPTIME_BEST%" "%LAPTIME_WIDE%" "%LAPTIME_LAP%" >nul 2>nul
pushd "%REPO%\ai_sw"
"%PY%" -m tools.trace_rl --circuit %CIRCUIT% --policy "%POLICIES%\%OUT_NAME%_best.npz" --out "%TRACE_PNG%" --lap-time-file "%LAPTIME_BEST%"
rem _bestwide.npz only exists once episode widening triggered mid-run (see
rem LONG_EPISODE_RATIO in rlenv.py): it tracks the best tier<=1 (not
rem necessarily fully clean) result seen after widening, separately from
rem _best.npz's strict tier==0 ratchet, since a run can easily stale-stop
rem still holding a pre-widening PERFECT that nothing weaker could ever
rem unseat even as the policy kept improving. Always worth comparing both.
if exist "%POLICIES%\%OUT_NAME%_bestwide.npz" (
  echo   ^(also found %OUT_NAME%_bestwide.npz -- tracing it too^)
  "%PY%" -m tools.trace_rl --circuit %CIRCUIT% --policy "%POLICIES%\%OUT_NAME%_bestwide.npz" --out "%REPO%\%OUT_NAME%_bestwide_trace.png" --lap-time-file "%LAPTIME_WIDE%"
  echo   Compare the two lap times printed above before choosing which to deploy.
)
rem _bestlap.npz: the weights behind the fastest strictly-clean lap any eval
rem drove, whatever its tier (see train_iqn_gpu.py). Only trainer builds from
rem 2026-09-20 on write it.
if exist "%POLICIES%\%OUT_NAME%_bestlap.npz" (
  echo   ^(also found %OUT_NAME%_bestlap.npz -- tracing it too^)
  "%PY%" -m tools.trace_rl --circuit %CIRCUIT% --policy "%POLICIES%\%OUT_NAME%_bestlap.npz" --out "%REPO%\%OUT_NAME%_bestlap_trace.png" --lap-time-file "%LAPTIME_LAP%"
)
popd
echo Trace saved to %TRACE_PNG%

echo.
echo === [6/6] deploy? ===
set HAVE_WIDE=
if exist "%POLICIES%\%OUT_NAME%_bestwide.npz" set HAVE_WIDE=1
set HAVE_LAP=
if exist "%POLICIES%\%OUT_NAME%_bestlap.npz" set HAVE_LAP=1
echo   [1] %OUT_NAME%_best.npz  (default)
if defined HAVE_WIDE echo   [2] %OUT_NAME%_bestwide.npz
if defined HAVE_LAP echo   [3] %OUT_NAME%_bestlap.npz  ^(fastest clean lap seen in any eval^)
echo   [N] skip -- leave %CIRCUIT%.npz untouched
:choosedeploy
rem A typo here used to be taken as a literal filename and handed straight
rem to `copy` -- silently failing (source not found) while everything past
rem it, including the RECORDS.md update, ran anyway as if it had deployed.
rem Numbered choices with re-prompt-on-anything-else close that off.
set DEPLOY_PICK=
set /p DEPLOY_PICK="Pick [1/2/3/N, Enter=1]: "
if "%DEPLOY_PICK%"=="" set DEPLOY_PICK=1
if /I "%DEPLOY_PICK%"=="1" (
  set DEPLOY_SRC=%OUT_NAME%_best.npz
  goto :deploypicked
)
if /I "%DEPLOY_PICK%"=="2" (
  if defined HAVE_WIDE (
    set DEPLOY_SRC=%OUT_NAME%_bestwide.npz
    goto :deploypicked
  )
  echo   No %OUT_NAME%_bestwide.npz for this run -- pick 1 or N.
  goto :choosedeploy
)
if /I "%DEPLOY_PICK%"=="3" (
  if defined HAVE_LAP (
    set DEPLOY_SRC=%OUT_NAME%_bestlap.npz
    goto :deploypicked
  )
  echo   No %OUT_NAME%_bestlap.npz for this run -- pick 1 or N.
  goto :choosedeploy
)
if /I "%DEPLOY_PICK%"=="n" (
  echo Skipped deployment -- %CIRCUIT%.npz left untouched.
  goto :afterdeploy
)
echo   "%DEPLOY_PICK%" is not 1, 2, 3, or N -- try again.
goto :choosedeploy
:deploypicked
echo Checkpoint: %POLICIES%\%DEPLOY_SRC%
echo Currently deployed: %POLICIES%\%CIRCUIT%.npz
set /p DEPLOY="Replace the live %CIRCUIT%.npz with this checkpoint? [y/N] "
if /I "%DEPLOY%"=="y" (
  if exist "%POLICIES%\%CIRCUIT%.npz" (
    copy /Y "%POLICIES%\%CIRCUIT%.npz" "%POLICY_DIR%\%CIRCUIT%_deployed_prev.npz" >nul
    echo Previous %CIRCUIT%.npz backed up to %POLICY_DIR%\%CIRCUIT%_deployed_prev.npz
  )
  copy /Y "%POLICIES%\%DEPLOY_SRC%" "%POLICIES%\%CIRCUIT%.npz" >nul
  if errorlevel 1 (
    echo.
    echo Copy failed -- %CIRCUIT%.npz was NOT replaced. Nothing below this
    echo ran, RECORDS.md is untouched.
    goto :afterdeploy
  )
  echo Deployed %DEPLOY_SRC% as the live %CIRCUIT%.npz.
  set DEPLOYED=1
  if "%DEPLOY_SRC%"=="%OUT_NAME%_bestwide.npz" (
    set LAPFILE=%LAPTIME_WIDE%
  ) else if "%DEPLOY_SRC%"=="%OUT_NAME%_bestlap.npz" (
    set LAPFILE=%LAPTIME_LAP%
  ) else (
    set LAPFILE=%LAPTIME_BEST%
  )
  set LAPTIME=
  if exist "!LAPFILE!" set /p LAPTIME=<"!LAPFILE!"
  if "!LAPTIME!"=="" (
    echo.
    echo No clean lap time was recorded for %DEPLOY_SRC% -- update the
    echo %CIRCUIT% row in ai_sw\policy\RECORDS.md by hand.
  ) else (
    pushd "%REPO%\ai_sw"
    "%PY%" -m tools.update_records --circuit %CIRCUIT% --lap !LAPTIME! --checkpoint "%CIRCUIT%/%DEPLOY_SRC%"
    popd
  )
) else (
  echo Skipped deployment -- %CIRCUIT%.npz left untouched.
)
:afterdeploy

rem The qualifying ghost is a recording of the deployed policy, so a new
rem policy leaves the old ghost lap stale. Only asked when a deploy happened.
if not defined DEPLOYED goto :afterghost
echo.
set GHOST=
set /p GHOST="Re-record the %CIRCUIT% qualifying ghost with the new policy? [y/N] "
if /I not "%GHOST%"=="y" (
  echo Skipped -- the old %CIRCUIT% ghost lap is unchanged. To do it later:
  echo   cd ai_sw ^&^& ..\.venv312\Scripts\python.exe tools\record_ghost.py --circuit %CIRCUIT%
  goto :afterghost
)
pushd "%REPO%\ai_sw"
"%PY%" tools\record_ghost.py --circuit %CIRCUIT%
popd
:afterghost

echo.
echo === filing checkpoints and the log under ai_sw\policy\%CIRCUIT%\ and ai_sw\logs\%CIRCUIT%\ ===
move /Y "%POLICIES%\%OUT_NAME%.npz" "%POLICY_DIR%\%OUT_NAME%.npz" >nul 2>nul
move /Y "%POLICIES%\%OUT_NAME%_best.npz" "%POLICY_DIR%\%OUT_NAME%_best.npz" >nul 2>nul
move /Y "%POLICIES%\%OUT_NAME%_bestwide.npz" "%POLICY_DIR%\%OUT_NAME%_bestwide.npz" >nul 2>nul
move /Y "%POLICIES%\%OUT_NAME%_bestlap.npz" "%POLICY_DIR%\%OUT_NAME%_bestlap.npz" >nul 2>nul
move /Y "%POLICIES%\%OUT_NAME%_state.pt" "%POLICY_DIR%\%OUT_NAME%_state.pt" >nul 2>nul
wsl -d kali-linux bash -c "cp $HOME/aisw_kaggle/out/*.log /mnt/c/Users/user/Desktop/WorkSpace/Dongari/STEM2026/ai_sw/logs/%CIRCUIT%/kaggle_%OUT_NAME%.log 2>/dev/null"

echo.
echo Done. %OUT_NAME%: see %TRACE_PNG% for the trace and printed lap time above.
endlocal
