#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import subprocess
import time
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Define the simulation time windows.
spinup_start = datetime.strptime("2017-10-01 00", "%Y-%m-%d %H")
spinup_end   = datetime.strptime("2018-10-01 00", "%Y-%m-%d %H")

# Climatology starts after spin-up and ends at global_end.
clim_start = spinup_end  # first 3-year block starts on spinup_end
global_end = datetime.strptime("2019-10-01 00", "%Y-%m-%d %H")

# Maximum period (in years) for each run.
max_period_years = 1

# Assume the current working directory is the main directory (where EXE, INPUT, OUTPUT, LOG reside)
main_dir = os.getcwd()
print("Main directory (absolute):", main_dir)

# Define executable file name
if sys.platform == 'darwin':
    exe_file = "ic-top_macos_v1_1.sh"
elif sys.platform == 'win32':
    exe_file = "ic-top_win_v1_1.exe"
elif sys.platform == "linux":
    exe_file = "pyic_top.sh"
else:
    print(f"Platform {sys.platform} not supported")
    sys.exit(0)

# Define relative directories.
exe_dir    = "/home/mzaics/codes/pyic-top/scripts"
input_dir  = "/home/mzaics/codes/pyic-top-files/INPUT_286"
output_dir = "/home/mzaics/codes/pyic-top-files/OUTPUT"
log_dir    = "/home/mzaics/codes/pyic-top-files/ARCHIVE"

# Build absolute paths.
exe_path = os.path.join(main_dir, exe_dir, exe_file)
sim_file_path = os.path.join(main_dir, "init.json")
initcond_folder = os.path.join(main_dir, input_dir, "initcond")
output_dir_abs = os.path.join(main_dir, output_dir)
log_initcond_folder = os.path.join(main_dir, log_dir, "initconds")
log_outconc_folder  = os.path.join(main_dir, log_dir, "OUT_CONC")

# Create LOG directories if they don't exist.
os.makedirs(os.path.join(main_dir, log_dir), exist_ok=True)
os.makedirs(log_initcond_folder, exist_ok=True)
os.makedirs(log_outconc_folder, exist_ok=True)

# Change working directory to the EXE folder for running the executable.
os.chdir(os.path.join(main_dir, exe_dir))


def update_sim_file(sim_file, new_start, new_end):
    """
    Update the simulation file with new start and end dates.
    The file is assumed to have the start time on line 2 and end time on line 3.
    """
    with open(sim_file) as f:
        initj = json.load(f)
        print(initj)

    start_str = new_start.strftime("%Y-%m-%d %H:00:00")
    end_str   = new_end.strftime("%Y-%m-%d %H:00:00")
    # R script updates lines 2 and 3 (1-indexed). In Python, these are lines[1] and lines[2]
    if len(initj) >= 3:
        initj['start_time'] = f"{start_str}"
        initj['end_time'] = f"{end_str}"
    else:
        print("The simulation file does not have the expected format.")
        return
    
    with open(sim_file, 'w') as f:
        json.dump(initj, f, ensure_ascii=False, indent=2)

    print(f"Updated simulation file with start={start_str} and end={end_str}")


def concatenate_outputs(source_folder, log_out_folder):
    """
    Concatenate files whose names start with 'out_'.
    If a combined file exists in the log folder, append the new data (dropping the header).
    Otherwise, create a new combined file.
    """
    out_files = glob.glob(os.path.join(source_folder, "out_*"))
    if not out_files:
        print("No output files starting with 'out_' found in the OUTPUT folder.")
        return

    for outfile in out_files:
        base_name = os.path.basename(outfile)
        combined_file = os.path.join(log_out_folder, f"combined_{base_name}")

        with open(outfile, 'r') as f:
            file_content = f.readlines()

        if not os.path.exists(combined_file):
            # Create combined file with full content (header + data).
            with open(combined_file, 'w') as cf:
                cf.writelines(file_content)
            print(f"Created new combined file: {combined_file}")
        else:
            # Append the content without the header (assumes the first line is the header).
            with open(combined_file, 'a') as cf:
                cf.writelines(file_content[1:])
            print(f"Appended data (without header) from {base_name} to combined file: {combined_file}")


def update_initial_conditions(source_folder, dest_folder, log_folder, period_label):
    """
    Update initial conditions by copying state files (starting with 'state_var_') from
    the OUTPUT directory to the INPUT/initcond folder and archiving them in LOG/initconds/<period_label>.
    """
    state_files = glob.glob(os.path.join(source_folder, "state_var_*"))
    if not state_files:
        print("No state_var_ files found in the OUTPUT folder to update initial conditions.")
        return

    # Create a period subfolder in the LOG/initconds folder.
    period_dir = os.path.join(log_folder, period_label)
    os.makedirs(period_dir, exist_ok=True)

    for sf in state_files:
        dest_file = os.path.join(dest_folder, os.path.basename(sf))
        shutil.copy2(sf, dest_file)
        print(f"Updated initial condition: copied {sf} to {dest_file}")

        # Also copy to the log directory.
        log_file = os.path.join(period_dir, os.path.basename(sf))
        shutil.copy2(sf, log_file)
        print(f"Logged initial condition copy to: {log_file}")


def compute_simulation_periods(spinup_start, spinup_end, clim_start, global_end, max_yr):
    """
    Compute simulation periods.
    The first period is the spin-up. Subsequent periods are computed in max_yr blocks until global_end.
    """
    periods = []
    # Add spin-up period.
    periods.append({'start': spinup_start, 'end': spinup_end})

    sim_start = clim_start
    while sim_start < global_end:
        sim_end_candidate = sim_start + relativedelta(years=max_yr)
        sim_end = global_end if sim_end_candidate > global_end else sim_end_candidate
        periods.append({'start': sim_start, 'end': sim_end})
        sim_start = sim_end
    return periods


sim_periods = compute_simulation_periods(spinup_start, spinup_end, clim_start, global_end, max_period_years)
print("Simulation periods to run:")
for i, period in enumerate(sim_periods, start=1):
    print(f"Run {i}: {period['start'].date()} to {period['end'].date()}")

# MAIN SIMULATION LOOP
for i, period in enumerate(sim_periods, start=1):
    period_label = f"{period['start'].year}_{period['end'].year}"
    print("\n============================================")
    print(f"Starting simulation run {i} for period: {period_label}")
    
    # 1. Update simulation file with new start and end dates.
    update_sim_file(sim_file_path, period['start'], period['end'])
    
    # Optional pause to ensure file-system consistency.
    time.sleep(1)
    
    # 2. Run the simulation executable.
    # This assumes the executable is callable in this fashion.
    if sys.platform == 'win32':
        # exec_command = f'echo.| "{os.path.basename(exe_path)}"'
        exec_command = f'echo.| "{exe_path}"'
        print("Running executable (with bypass for Fortran pause)...")
        exec_status = subprocess.call(exec_command, shell=True)
    elif sys.platform in ['linux', 'darwin']:
        # exec_command = f'bash "{os.path.basename(exe_path)}"'
        exec_command = f'bash {exe_path}'
        print("Running bash...")
        exec_status = subprocess.call(exec_command, shell=True)
    
    if exec_status != 0:
        print(
            f"ERROR: Executable terminated with status "
            f"{exec_status} at run {i}. Aborting further runs")
        break
    print("Executable run completed successfully.")
    
    # 3. Concatenate output files.
    print("Concatenating output files...")
    concatenate_outputs(output_dir_abs, log_outconc_folder)
    
    # 4. Update initial conditions.
    print("Updating initial conditions and archiving copies...")
    update_initial_conditions(output_dir_abs, initcond_folder, log_initcond_folder, period_label)
    
    print(f"Completed simulation run {i} for period: {period_label}")

print("\nAll simulation runs completed.")
