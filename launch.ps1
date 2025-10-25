# Drone Tracking System Launch Script for Windows (PowerShell)
# Usage: .\launch.ps1 [options]

param(
    [Parameter(Position=0)]
    [Alias("c")]
    [string]$Camera = "0",

    [Alias("t")]
    [string]$TextPrompt = "drone",

    [Alias("n")]
    [string]$TestName = "default",

    [Alias("w")]
    [int]$WindowSize = 16,

    [Alias("r")]
    [int]$RstMult = 16,

    [Alias("b")]
    [int]$BbMult = 8,

    [Alias("s")]
    [int]$Size = 1330,

    [switch]$SendToBoard,
    [switch]$PrintCoord,
    [switch]$WriteOut,
    [switch]$DispOut,
    [switch]$Benchmarking,
    [switch]$Help
)

# Function to print colored messages
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

# Show help if requested
if ($Help) {
    @"
Usage: .\launch.ps1 [OPTIONS]

Drone tracking system using CoTracker and Grounding DINO

OPTIONS:
    -Camera <id>            Camera ID (integer) or path to video file
                           (default: media/ds_pan_cut.mp4)
                           Alias: -c

    -TextPrompt <text>     Text prompt for detection (default: "drone")
                           Alias: -t

    -TestName <name>       Name for test run (default: "default")
                           Alias: -n

    -WindowSize <size>     CoTracker window size (default: 16)
                           Alias: -w

    -RstMult <mult>        Reset interval multiplier (default: 16)
                           Alias: -r

    -BbMult <mult>         Bounding box check multiplier (default: 8)
                           Alias: -b

    -Size <size>           Max image width (default: 1330)
                           Alias: -s

    -SendToBoard           Send coordinates to board
    -PrintCoord            Print coordinates to console
    -WriteOut              Write output to file
    -DispOut               Display output window
    -Benchmarking          Enable benchmarking mode

    -Help                  Display this help message

EXAMPLES:
    # Run with video file and display output
    .\launch.ps1 -Camera media/video.mp4 -DispOut -PrintCoord

    # Using aliases
    .\launch.ps1 -c media/video.mp4 -DispOut -PrintCoord

    # Run with camera 0, benchmarking enabled
    .\launch.ps1 -c 0 -Benchmarking -WriteOut

    # Track a specific object with custom settings
    .\launch.ps1 -c media/bird.mp4 -t "bird" -n "bird_test" -DispOut

    # Set custom image size
    .\launch.ps1 -c 0 -s 800 -DispOut

"@
    exit 0
}

# Check if Python is available
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw
    }
} catch {
    Write-Error-Custom "Python is not installed or not in PATH"
    exit 1
}

# Check if main.py exists
if (-not (Test-Path "main.py")) {
    Write-Error-Custom "main.py not found in current directory"
    exit 1
}

# Print configuration
Write-Info "Starting drone tracking system..."
Write-Info "Camera: $Camera"
Write-Info "Text prompt: $TextPrompt"
Write-Info "Test name: $TestName"
Write-Info "Window size: $WindowSize"
Write-Info "Image size: $Size"
Write-Host ""

# Build argument list
$arguments = @(
    "main.py",
    $Camera,
    "--text-prompt", "`"$TextPrompt`"",
    "--test-name", "`"$TestName`"",
    "--window-size", $WindowSize,
    "--rst-interval-mult", $RstMult,
    "--bb-check-mult", $BbMult,
    "--img-size", $Size
)

if ($SendToBoard) { $arguments += "--send-to-board" }
if ($PrintCoord) { $arguments += "--print-coord" }
if ($WriteOut) { $arguments += "--write-out" }
if ($DispOut) { $arguments += "--disp-out" }
if ($Benchmarking) { $arguments += "--benchmarking" }

# Execute the command
Write-Info "Executing: python $($arguments -join ' ')"
Write-Host ""

try {
    & python $arguments

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Info "Program completed successfully"
    } else {
        Write-Host ""
        Write-Error-Custom "Program exited with error code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-Host ""
    Write-Error-Custom "Failed to execute program: $_"
    exit 1
}

exit 0