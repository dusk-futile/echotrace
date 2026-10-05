# EchoTrace-Check.ps1
# Read-only survey of this PC for the EchoTrace project.
#  - needs NO administrator rights
#  - installs nothing, changes nothing
#  - sends nothing anywhere (the only network step is a TCP connection test)
# It speaks a short message at the start and at the end, writes EchoTrace-report.txt
# to the Desktop and copies the same text to the clipboard.
# ASCII only on purpose: Windows PowerShell 5.1 misreads UTF-8 without a BOM.

$ErrorActionPreference = 'Continue'
$script:lines = New-Object System.Collections.ArrayList

function Out-Report([string]$text) {
    [void]$script:lines.Add($text)
    Write-Host $text
}

function Section([string]$title) {
    Out-Report ''
    Out-Report ('== ' + $title + ' ==')
}

function Say([string]$text) {
    try {
        Add-Type -AssemblyName System.Speech -ErrorAction Stop
        $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $s.Speak($text)
        $s.Dispose()
    } catch { }
}

function Safe([string]$label, [scriptblock]$block) {
    try { & $block } catch { Out-Report ($label + ': could not read (' + $_.Exception.Message + ')') }
}

Say 'Checking this computer. This takes about one minute.'

Out-Report ('EchoTrace PC check, ' + (Get-Date -Format 'yyyy-MM-dd HH:mm'))
Out-Report ('Computer: ' + $env:COMPUTERNAME + '   User: ' + $env:USERNAME)
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Out-Report ('Running as administrator: ' + $isAdmin + '   (expected: False)')
Out-Report ('PowerShell version: ' + $PSVersionTable.PSVersion.ToString())

Section 'Windows'
Safe 'Windows' {
    $os = Get-CimInstance Win32_OperatingSystem
    Out-Report ('Name: ' + $os.Caption)
    Out-Report ('Version / build: ' + $os.Version + ' / ' + $os.BuildNumber)
    Out-Report ('Architecture: ' + $os.OSArchitecture)
    $cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    Out-Report ('Release: ' + $cv.DisplayVersion + ' ' + $cv.ReleaseId)
}

Section 'Processor'
Safe 'Processor' {
    foreach ($c in @(Get-CimInstance Win32_Processor)) {
        Out-Report ('Name: ' + $c.Name.Trim())
        Out-Report ('Cores / threads: ' + $c.NumberOfCores + ' / ' + $c.NumberOfLogicalProcessors)
        Out-Report ('Max clock (MHz): ' + $c.MaxClockSpeed)
    }
}
Safe 'Instruction sets' {
    Add-Type -Namespace EchoNative -Name K32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint f);'
    $features = [ordered]@{ 'SSE3' = 13; 'SSSE3' = 36; 'SSE4.1' = 37; 'SSE4.2' = 38; 'AVX' = 39; 'AVX2' = 40 }
    $parts = @()
    foreach ($k in $features.Keys) {
        $parts += ($k + '=' + [EchoNative.K32]::IsProcessorFeaturePresent([uint32]$features[$k]))
    }
    Out-Report ('Instruction sets: ' + ($parts -join '  '))
    Out-Report '(Recent NumPy needs SSE4.2. If it says False the Python packages must be pinned to older versions.)'
}

Section 'Memory and disk'
Safe 'Memory' {
    $cs = Get-CimInstance Win32_ComputerSystem
    $os = Get-CimInstance Win32_OperatingSystem
    Out-Report ('RAM installed (GB): ' + [math]::Round($cs.TotalPhysicalMemory / 1GB, 1))
    Out-Report ('RAM free right now (GB): ' + [math]::Round($os.FreePhysicalMemory / 1MB, 1))
}
Safe 'Disk space' {
    $drive = $env:SystemDrive
    $d = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $drive + "'")
    Out-Report ('System drive ' + $drive + ' free / total (GB): ' + [math]::Round($d.FreeSpace / 1GB, 1) + ' / ' + [math]::Round($d.Size / 1GB, 1))
}
Safe 'Disk type' {
    foreach ($p in @(Get-PhysicalDisk -ErrorAction Stop)) {
        Out-Report ('Disk: ' + $p.FriendlyName + ', type ' + $p.MediaType + ', ' + [math]::Round($p.Size / 1GB) + ' GB')
    }
}

Section 'Screen and audio'
Safe 'Screen' {
    Add-Type -AssemblyName System.Windows.Forms
    foreach ($s in [System.Windows.Forms.Screen]::AllScreens) {
        $tag = ''
        if ($s.Primary) { $tag = ' (primary)' }
        Out-Report ('Screen: ' + $s.Bounds.Width + 'x' + $s.Bounds.Height + $tag)
    }
    $dpi = (Get-ItemProperty 'HKCU:\Control Panel\Desktop\WindowMetrics' -ErrorAction SilentlyContinue).AppliedDPI
    if ($dpi) { Out-Report ('Display scaling: ' + [math]::Round($dpi / 96 * 100) + ' percent') }
}
Safe 'Audio' {
    foreach ($a in @(Get-CimInstance Win32_SoundDevice)) {
        Out-Report ('Sound device: ' + $a.Name + ' (' + $a.Status + ')')
    }
}

Section 'NVDA'
Safe 'NVDA' {
    $procs = @(Get-Process -Name nvda -ErrorAction SilentlyContinue)
    if ($procs.Count -gt 0) {
        $path = $null
        try { $path = $procs[0].Path } catch { }
        if ($path) { Out-Report ('NVDA running: yes (' + $path + ')') } else { Out-Report 'NVDA running: yes' }
        if ($path -and (Test-Path $path)) {
            $vi = (Get-Item $path).VersionInfo
            Out-Report ('NVDA version: ' + $vi.FileVersion + '   product: ' + $vi.ProductVersion)
        }
        try { Out-Report ('NVDA memory right now (MB): ' + [math]::Round($procs[0].WorkingSet64 / 1MB)) } catch { }
    } else {
        Out-Report 'NVDA running: no'
    }
    foreach ($p in @("$env:ProgramFiles\NVDA\nvda.exe", "${env:ProgramFiles(x86)}\NVDA\nvda.exe")) {
        if (Test-Path $p) { Out-Report ('NVDA installed at: ' + $p + ' (version ' + (Get-Item $p).VersionInfo.FileVersion + ')') }
    }
    $cfg = Join-Path $env:APPDATA 'nvda'
    if (Test-Path $cfg) {
        Out-Report ('NVDA settings folder: ' + $cfg)
        $ad = Join-Path $cfg 'addons'
        if (Test-Path $ad) {
            foreach ($d in @(Get-ChildItem $ad -Directory)) { Out-Report ('  add-on: ' + $d.Name) }
        }
    }
}

Section 'Speech voices'
Safe 'SAPI voices' {
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    foreach ($v in $synth.GetInstalledVoices()) {
        $i = $v.VoiceInfo
        Out-Report ('SAPI voice: ' + $i.Name + ' [' + $i.Culture + ', ' + $i.Gender + ']')
    }
    $synth.Dispose()
}
Safe 'OneCore voices' {
    $k = 'HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens'
    if (Test-Path $k) {
        foreach ($t in @(Get-ChildItem $k)) {
            $name = (Get-ItemProperty $t.PSPath).'(default)'
            Out-Report ('OneCore voice: ' + $name + '  (' + $t.PSChildName + ')')
        }
    }
}

Section 'Languages and Windows OCR'
Safe 'User languages' {
    foreach ($l in @(Get-WinUserLanguageList)) {
        Out-Report ('User language: ' + $l.LanguageTag + ' (' + $l.LocalizedName + ')')
    }
}
Safe 'Windows OCR' {
    [void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
    $langs = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages
    $arabic = $false
    foreach ($x in $langs) {
        Out-Report ('Windows OCR language: ' + $x.LanguageTag + ' (' + $x.DisplayName + ')')
        if ($x.LanguageTag -like 'ar*') { $arabic = $true }
    }
    Out-Report ('Arabic available to Windows OCR: ' + $arabic)
}

Section 'Network (connection test only, nothing is sent)'
function Test-Host([string]$h, [int]$port) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $ar = $c.BeginConnect($h, $port, $null, $null)
        $ok = $ar.AsyncWaitHandle.WaitOne(4000, $false)
        $sw.Stop()
        if ($ok -and $c.Connected) {
            Out-Report ($h + ':' + $port + ' reachable in ' + $sw.ElapsedMilliseconds + ' ms')
        } else {
            Out-Report ($h + ':' + $port + ' NOT reachable')
        }
        $c.Close()
    } catch {
        Out-Report ($h + ':' + $port + ' NOT reachable (' + $_.Exception.Message + ')')
    }
}
Test-Host 'generativelanguage.googleapis.com' 443
Test-Host 'github.com' 443
Test-Host 'huggingface.co' 443

Section 'Security software'
Safe 'Antivirus' {
    foreach ($av in @(Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct)) {
        Out-Report ('Antivirus: ' + $av.displayName)
    }
}

Section 'Speed test (higher is faster; compare the lab PCs with each other)'
Safe 'Speed' {
    Add-Type -TypeDefinition @'
public static class EchoBench {
    public static double Run() {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        double acc = 0;
        long n = 0;
        while (sw.ElapsedMilliseconds < 1500) {
            for (int i = 1; i < 20000; i++) { acc += System.Math.Sqrt(i) * System.Math.Sin(i); n++; }
        }
        if (acc == 12345.678) { n++; }
        return n / sw.Elapsed.TotalSeconds / 1000000.0;
    }
}
'@
    $score = [EchoBench]::Run()
    Out-Report ('Single-core score (millions of operations per second): ' + [math]::Round($score, 2))
}

Section 'Done'
$desktop = [Environment]::GetFolderPath('Desktop')
$report = Join-Path $desktop 'EchoTrace-report.txt'
try { $script:lines | Set-Content -Path $report -Encoding UTF8 } catch { Write-Host ('Could not save the report: ' + $_.Exception.Message) }
try { Set-Clipboard -Value (($script:lines) -join "`r`n") } catch { }
Out-Report ('Report saved to: ' + $report + '  (also copied to the clipboard)')
Say 'The check is finished. The report is on the desktop and on the clipboard. Please send it to the person who gave you this tool.'
