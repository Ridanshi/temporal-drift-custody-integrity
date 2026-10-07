# Recursively downloads VISION dataset device folders from the Apache-style
# directory listing at lesc.dinfo.unifi.it, preserving folder structure.
#
# Usage:
#   .\download_vision_devices.ps1 -Devices "D02_Apple_iPhone4s","D06_Apple_iPhone6"
#
# Downloads only the videos/ subfolder for each device (skips images/).

param(
    [Parameter(Mandatory = $true)]
    [string[]]$Devices,

    [string]$BaseUrl = "http://lesc.dinfo.unifi.it/VISION/dataset",
    [string]$OutRoot = "D:\DF paper\data\vision"
)

function Download-Recursive {
    param(
        [string]$Url,
        [string]$LocalDir
    )

    if (-not (Test-Path $LocalDir)) {
        New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
    }

    $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing
    $links = $resp.Links | Where-Object { $_.href -and $_.href -ne "../" -and -not $_.href.StartsWith("?") }

    foreach ($link in $links) {
        $href = $link.href
        $childUrl = if ($href -match "^https?://") { $href } else { "$Url/$href".Replace("//", "/").Replace(":/", "://") }
        $name = [System.Uri]::UnescapeDataString($href.TrimEnd("/"))

        if ($href.EndsWith("/")) {
            Write-Host "Entering folder: $name"
            Download-Recursive -Url ($Url.TrimEnd("/") + "/" + $href) -LocalDir (Join-Path $LocalDir $name)
        }
        else {
            $localFile = Join-Path $LocalDir $name
            if (Test-Path $localFile) {
                Write-Host "Skip (exists): $name"
                continue
            }
            Write-Host "Downloading: $name"
            try {
                Invoke-WebRequest -Uri ($Url.TrimEnd("/") + "/" + $href) -OutFile $localFile -UseBasicParsing
            }
            catch {
                Write-Warning "Failed: $name -> $($_.Exception.Message)"
            }
        }
    }
}

foreach ($device in $Devices) {
    $videosUrl = "$BaseUrl/$device/videos"
    $localVideosDir = Join-Path $OutRoot "$device\videos"
    Write-Host "=== Device: $device ==="
    Download-Recursive -Url $videosUrl -LocalDir $localVideosDir
}

Write-Host "Done."
