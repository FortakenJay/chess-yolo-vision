$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "open-firewall.ps1")

$ip = (
  Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object { $_.IPAddress -match '^192\.168\.' } |
  Select-Object -First 1 -ExpandProperty IPAddress
)
if (-not $ip) { $ip = "192.168.0.2" }

$certDir = Join-Path (Join-Path $PSScriptRoot "..") "certs"
$key = Join-Path $certDir "dev-key.pem"
$cert = Join-Path $certDir "dev-cert.pem"
if (-not (Test-Path $cert)) {
  Write-Host "Missing certs — run: npm run certs:gen"
  exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  iPhone Safari (same Wi-Fi):"
Write-Host "  https://${ip}:3000"
Write-Host ""
Write-Host "  Type https:// exactly. If Safari refuses the cert:"
Write-Host "  tap Show Details -> visit this website"
Write-Host ""
Write-Host "  Easier option (no cert): npm run dev:phone"
Write-Host "============================================================"
Write-Host ""

Set-Location (Join-Path $PSScriptRoot "..")
npx next dev --experimental-https -H 0.0.0.0 -p 3000 `
  --experimental-https-key $key --experimental-https-cert $cert
