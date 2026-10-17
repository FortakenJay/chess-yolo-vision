$ErrorActionPreference = "Stop"
$certDir = Join-Path (Join-Path $PSScriptRoot "..") "certs"
New-Item -ItemType Directory -Force -Path $certDir | Out-Null

$ip = (
  Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object { $_.IPAddress -match '^192\.168\.' } |
  Select-Object -First 1 -ExpandProperty IPAddress
)
if (-not $ip) {
  $ip = "192.168.0.2"
  Write-Warning "No 192.168.x address found; using default $ip"
}

$key = Join-Path $certDir "dev-key.pem"
$cert = Join-Path $certDir "dev-cert.pem"

$opensslCmd = Get-Command openssl -ErrorAction SilentlyContinue
$opensslExe = if ($opensslCmd) { $opensslCmd.Source } else { $null }
if (-not $opensslExe) {
  $gitOpenSsl = "C:\Program Files\Git\usr\bin\openssl.exe"
  if (Test-Path $gitOpenSsl) { $opensslExe = $gitOpenSsl }
}
if (-not $opensslExe) {
  Write-Host ""
  Write-Host "OpenSSL not found. Install one of:"
  Write-Host "  - Git for Windows (includes openssl in Git Bash)"
  Write-Host "  - choco install openssl"
  Write-Host "  - mkcert: https://github.com/FiloSottile/mkcert"
  Write-Host ""
  Write-Host "With mkcert:"
  Write-Host "  mkcert -install"
  Write-Host "  mkcert -key-file `"$key`" -cert-file `"$cert`" localhost 127.0.0.1 $ip"
  exit 1
}

$san = "DNS:localhost,IP:127.0.0.1,IP:$ip"
& $opensslExe req -x509 -newkey rsa:2048 -nodes `
  -keyout $key -out $cert -days 825 `
  -subj "/CN=chess-stream-dev" `
  -addext "subjectAltName=$san"

Write-Host ""
Write-Host "Certificate written:"
Write-Host "  $cert"
Write-Host "  SAN: $san"
Write-Host ""
Write-Host "On iPhone Safari open:  https://${ip}:3000"
Write-Host "Accept the certificate warning once, then allow camera."
