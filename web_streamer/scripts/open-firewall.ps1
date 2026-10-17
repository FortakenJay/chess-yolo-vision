# Allow iPhone/LAN access to Next.js (3000) and Python bridge (8080) on private Wi-Fi.
$rules = @(
  @{ Name = "Chess Next.js dev 3000"; Port = 3000 },
  @{ Name = "Chess phone bridge 8080"; Port = 8080 }
)

foreach ($rule in $rules) {
  $existing = netsh advfirewall firewall show rule name=$rule.Name 2>$null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Firewall rule already exists: $($rule.Name)"
    continue
  }
  netsh advfirewall firewall add rule name=$rule.Name dir=in action=allow protocol=TCP localport=$rule.Port profile=private | Out-Null
  Write-Host "Added firewall rule: $($rule.Name) (TCP $($rule.Port), private networks)"
}

$ip = (
  Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object { $_.IPAddress -match '^192\.168\.' } |
  Select-Object -First 1 -ExpandProperty IPAddress
)
if ($ip) {
  Write-Host ""
  Write-Host "LAN URLs (after npm run dev:https):"
  Write-Host "  https://${ip}:3000"
}
