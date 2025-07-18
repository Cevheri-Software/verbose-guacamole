#!/usr/bin/env python3

# mavsdk_takeoff_and_forward.py

# 

# Gereksinimler:

# pip install mavsdk==1.* ( >=1.9.0 önerilir )

# Çalıştır:  python3 mavsdk_takeoff_and_forward.py

import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError,
PositionNedYaw)

TAKEOFF_ALT = 3.0      # metre
LEG_DISTANCE = 10.0    # metre (kuzeye = ileri)

class OffboardController:
def **init**(self):
self.current_setpoint = None
self.setpoint_task = None
self.running = False

```
async def start_setpoint_stream(self, drone, setpoint):
    """Continuously send setpoints at 20Hz"""
    self.current_setpoint = setpoint
    self.running = True
    self.setpoint_task = asyncio.create_task(self._setpoint_loop(drone))

async def update_setpoint(self, setpoint):
    """Update the current setpoint"""
    self.current_setpoint = setpoint

async def stop_setpoint_stream(self):
    """Stop the setpoint stream"""
    self.running = False
    if self.setpoint_task:
        self.setpoint_task.cancel()
        try:
            await self.setpoint_task
        except asyncio.CancelledError:
            pass

async def _setpoint_loop(self, drone):
    """Internal loop that sends setpoints at 20Hz"""
    while self.running:
        if self.current_setpoint:
            try:
                await drone.offboard.set_position_ned(self.current_setpoint)
                await asyncio.sleep(0.05)  # 20Hz
            except Exception as e:
                print(f"Setpoint gönderme hatası: {e}")
                break
```

async def run():
drone = System(mavsdk_server_address=“localhost”, port=50051)
print(“Drone’a bağlanıyor…”)
await drone.connect()

```
# Bağlantı doğrulama
async for state in drone.core.connection_state():
    if state.is_connected:
        print("Drone'a bağlantı başarılı!")
        break

# GPS lock bekle
print("GPS lock bekleniyor...")
async for health in drone.telemetry.health():
    if health.is_global_position_ok and health.is_home_position_ok:
        print("✓ GPS lock tamam!")
        break

# Arm + Kalk
print("► Arm ediliyor…")
await drone.action.arm()

print(f"► {TAKEOFF_ALT} m kalkış...")
await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
await drone.action.takeoff()

# İstenen irtifaya çıkana kadar bekle
print("Kalkış tamamlanması bekleniyor...")
async for position in drone.telemetry.position():
    if abs(position.relative_altitude_m - TAKEOFF_ALT) < 0.5:
        print("✓ Kalkış tamamlandı!")
        break
    await asyncio.sleep(0.1)

# Kısa bir stabilizasyon beklemesi
await asyncio.sleep(2)

# Offboard controller başlat
controller = OffboardController()

# Mevcut pozisyonu al ve başlangıç setpoint'i oluştur
current_pos = None
async for position in drone.telemetry.position_velocity_ned():
    current_pos = position.position
    break

if current_pos is None:
    print("Mevcut pozisyon alınamadı!")
    await drone.action.land()
    return

# Başlangıç setpoint'i (mevcut pozisyonu koru)
initial_setpoint = PositionNedYaw(
    north_m=current_pos.north_m,
    east_m=current_pos.east_m,
    down_m=current_pos.down_m,
    yaw_deg=0.0
)

print("► Setpoint stream başlatılıyor...")
await controller.start_setpoint_stream(drone, initial_setpoint)

# Setpoint stream başladıktan sonra kısa bekle
await asyncio.sleep(1)

print("► Offboard başlatılıyor…")
try:
    await drone.offboard.start()
    print("✓ Offboard modu aktif!")
except OffboardError as e:
    print(f"Offboard başlatılamadı: {e}")
    await controller.stop_setpoint_stream()
    print("İniş yapılıyor…")
    await drone.action.land()
    return

# 10 m ileri hedef setpoint
target = PositionNedYaw(
    north_m=current_pos.north_m + LEG_DISTANCE,
    east_m=current_pos.east_m,
    down_m=current_pos.down_m,  # Aynı irtifada kal
    yaw_deg=0.0
)

print("► 10 m ileri uçuş komutu gönderildi.")
await controller.update_setpoint(target)

# Hedefe yaklaşana kadar konum izle
target_reached = False
timeout_counter = 0
max_timeout = 300  # 30 saniye (0.1s * 300)

async for odom in drone.telemetry.position_velocity_ned():
    north_err = abs((current_pos.north_m + LEG_DISTANCE) - odom.position.north_m)
    
    if north_err < 1.0:  # 1 metre tolerans
        print("✓ Hedefe ulaşıldı.")
        target_reached = True
        break
    
    timeout_counter += 1
    if timeout_counter > max_timeout:
        print("⚠ Timeout: Hedefe ulaşılamadı")
        break
        
    await asyncio.sleep(0.1)

# 5 saniye hedefe yakın pozisyonda bekle
if target_reached:
    print("► Hedefe yakın pozisyonda 5 saniye bekleniyor...")
    await asyncio.sleep(5)

# Offboard çıkışı ve iniş
print("► Offboard durduruluyor...")
try:
    await drone.offboard.stop()
    print("✓ Offboard modu durduruldu")
except OffboardError:
    print("Offboard zaten devre dışı.")

# Setpoint stream'i durdur
await controller.stop_setpoint_stream()

print("► Return to Launch...")
await drone.action.return_to_launch()

# İnişi bekle
print("İniş bekleniyor...")
async for in_air in drone.telemetry.in_air():
    if not in_air:
        print("✓ İniş tamamlandı!")
        break
    await asyncio.sleep(1)

print("► Disarm...")
await drone.action.disarm()
print("✓ Görev tamamlandı!")
```

if **name** == “**main**”:
asyncio.run(run())
