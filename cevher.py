from ultralytics import YOLO
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel
import numpy as np
import time
import socket

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

class DetectionSystem:
    def __init__(self, source=0, output_ip="192.168.1.100", output_port=5000):
        # Modeller
        torch.serialization.add_safe_globals([DetectionModel])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.coco_model = YOLO('yolov8n.pt').to(self.device)
        self.license_plate_detector = YOLO('license_plate_detector.pt').to(self.device)
        self.gun_detector = YOLO('gun_deteckter.pt').to(self.device)
        
        # Tracking
        self.mot_tracker = Sort()
        self.mot_gun_tracker = Sort()
        self.results = {}
        self.tracked_guns = {}
        self.detected_plates = set()
        self.frame_nmr = -1
        
        # Ayarlar
        self.vehicles = [2, 3, 5, 7]
        self.SCORE_THRESHOLD = 0.6
        self.MIN_GUN_AREA = 800
        self.MAX_GUN_AREA = 50000
        self.GUN_TRACK_MIN_FRAMES = 15
        
        # Video giriş
        self.setup_input(source)
        
        # Video çıkış (diğer cihaza stream)
        self.output_ip = output_ip
        self.output_port = output_port
        self.wait_for_target_device()
        self.setup_output(output_ip, output_port)
    
    def wait_for_target_device(self):
        """Hedef cihaz hazır olana kadar bekle"""
        print("="*60)
        print("🔍 Hedef cihaz bekleniyor...")
        print(f"   IP: {self.output_ip}")
        print(f"   Port: {self.output_port}")
        print("="*60)
        print("ℹ️  Hedef cihazda aşağıdaki komutlardan birini çalıştırın:")
        print("   • VLC: udp://@:5000")
        print("   • GStreamer: gst-launch-1.0 udpsrc port=5000 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink")
        print("   • FFplay: ffplay udp://localhost:5000")
        print("="*60)
        
        attempt = 1
        while True:
            try:
                # UDP socket ile hedef cihaza test paketi gönder
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # Timeout ayarla
                test_socket.settimeout(2)
                
                # Test mesajı gönder
                test_message = b"connection_test"
                test_socket.sendto(test_message, (self.output_ip, self.output_port))
                
                print(f"✅ Hedef cihaz hazır! ({attempt}. deneme)")
                test_socket.close()
                break
                
            except socket.timeout:
                print(f"⏳ Hedef cihaz bekleniyor... ({attempt}. deneme)")
            except socket.error as e:
                print(f"⏳ Hedef cihaz bekleniyor... ({attempt}. deneme) - {e}")
            except Exception as e:
                print(f"⏳ Hedef cihaz bekleniyor... ({attempt}. deneme) - {e}")
            finally:
                try:
                    test_socket.close()
                except:
                    pass
            
            attempt += 1
            time.sleep(3)  # 3 saniye bekle
    
    def setup_input(self, source):
        """Video girişini ayarla"""
        print(f"Video giriş kaynağı ayarlanıyor: {source}")
        
        if isinstance(source, str) and source.startswith('rtsp://'):
            print("RTSP stream bağlantısı kuruluyor...")
            # RTSP için GStreamer
            pipeline = (
                f"rtspsrc location={source} latency=0 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! appsink drop=true max-buffers=1"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            print(f"Kamera ID {source} açılıyor...")
            # Normal kamera
            self.cap = cv2.VideoCapture(source)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Kamera kontrolü
        if not self.cap.isOpened():
            print("❌ HATA: Video kaynağı açılamadı!")
            print(f"   Kaynak: {source}")
            if isinstance(source, int):
                print("   Kamera bağlı olduğundan emin olun.")
            else:
                print("   RTSP URL'sini kontrol edin.")
            raise Exception("Video input initialization failed")
        else:
            print("✅ Video giriş kaynağı başarıyla açıldı")
    
    def setup_output(self, ip, port):
        """Diğer cihaza stream çıkışı"""
        print(f"Stream çıkışı ayarlanıyor: {ip}:{port}")
        
        # GStreamer ile UDP stream
        self.output_pipeline = (
            f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 ! "
            f"rtph264pay ! udpsink host={ip} port={port}"
        )
        
        try:
            self.out = cv2.VideoWriter(
                self.output_pipeline, 
                cv2.CAP_GSTREAMER, 
                0, 30, (1280, 720)
            )
            
            if not self.out.isOpened():
                print("❌ HATA: Video stream çıkışı açılamadı!")
                print(f"   Hedef IP: {ip}\n port:{port}")
                print("   GStreamer kurulumunu kontrol edin.")
                raise Exception("Video output initialization failed")
            else:
                print(f"✅ Video stream çıkışı başarıyla ayarlandı: {ip}:{port}")
                
        except Exception as e:
            print(f"❌ Stream çıkışı hatası: {e}")
            raise
    
    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    def process_frame(self, frame):
        self.frame_nmr += 1
        self.results[self.frame_nmr] = {}
        
        # Araç tespiti
        detections = self.coco_model(frame, device=self.device)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # Araç tracking
        track_ids = self.mot_tracker.update(np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5)))
        
        # Plaka tespiti
        license_plates = self.license_plate_detector(frame, device=self.device)[0]
        drawn_car_ids = set()
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1 and car_id not in drawn_car_ids:
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                drawn_car_ids.add(car_id)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Plaka oku
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_plate_crop.size > 0:
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text and license_plate_text not in self.detected_plates:
                    with open('plakalar.txt', 'a', encoding='utf-8') as f:
                        f.write(license_plate_text + '\n')
                    self.detected_plates.add(license_plate_text)
                    
                    self.results[self.frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
        
        # Silah tespiti
        gun_detections = self.gun_detector(frame, device=self.device)[0]
        gun_detections_ = []
        for gun in gun_detections.boxes.data.tolist():
            gx1, gy1, gx2, gy2, gscore, gclass_id = gun
            area = (gx2 - gx1) * (gy2 - gy1)
            if gscore >= self.SCORE_THRESHOLD and self.MIN_GUN_AREA < area < self.MAX_GUN_AREA:
                gun_detections_.append([gx1, gy1, gx2, gy2, gscore])
        
        # Silah tracking
        gun_tracks = self.mot_gun_tracker.update(np.asarray(gun_detections_) if len(gun_detections_) > 0 else np.empty((0, 5)))
        
        for track in gun_tracks:
            x1, y1, x2, y2, gun_id = track
            gun_id = int(gun_id)
            
            if gun_id not in self.tracked_guns:
                self.tracked_guns[gun_id] = {'count': 1, 'last_frame': self.frame_nmr}
            else:
                self.tracked_guns[gun_id]['count'] += 1
                self.tracked_guns[gun_id]['last_frame'] = self.frame_nmr
            
            # Silah göster
            if self.tracked_guns[gun_id]['count'] >= self.GUN_TRACK_MIN_FRAMES:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(frame, 'GUN DETECTED', (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        return frame
    
    def run(self):
        print("="*60)
        print("🚀 Tespit Sistemi Başlatıldı")
        print("="*60)
        print("⚠️  HEDEF CİHAZ UYARISI:")
        print(f"   Stream gönderiliyor: 192.168.1.100:5000")
        print("   Hedef cihazda aşağıdaki komutlardan birini çalıştırın:")
        print("   • VLC: udp://@:5000")
        print("   • GStreamer: gst-launch-1.0 udpsrc port=5000 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink")
        print("   • FFplay: ffplay udp://localhost:5000")
        print("="*60)
        print("📋 Kontroller:")
        print("   'q' - Sistemi durdur")
        print("   's' - Screenshot al")
        print("="*60)
        
        frame_count = 0
        start_time = time.time()
        last_status_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ HATA: Frame okunamadı! Kamera bağlantısını kontrol edin.")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Frame işle
            try:
                processed_frame = self.process_frame(frame)
            except Exception as e:
                print(f"❌ Frame işleme hatası: {e}")
                continue
            
            # Diğer cihaza stream gönder
            try:
                frame_resized = cv2.resize(processed_frame, (1280, 720))
                success = self.out.write(frame_resized)
                if not success:
                    print("⚠️  Stream gönderimi başarısız! Hedef cihaz bağlantısını kontrol edin.")
            except Exception as e:
                print(f"❌ Stream gönderim hatası: {e}")
            
            # Her 5 saniyede durum bilgisi
            if current_time - last_status_time >= 5.0:
                fps = frame_count / (current_time - start_time)
                print(f"📊 Durum: {frame_count} frame işlendi, FPS: {fps:.1f}")
                last_status_time = current_time
            
            # Klavye kontrolü (q ile çıkış, s ile screenshot)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Sistem durduruluyor...")
                break
            elif key == ord('s'):
                screenshot_name = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(screenshot_name, processed_frame)
                print(f"📸 Screenshot kaydedildi: {screenshot_name}")
        
        self.cleanup()
    
    def cleanup(self):
        print("🧹 Sistem kapatılıyor...")
        
        try:
            self.cap.release()
            print("✅ Video girişi kapatıldı")
        except Exception as e:
            print(f"⚠️  Video girişi kapatma hatası: {e}")
            
        try:
            self.out.release()
            print("✅ Video çıkışı kapatıldı")
        except Exception as e:
            print(f"⚠️  Video çıkışı kapatma hatası: {e}")
            
        cv2.destroyAllWindows()
        
        # CSV kaydet
        try:
            write_csv(self.results, './test.csv')
            print("✅ Veriler test.csv dosyasına kaydedildi")
        except Exception as e:
            print(f"⚠️  CSV kaydetme hatası: {e}")
            
        print("🎯 Sistem başarıyla kapatıldı")

# Kullanım
if __name__ == "__main__":
    # Ayarları değiştir
    system = DetectionSystem(
        source=0,  # Kamera ID veya RTSP URL
        output_ip="192.168.1.100",  # Hedef cihaz IP
        output_port=5000  # Port
    )
    system.run()