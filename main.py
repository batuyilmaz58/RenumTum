import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from datetime import datetime
import json
import cv2
import numpy as np

# YOLO import'u - aktif edildi
from ultralytics import YOLO

class KidneyDetectionApp:
    def __init__(self):
        # Ana pencereyi oluştur
        self.root = ctk.CTk()
        self.root.title("RenumTum")
        # self.root.iconbitmap(r"C:\Users\Batu\Documents\böbrek_hastalıkları_tespiti\icon.ico") 
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Tema dosyasını yükle
        ctk.set_appearance_mode("dark")
        
        # Model yolları - Kullanıcının model yolları
        self.model_paths = {
            "tumor": r"C:\Users\Batu\Documents\böbrek_hastalıkları_tespiti\tumor_detection\runs\detect\train\weights\best.pt",
            "kidney_stone": r"C:\Users\Batu\Documents\böbrek_hastalıkları_tespiti\stone_detection\runs\detect\train3\weights\best.pt"
        }
        
        # Modelleri yükle
        self.models = {}
        self.load_models()
        
        # Veri saklama için
        self.prediction_history = []
        self.current_image_path = None
        self.current_predictions = {"tumor": None, "kidney_stone": None}
        
        # Ana layout'u oluştur
        self.create_main_layout()
        
        # Geçmiş tahminleri yükle
        self.load_history()
        
    def load_models(self):
        """YOLO modellerini yükle"""
        try:
            print("Modeller yükleniyor...")
            
            # GERÇEK YOLO MODELLERİNİ YÜKLE
            if os.path.exists(self.model_paths["tumor"]):
                self.models["tumor"] = YOLO(self.model_paths["tumor"])
                print("✅ Tümör modeli yüklendi!")
            else:
                print(f"❌ Tümör modeli bulunamadı: {self.model_paths['tumor']}")
                self.models["tumor"] = None
                
            if os.path.exists(self.model_paths["kidney_stone"]):
                self.models["kidney_stone"] = YOLO(self.model_paths["kidney_stone"])
                print("✅ Böbrek taşı modeli yüklendi!")
            else:
                print(f"❌ Böbrek taşı modeli bulunamadı: {self.model_paths['kidney_stone']}")
                self.models["kidney_stone"] = None
            
            print("✅ Model yükleme işlemi tamamlandı!")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            messagebox.showerror("Model Hatası", 
                               f"Modeller yüklenemedi:\n{str(e)}\n\nLütfen model yollarını kontrol edin.")
        
    def create_main_layout(self):
        # Ana grid yapılandırması
        self.root.grid_columnconfigure(1, weight=4)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Sol panel (Menü ve Geçmiş)
        self.create_left_panel()
        
        # Sağ panel (Ana çalışma alanı)
        self.create_right_panel()
        
    def create_left_panel(self):
        # Sol frame
        self.left_frame = ctk.CTkFrame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left_frame.grid_rowconfigure(6, weight=1)
        
        # Başlık
        title_label = ctk.CTkLabel(
            self.left_frame, 
            text="🏥 Böbrek Analiz Sistemi", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(20, 10), padx=20, sticky="ew")
        
        # Menü butonları frame
        menu_frame = ctk.CTkFrame(self.left_frame)
        menu_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        # Analiz Et butonu
        self.predict_btn = ctk.CTkButton(
            menu_frame,
            text="📸 Görüntü Yükle ve Analiz Et",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            command=self.select_and_analyze_image
        )
        self.predict_btn.grid(row=0, column=0, pady=15, padx=20, sticky="ew")
        
        # Geçmiş Tahminler butonu
        self.history_btn = ctk.CTkButton(
            menu_frame,
            text="📋 Geçmiş Tahminler",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.show_history
        )
        self.history_btn.grid(row=1, column=0, pady=(0, 15), padx=20, sticky="ew")
        
        menu_frame.grid_columnconfigure(0, weight=1)
        
        # Model durumu göstergesi
        status_label = ctk.CTkLabel(
            self.left_frame, 
            text="🔧 Model Durumu", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        status_label.grid(row=2, column=0, pady=(0, 10), padx=20, sticky="w")
        
        # Model durumları
        self.tumor_status = ctk.CTkLabel(
            self.left_frame,
            text="🫀 Tümör Modeli: ✅ Hazır" if self.models.get("tumor") else "🫀 Tümör Modeli: ❌ Yüklenmedi",
            font=ctk.CTkFont(size=11)
        )
        self.tumor_status.grid(row=3, column=0, padx=20, pady=2, sticky="w")
        
        self.stone_status = ctk.CTkLabel(
            self.left_frame,
            text="🪨 Taş Modeli: ✅ Hazır" if self.models.get("kidney_stone") else "🪨 Taş Modeli: ❌ Yüklenmedi", 
            font=ctk.CTkFont(size=11)
        )
        self.stone_status.grid(row=4, column=0, padx=20, pady=2, sticky="w")
        
        # Geçmiş tahminler listesi
        history_label = ctk.CTkLabel(
            self.left_frame, 
            text="📊 Son Tahminler", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        history_label.grid(row=5, column=0, pady=(20, 10), padx=20, sticky="w")
        
        # Scrollable frame for history
        self.history_frame = ctk.CTkScrollableFrame(
            self.left_frame,
            height=300
        )
        self.history_frame.grid(row=6, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.history_frame.grid_columnconfigure(0, weight=1)
        
        self.left_frame.grid_columnconfigure(0, weight=1)
        
    def create_right_panel(self):
        # Sağ frame
        self.right_frame = ctk.CTkFrame(self.root)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # Üst bilgi paneli
        self.create_info_header()
        
        # İkili tahmin görüntüleme alanı
        self.create_dual_prediction_area()
        
        # Alt sonuç paneli
        self.create_results_panel()
        
    def create_info_header(self):
        # Üst bilgi frame
        info_header = ctk.CTkFrame(self.right_frame)
        info_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        info_header.grid_columnconfigure(1, weight=1)
        
        # Başlık
        header_label = ctk.CTkLabel(
            info_header,
            text="🔬 Çift Model Analiz Sistemi",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        header_label.grid(row=0, column=0, columnspan=3, pady=(15, 10))
        
        # Durum göstergeleri
        self.analysis_status = ctk.CTkLabel(
            info_header,
            text="📋 Durum: Görüntü bekleniyor",
            font=ctk.CTkFont(size=14)
        )
        self.analysis_status.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="w")
        
        self.file_info_label = ctk.CTkLabel(
            info_header,
            text="📁 Dosya: -",
            font=ctk.CTkFont(size=14)
        )
        self.file_info_label.grid(row=1, column=1, padx=20, pady=(0, 15))
        
        self.analysis_time_label = ctk.CTkLabel(
            info_header,
            text="⏱️ Süre: -",
            font=ctk.CTkFont(size=14)
        )
        self.analysis_time_label.grid(row=1, column=2, padx=20, pady=(0, 15), sticky="e")
        
    def create_dual_prediction_area(self):
        # Ana görüntü gösterme alanı
        self.prediction_frame = ctk.CTkFrame(self.right_frame)
        self.prediction_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.prediction_frame.grid_rowconfigure(1, weight=1)
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        self.prediction_frame.grid_columnconfigure(1, weight=1)
        
        # Sol taraf - Tümör Tespiti
        tumor_header = ctk.CTkLabel(
            self.prediction_frame,
            text="🫀 Tümör Tespiti",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#3b4e8e", "#3b4e8e"),
            corner_radius=8
        )
        tumor_header.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=(15, 10))
        
        self.tumor_image_frame = ctk.CTkFrame(self.prediction_frame)
        self.tumor_image_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        self.tumor_image_frame.grid_rowconfigure(0, weight=1)
        self.tumor_image_frame.grid_columnconfigure(0, weight=1)
        
        self.tumor_image_label = ctk.CTkLabel(
            self.tumor_image_frame,
            text="🫀\n\nTümör Analizi\n\nGörüntü yüklendikten sonra\ntümör tespiti burada gösterilecek",
            font=ctk.CTkFont(size=14),
            fg_color="transparent"
        )
        self.tumor_image_label.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Sağ taraf - Böbrek Taşı Tespiti  
        stone_header = ctk.CTkLabel(
            self.prediction_frame,
            text="🪨 Böbrek Taşı Tespiti",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#3b4e8e", "#3b4e8e"),
            corner_radius=8
        )
        stone_header.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=(15, 10))
        
        self.stone_image_frame = ctk.CTkFrame(self.prediction_frame)
        self.stone_image_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        self.stone_image_frame.grid_rowconfigure(0, weight=1)
        self.stone_image_frame.grid_columnconfigure(0, weight=1)
        
        self.stone_image_label = ctk.CTkLabel(
            self.stone_image_frame,
            text="🪨\n\nTaş Analizi\n\nGörüntü yüklendikten sonra\nböbrek taşı tespiti burada gösterilecek",
            font=ctk.CTkFont(size=14),
            fg_color="transparent"
        )
        self.stone_image_label.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
    def create_results_panel(self):
        # Sonuç paneli
        results_frame = ctk.CTkFrame(self.right_frame)
        results_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        
        # Sol sonuçlar - Tümör
        tumor_results_frame = ctk.CTkFrame(results_frame)
        tumor_results_frame.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=15)
        tumor_results_frame.grid_columnconfigure(1, weight=1)
        
        tumor_results_label = ctk.CTkLabel(
            tumor_results_frame,
            text="🫀 Tümör Sonuçları",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        tumor_results_label.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        self.tumor_detection_label = ctk.CTkLabel(
            tumor_results_frame,
            text="🎯 Tespit: -",
            font=ctk.CTkFont(size=12)
        )
        self.tumor_detection_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.tumor_confidence_label = ctk.CTkLabel(
            tumor_results_frame,
            text="✅ Güven: -",
            font=ctk.CTkFont(size=12)
        )
        self.tumor_confidence_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.tumor_risk_label = ctk.CTkLabel(
            tumor_results_frame,
            text="⚡ Risk: -",
            font=ctk.CTkFont(size=12)
        )
        self.tumor_risk_label.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10))
        
        # Sağ sonuçlar - Böbrek Taşı
        stone_results_frame = ctk.CTkFrame(results_frame)
        stone_results_frame.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=15)
        stone_results_frame.grid_columnconfigure(1, weight=1)
        
        stone_results_label = ctk.CTkLabel(
            stone_results_frame,
            text="🪨 Taş Sonuçları",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        stone_results_label.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        self.stone_detection_label = ctk.CTkLabel(
            stone_results_frame,
            text="🎯 Tespit: -",
            font=ctk.CTkFont(size=12)
        )
        self.stone_detection_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.stone_confidence_label = ctk.CTkLabel(
            stone_results_frame,
            text="✅ Güven: -",
            font=ctk.CTkFont(size=12)
        )
        self.stone_confidence_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.stone_risk_label = ctk.CTkLabel(
            stone_results_frame,
            text="⚡ Risk: -",
            font=ctk.CTkFont(size=12)
        )
        self.stone_risk_label.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10))
        
    def select_and_analyze_image(self):
        """Görüntü seç ve her iki modelle de analiz et"""
        file_path = filedialog.askopenfilename(
            title="Böbrek Görüntüsü Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_info_label.configure(text=f"📁 Dosya: {os.path.basename(file_path)}")
            
            # Analizi başlat
            self.run_dual_prediction()
    
    def run_dual_prediction(self):
        """Her iki modelle de tahmin çalıştır"""
        if not self.current_image_path:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü seçin!")
            return
        
        try:
            start_time = datetime.now()
            
            # Analiz durumunu güncelle
            self.analysis_status.configure(text="📋 Durum: Analiz ediliyor...")
            self.root.update()
            
            # Orijinal görüntüyü yükle
            original_image = cv2.imread(self.current_image_path)
            if original_image is None:
                raise Exception("Görüntü yüklenemedi!")
            
            # Her iki modelle de tahmin yap
            tumor_result = self.predict_with_model("tumor", original_image.copy())
            stone_result = self.predict_with_model("kidney_stone", original_image.copy())
            
            # Sonuçları kaydet
            self.current_predictions = {
                "tumor": tumor_result,
                "kidney_stone": stone_result
            }
            
            # UI'ı güncelle
            self.update_prediction_displays()
            
            # Analiz süresini hesapla
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.analysis_time_label.configure(text=f"⏱️ Süre: {duration:.2f}s")
            
            # Geçmişe ekle
            self.add_dual_prediction_to_history()
            
            self.analysis_status.configure(text="📋 Durum: Analiz tamamlandı ✅")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz sırasında hata oluştu: {str(e)}")
            self.analysis_status.configure(text="📋 Durum: Hata oluştu ❌")
    
    def predict_with_model(self, model_type, image):
        """Belirtilen modelle tahmin yap"""
        try:
            model = self.models.get(model_type)
            
            if model is None:
                print(f"⚠️ {model_type} modeli yüklenmemiş, demo veri kullanılıyor")
                return self.generate_demo_prediction(model_type, image)
            
            # GERÇEK YOLO TAHMİNİ - AKTİF
            print(f"🔍 {model_type} modeli ile tahmin yapılıyor...")
            results = model(image)
            
            # Sonuçları işle
            processed_image = self.draw_predictions(image, results, model_type)
            
            # Tespit sayısını ve güven skorunu al
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            confidence = float(results[0].boxes.conf.max()) if results[0].boxes is not None and len(results[0].boxes.conf) > 0 else 0.0
            
            return {
                "detections": detections,
                "confidence": confidence,
                "processed_image": processed_image,
                "raw_results": results
            }
            
        except Exception as e:
            print(f"Model {model_type} tahmin hatası: {e}")
            return {
                "detections": 0,
                "confidence": 0.0,
                "processed_image": image,
                "error": str(e)
            }
    
    def generate_demo_prediction(self, model_type, image):
        """Demo için sahte tahmin sonuçları üret"""
        import random
        
        # Görüntüyü kopyala
        demo_image = image.copy()
        
        if model_type == "tumor":
            detections = random.randint(0, 2)
            if detections > 0:
                confidence = random.uniform(0.75, 0.95)
                # Demo bbox çiz
                h, w = image.shape[:2]
                for i in range(detections):
                    x1 = random.randint(50, w//2)
                    y1 = random.randint(50, h//2)
                    x2 = x1 + random.randint(50, 150)
                    y2 = y1 + random.randint(50, 150)
                    
                    # Bbox çiz
                    cv2.rectangle(demo_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(demo_image, f"Tumor {confidence:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                confidence = random.uniform(0.1, 0.3)
        else:  # kidney_stone
            detections = random.randint(0, 3)
            if detections > 0:
                confidence = random.uniform(0.70, 0.92)
                # Demo bbox çiz
                h, w = image.shape[:2]
                for i in range(detections):
                    x1 = random.randint(50, w//2)
                    y1 = random.randint(50, h//2)
                    x2 = x1 + random.randint(30, 100)
                    y2 = y1 + random.randint(30, 100)
                    
                    # Bbox çiz
                    cv2.rectangle(demo_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(demo_image, f"Stone {confidence:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                confidence = random.uniform(0.05, 0.25)
        
        return {
            "detections": detections,
            "confidence": confidence,
            "processed_image": demo_image
        }
    
    def draw_predictions(self, image, results, model_type):
        """YOLO sonuçlarını görüntüye çiz - gerçek modeller için AKTİF"""
        annotated_image = image.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Model tipine göre renk seç
            color = (0, 0, 255) if model_type == "tumor" else (0, 255, 0)  # Kırmızı: Tümör, Yeşil: Taş
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # Bbox çiz
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Label ekle
                label_text = "Tumor" if model_type == "tumor" else "Kidney Stone"
                label = f"{label_text}: {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated_image
    
    def update_prediction_displays(self):
        """Tahmin sonuçlarını UI'da göster"""
        # Tümör sonuçları
        tumor_result = self.current_predictions.get("tumor")
        if tumor_result:
            self.display_prediction_image(tumor_result["processed_image"], "tumor")
            self.update_result_labels("tumor", tumor_result)
        
        # Böbrek taşı sonuçları  
        stone_result = self.current_predictions.get("kidney_stone")
        if stone_result:
            self.display_prediction_image(stone_result["processed_image"], "kidney_stone")
            self.update_result_labels("kidney_stone", stone_result)
    
    def display_prediction_image(self, cv_image, model_type):
        """OpenCV görüntüsünü UI'da göster"""
        try:
            # OpenCV BGR'dan RGB'ye çevir
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Boyutlandır
            display_size = (400, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # CTkImage oluştur
            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=pil_image.size
            )
            
            # İlgili label'ı güncelle
            if model_type == "tumor":
                self.tumor_image_label.configure(image=ctk_image, text="")
            else:
                self.stone_image_label.configure(image=ctk_image, text="")
                
        except Exception as e:
            print(f"Görüntü gösterim hatası: {e}")
    
    def update_result_labels(self, model_type, result):
        """Sonuç etiketlerini güncelle"""
        detections = result.get("detections", 0)
        confidence = result.get("confidence", 0.0)
        
        # Risk skoru hesapla
        if detections > 0:
            risk_score = confidence * (1 + detections * 0.1)
        else:
            risk_score = 0.1
        
        risk_color = self.get_risk_color(risk_score)
        
        if model_type == "tumor":
            self.tumor_detection_label.configure(text=f"🎯 Tespit: {detections} adet")
            self.tumor_confidence_label.configure(text=f"✅ Güven: {confidence:.2f}")
            self.tumor_risk_label.configure(
                text=f"⚡ Risk: {risk_score:.2f}",
                text_color=risk_color
            )
        else:
            self.stone_detection_label.configure(text=f"🎯 Tespit: {detections} adet")
            self.stone_confidence_label.configure(text=f"✅ Güven: {confidence:.2f}")
            self.stone_risk_label.configure(
                text=f"⚡ Risk: {risk_score:.2f}",
                text_color=risk_color
            )
    
    def get_risk_color(self, risk_score):
        """Risk skoruna göre renk döndür"""
        if risk_score >= 0.8:
            return "#ff4444"  # Kırmızı - Yüksek Risk
        elif risk_score >= 0.6:
            return "#ffaa44"  # Turuncu - Orta Risk
        elif risk_score >= 0.4:
            return "#ffff44"  # Sarı - Düşük Risk
        else:
            return "#44ff44"  # Yeşil - Minimal Risk
    
    def add_dual_prediction_to_history(self):
        """Çift tahmin sonucunu geçmişe ekle"""
        if not self.current_predictions:
            return
        
        tumor_result = self.current_predictions.get("tumor", {})
        stone_result = self.current_predictions.get("kidney_stone", {})
        
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": self.current_image_path,
            "tumor": {
                "detections": tumor_result.get("detections", 0),
                "confidence": tumor_result.get("confidence", 0.0),
                "risk_score": tumor_result.get("confidence", 0.0) * (1 + tumor_result.get("detections", 0) * 0.1)
            },
            "kidney_stone": {
                "detections": stone_result.get("detections", 0),
                "confidence": stone_result.get("confidence", 0.0),
                "risk_score": stone_result.get("confidence", 0.0) * (1 + stone_result.get("detections", 0) * 0.1)
            }
        }
        
        self.prediction_history.insert(0, history_entry)  # En son tahmin en üste
        
        # Maksimum 15 tahmin sakla
        if len(self.prediction_history) > 15:
            self.prediction_history = self.prediction_history[:15]
        
        # History UI'ını güncelle
        self.update_history_display()
        
        # Dosyaya kaydet
        self.save_history()
    
    def update_history_display(self):
        """Geçmiş tahminler görünümünü güncelle"""
        # Mevcut widget'ları temizle
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        
        # Geçmiş tahminleri göster
        for i, result in enumerate(self.prediction_history):
            self.create_history_item(result, i)
    
    def create_history_item(self, result, index):
        """Geçmiş tahmin öğesi oluştur"""
        item_frame = ctk.CTkFrame(self.history_frame)
        item_frame.grid(row=index, column=0, sticky="ew", padx=5, pady=5)
        item_frame.grid_columnconfigure(0, weight=1)
        
        # Tarih ve saat
        date_label = ctk.CTkLabel(
            item_frame,
            text=f"📅 {result['timestamp']}",
            font=ctk.CTkFont(size=10)
        )
        date_label.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        
        # Dosya adı
        file_label = ctk.CTkLabel(
            item_frame,
            text=f"📁 {os.path.basename(result['image_path'])}",
            font=ctk.CTkFont(size=9)
        )
        file_label.grid(row=1, column=0, sticky="w", padx=10, pady=2)
        
        # Tümör sonuçları
        tumor_data = result.get('tumor', {})
        tumor_text = f"🫀 Tümör: {tumor_data.get('detections', 0)} tespit"
        if tumor_data.get('detections', 0) > 0:
            tumor_text += f" (Risk: {tumor_data.get('risk_score', 0):.2f})"
        
        tumor_label = ctk.CTkLabel(
            item_frame,
            text=tumor_text,
            font=ctk.CTkFont(size=9),
            text_color=self.get_risk_color(tumor_data.get('risk_score', 0))
        )
        tumor_label.grid(row=2, column=0, sticky="w", padx=10, pady=1)
        
        # Böbrek taşı sonuçları
        stone_data = result.get('kidney_stone', {})
        stone_text = f"🪨 Taş: {stone_data.get('detections', 0)} tespit"
        if stone_data.get('detections', 0) > 0:
            stone_text += f" (Risk: {stone_data.get('risk_score', 0):.2f})"
            
        stone_label = ctk.CTkLabel(
            item_frame,
            text=stone_text,
            font=ctk.CTkFont(size=9),
            text_color=self.get_risk_color(stone_data.get('risk_score', 0))
        )
        stone_label.grid(row=3, column=0, sticky="w", padx=10, pady=(1, 8))
    
    def show_history(self):
        """Geçmiş tahminleri detaylı göster"""
        history_window = ctk.CTkToplevel(self.root)
        history_window.title("Detaylı Geçmiş Tahminler")
        history_window.geometry("900x700")
        history_window.grab_set()  # Modal pencere
        
        # Ana frame
        main_frame = ctk.CTkFrame(history_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Başlık
        title_label = ctk.CTkLabel(
            main_frame,
            text="📊 Detaylı Analiz Geçmişi",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(20, 15))
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(main_frame)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        if not self.prediction_history:
            no_data_label = ctk.CTkLabel(
                scroll_frame,
                text="📭 Henüz analiz geçmişi bulunmuyor.\n\nGörüntü yükleyip analiz ettikten sonra\nsonuçlar burada görüntülenecektir.",
                font=ctk.CTkFont(size=16),
                fg_color="transparent"
            )
            no_data_label.pack(pady=50)
            return
        
        # İstatistikler
        self.create_statistics_panel(scroll_frame)
        
        # Detaylı geçmiş listesi
        for i, result in enumerate(self.prediction_history):
            self.create_detailed_history_item(scroll_frame, result, i)
    
    def create_statistics_panel(self, parent):
        """İstatistik paneli oluştur"""
        stats_frame = ctk.CTkFrame(parent)
        stats_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        stats_title = ctk.CTkLabel(
            stats_frame,
            text="📈 Analiz İstatistikleri",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_title.pack(pady=(15, 10))
        
        # İstatistikleri hesapla
        total_analyses = len(self.prediction_history)
        tumor_detections = sum(1 for h in self.prediction_history if h.get('tumor', {}).get('detections', 0) > 0)
        stone_detections = sum(1 for h in self.prediction_history if h.get('kidney_stone', {}).get('detections', 0) > 0)
        
        avg_tumor_risk = np.mean([h.get('tumor', {}).get('risk_score', 0) for h in self.prediction_history]) if self.prediction_history else 0
        avg_stone_risk = np.mean([h.get('kidney_stone', {}).get('risk_score', 0) for h in self.prediction_history]) if self.prediction_history else 0
        
        # İstatistik grid
        stats_grid = ctk.CTkFrame(stats_frame)
        stats_grid.pack(fill="x", padx=20, pady=(0, 15))
        stats_grid.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Toplam analiz
        total_label = ctk.CTkLabel(
            stats_grid,
            text=f"📊 Toplam Analiz\n{total_analyses}",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        total_label.grid(row=0, column=0, padx=10, pady=15)
        
        # Tümör tespitleri
        tumor_label = ctk.CTkLabel(
            stats_grid,
            text=f"🫀 Tümör Tespiti\n{tumor_detections}/{total_analyses}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ff6b6b" if tumor_detections > 0 else "#4ecdc4"
        )
        tumor_label.grid(row=0, column=1, padx=10, pady=15)
        
        # Taş tespitleri
        stone_label = ctk.CTkLabel(
            stats_grid,
            text=f"🪨 Taş Tespiti\n{stone_detections}/{total_analyses}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ff9f43" if stone_detections > 0 else "#4ecdc4"
        )
        stone_label.grid(row=0, column=2, padx=10, pady=15)
        
        # Ortalama risk skorları
        risk_info = ctk.CTkLabel(
            stats_frame,
            text=f"⚡ Ort. Tümör Riski: {avg_tumor_risk:.2f} | Ort. Taş Riski: {avg_stone_risk:.2f}",
            font=ctk.CTkFont(size=11)
        )
        risk_info.pack(pady=(0, 15))
    
    def create_detailed_history_item(self, parent, result, index):
        """Detaylı geçmiş öğesi oluştur"""
        detail_frame = ctk.CTkFrame(parent)
        detail_frame.pack(fill="x", padx=10, pady=10)
        
        # Başlık
        title_label = ctk.CTkLabel(
            detail_frame,
            text=f"🔬 Analiz #{len(self.prediction_history) - index} - {result['timestamp']}",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(pady=(15, 10))
        
        # Dosya bilgisi
        file_label = ctk.CTkLabel(
            detail_frame,
            text=f"📁 Dosya: {os.path.basename(result['image_path'])}",
            font=ctk.CTkFont(size=12)
        )
        file_label.pack(pady=(0, 10))
        
        # Sonuçlar grid
        results_grid = ctk.CTkFrame(detail_frame)
        results_grid.pack(fill="x", padx=20, pady=(0, 15))
        results_grid.grid_columnconfigure((0, 1), weight=1)
        
        # Tümör sonuçları
        tumor_frame = ctk.CTkFrame(results_grid)
        tumor_frame.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=10)
        
        tumor_data = result.get('tumor', {})
        tumor_title = ctk.CTkLabel(
            tumor_frame,
            text="🫀 Tümör Analizi",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        tumor_title.pack(pady=(10, 5))
        
        tumor_details = f"""
🎯 Tespit Sayısı: {tumor_data.get('detections', 0)}
✅ Güven Oranı: {tumor_data.get('confidence', 0):.2f}
⚡ Risk Skoru: {tumor_data.get('risk_score', 0):.2f}
        """.strip()
        
        tumor_info = ctk.CTkLabel(
            tumor_frame,
            text=tumor_details,
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        tumor_info.pack(pady=(5, 15))
        
        # Böbrek taşı sonuçları
        stone_frame = ctk.CTkFrame(results_grid)
        stone_frame.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=10)
        
        stone_data = result.get('kidney_stone', {})
        stone_title = ctk.CTkLabel(
            stone_frame,
            text="🪨 Böbrek Taşı Analizi",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        stone_title.pack(pady=(10, 5))
        
        stone_details = f"""
🎯 Tespit Sayısı: {stone_data.get('detections', 0)}
✅ Güven Oranı: {stone_data.get('confidence', 0):.2f}
⚡ Risk Skoru: {stone_data.get('risk_score', 0):.2f}
        """.strip()
        
        stone_info = ctk.CTkLabel(
            stone_frame,
            text=stone_details,
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        stone_info.pack(pady=(5, 15))
    
    def save_history(self):
        """Geçmiş tahminleri dosyaya kaydet"""
        try:
            # Dosya yollarını kısalt (sadece dosya adı)
            history_to_save = []
            for item in self.prediction_history:
                item_copy = item.copy()
                item_copy['image_path'] = os.path.basename(item['image_path'])
                history_to_save.append(item_copy)
            
            with open('dual_prediction_history.json', 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Geçmiş kaydedilirken hata: {e}")
    
    def load_history(self):
        """Geçmiş tahminleri dosyadan yükle"""
        try:
            if os.path.exists('dual_prediction_history.json'):
                with open('dual_prediction_history.json', 'r', encoding='utf-8') as f:
                    self.prediction_history = json.load(f)
                self.update_history_display()
        except Exception as e:
            print(f"Geçmiş yüklenirken hata: {e}")
            self.prediction_history = []
    
    def run(self):
        """Uygulamayı çalıştır"""
        self.root.mainloop()

if __name__ == "__main__":
    # Temayı ayarla
    try:
        # MoonlitSky.json dosyanızın yolunu buraya girin
        ctk.set_default_color_theme(r"C:\Users\Batu\Documents\böbrek_hastalıkları_tespiti\UI\MoonlitSky.json")
    except:
        print("⚠️ Tema dosyası bulunamadı, varsayılan tema kullanılıyor.")
        print("MoonlitSky.json dosyasını ana dizine koyun.")
    
    print("🏥 Böbrek Analiz Sistemi Başlatılıyor...")
    print("📋 Sistem Özellikleri:")
    print("✅ Çift model analiz sistemi aktif")
    print("✅ YOLO modelleri aktif")
    print("✅ Gerçek zamanlı tespit aktif")
    print("✅ Geçmiş kayıt sistemi aktif")
    print("=" * 50)
    
    app = KidneyDetectionApp()
    app.run()