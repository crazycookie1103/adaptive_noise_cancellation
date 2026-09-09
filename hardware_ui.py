import tkinter as tk
from tkinter import ttk
import random
import math

class CompleteHardwareBoxSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartAudio Tactical Hybrid AI-ANC Hardware UI Simulator")
        self.geometry("900x820")  # Fixed: Removed space in geometry string
        self.configure(bg="#121214")

        # Animation state initialization
        self.time_step = 0

        # Core Hardware State
        self.power_state = tk.BooleanVar(value=True)
        self.anc_mode = tk.StringVar(value="ANC: ON")
        self.dsp_bypass = tk.BooleanVar(value=False)
        self.battery_level = tk.IntVar(value=92)
        self.volume_level = tk.IntVar(value=65)
        self.noise_db = tk.DoubleVar(value=118.0)
        
        # Performance Evaluation Metrics (SIH Requirements)
        self.snr_val = tk.DoubleVar(value=18.4)      # Target > 15 dB
        self.stoi_val = tk.DoubleVar(value=0.89)     # Target > 0.85
        self.pesq_val = tk.DoubleVar(value=2.82)     # Target > 2.5
        self.latency_ms = tk.DoubleVar(value=3.8)    # Real-time low latency
        
        # TinyML State
        self.tinyml_class = tk.StringVar(value="Gunshots")
        self.tinyml_confidence = tk.IntVar(value=96)
        
        # Tactical Environmental Scenes
        self.scenes = {
            "Gunshots": {"db": 118.0, "class": "Gunshots", "hazard": True, "snr": 16.5, "stoi": 0.86, "pesq": 2.65},
            "Artillery Fire": {"db": 126.0, "class": "Artillery", "hazard": True, "snr": 15.2, "stoi": 0.85, "pesq": 2.52},
            "Helicopter Rotor": {"db": 98.0, "class": "Helicopter", "hazard": False, "snr": 19.8, "stoi": 0.91, "pesq": 2.95},
            "Armoured Vehicle": {"db": 92.0, "class": "Armoured Veh", "hazard": False, "snr": 21.4, "stoi": 0.93, "pesq": 3.12},
            "Emergency Siren": {"db": 95.0, "class": "Siren", "hazard": True, "snr": 18.1, "stoi": 0.88, "pesq": 2.78},
            "Quiet Speech": {"db": 42.0, "class": "Clean Speech", "hazard": False, "snr": 24.5, "stoi": 0.96, "pesq": 3.40}
        }
        self.env_scene = tk.StringVar(value="Gunshots")

        # UI Overlay & Hazard State
        self.overlay_counter = 0
        self.siren_flash = False

        self._build_gui()
        self._update_loop()

    def _build_gui(self):
        # Top Bar: Test Scene Selector
        top_bar = tk.Frame(self, bg="#1a1a1e", pady=8)
        top_bar.pack(fill="x")
        
        tk.Label(top_bar, text="Tactical Audio Test Scene:", fg="#aaaaaa", bg="#1a1a1e", font=("Helvetica", 9, "bold")).pack(side="left", padx=10)
        scene_combo = ttk.Combobox(top_bar, textvariable=self.env_scene, values=list(self.scenes.keys()), state="readonly", width=20)
        scene_combo.pack(side="left")
        scene_combo.bind("<<ComboboxSelected>>", self._on_scene_change)

        # Physical Hardware Box Enclosure
        self.box_frame = tk.Frame(self, bg="#26262e", bd=4, relief="raised", padx=25, pady=20)
        self.box_frame.pack(pady=15, padx=20)

        # Silkscreen Branding
        tk.Label(self.box_frame, text="SmartAudio Defence Hybrid AI-ANC Engine", fg="#888899", bg="#26262e", font=("Helvetica", 10, "bold")).pack(anchor="w")
        tk.Label(self.box_frame, text="MODEL: SA-TML100 • SIH 2026 DEFENCE REV B", fg="#555566", bg="#26262e", font=("Helvetica", 7)).pack(anchor="w", pady=(0, 10))

        # Top Hardware Row: LEDs & OLED Screen
        top_row = tk.Frame(self.box_frame, bg="#26262e")
        top_row.pack(fill="x", pady=5)

        # Status LEDs
        led_panel = tk.Frame(top_row, bg="#1e1e24", bd=2, relief="sunken", padx=10, pady=10)
        led_panel.pack(side="left", fill="y", padx=(0, 15))

        self.pwr_led = tk.Canvas(led_panel, width=16, height=16, bg="#1e1e24", highlightthickness=0)
        self.pwr_led.pack(pady=4)
        tk.Label(led_panel, text="PWR", fg="#888", bg="#1e1e24", font=("Helvetica", 7, "bold")).pack()

        self.dsp_led = tk.Canvas(led_panel, width=16, height=16, bg="#1e1e24", highlightthickness=0)
        self.dsp_led.pack(pady=4)
        tk.Label(led_panel, text="DSP", fg="#888", bg="#1e1e24", font=("Helvetica", 7, "bold")).pack()

        # Monochromatic OLED Display Simulation
        oled_bezel = tk.Frame(top_row, bg="#111111", bd=3, relief="sunken")
        oled_bezel.pack(side="left")

        self.oled = tk.Canvas(oled_bezel, width=440, height=230, bg="#020804", highlightthickness=0)
        self.oled.pack()

        # Bottom Hardware Row: Controls & Knob
        ctrl_panel = tk.Frame(self.box_frame, bg="#1e1e24", bd=2, relief="groove", pady=15, padx=15)
        ctrl_panel.pack(fill="x", pady=15)

        # Switches
        tk.Checkbutton(ctrl_panel, text="Power", variable=self.power_state,
                       command=self.draw_oled, bg="#1e1e24", fg="white", 
                       selectcolor="#333340", activebackground="#1e1e24", activeforeground="white").grid(row=0, column=0, padx=10)

        tk.Button(ctrl_panel, text="ANC Mode", command=self._toggle_anc, 
                  bg="#383845", fg="white", activebackground="#505060", activeforeground="white").grid(row=0, column=1, padx=10)

        tk.Checkbutton(ctrl_panel, text="Bypass DSP", variable=self.dsp_bypass,
                       command=self.draw_oled, bg="#1e1e24", fg="white", 
                       selectcolor="#333340", activebackground="#1e1e24", activeforeground="white").grid(row=0, column=2, padx=10)

        # Rotary Knob Volume Controller
        knob_frame = tk.Frame(ctrl_panel, bg="#1e1e24")
        knob_frame.grid(row=0, column=3, padx=15)
        tk.Label(knob_frame, text="Volume Knob", fg="white", bg="#1e1e24", font=("Helvetica", 8)).pack()
        
        vol_slider = ttk.Scale(knob_frame, from_=0, to=100, variable=self.volume_level, command=self._on_vol_change)
        vol_slider.pack()

        # Hardware Ports Visualizer
        io_frame = tk.Frame(self.box_frame, bg="#26262e")
        io_frame.pack(fill="x", pady=(5,0))
        tk.Label(io_frame, text="[ USB-C CHARGE ]    [ AUX OUT / HEADSET ]    [ MIC ARRAY IN ]", fg="#444455", bg="#26262e", font=("Courier", 8, "bold")).pack()

    def _draw_leds(self):
        self.pwr_led.delete("all")
        pwr_c = "#00ff66" if self.power_state.get() else "#222222"
        self.pwr_led.create_oval(1, 1, 15, 15, fill=pwr_c, outline="")

        self.dsp_led.delete("all")
        dsp_c = "#00aaff" if (not self.dsp_bypass.get() and self.power_state.get()) else "#222222"
        self.dsp_led.create_oval(1, 1, 15, 15, fill=dsp_c, outline="")

    def draw_oled(self):
        self.oled.delete("all")
        self._draw_leds()

        if not self.power_state.get():
            self.box_frame.configure(bg="#26262e")
            return

        # Volume Transient Pop-up Overlay
        if self.overlay_counter > 0:
            self.overlay_counter -= 1
            vol = self.volume_level.get()
            self.oled.create_rectangle(50, 30, 390, 200, fill="#00ffcc", outline="")
            self.oled.create_text(220, 70, text="VOLUME", fill="#000000", font=("Courier", 18, "bold"))
            self.oled.create_text(220, 115, text=f"{vol}%", fill="#000000", font=("Courier", 28, "bold"))
            self.oled.create_rectangle(70, 155, 370, 170, fill="#000000", outline="")
            fill_w = 70 + int((vol / 100.0) * 300)
            if fill_w > 70:
                self.oled.create_rectangle(72, 157, fill_w, 168, fill="#00ffcc", outline="")
            return

        # Header Bar
        self.oled.create_text(40, 15, text=f"BAT {self.battery_level.get()}%", fill="#00ffcc", font=("Courier", 10, "bold"))
        self.oled.create_text(370, 15, text=self.anc_mode.get(), fill="#00ffcc", font=("Courier", 10, "bold"))
        self.oled.create_line(10, 28, 430, 28, fill="#00ffcc")

        # --- REAL-TIME NOISE VS ANTI-WAVE CANCELATION VISUALIZER ---
        wave_center_y = 70
        self.oled.create_text(110, 38, text="ANTI-WAVE PHASE INVERSION", fill="#888888", font=("Courier", 7))
        
        for x in range(10, 220, 3):
            # Input Noise Wave (Cyan)
            noise_y = wave_center_y + int(math.sin((x + self.time_step) * 0.1) * 14)
            self.oled.create_line(x, noise_y, x + 2, noise_y, fill="#00ffcc", width=1)
            
            # Generated Inverted Anti-Phase Wave (Red) & Cancelled Signal (White Flat Line)
            if "ON" in self.anc_mode.get() and not self.dsp_bypass.get():
                anti_y = wave_center_y - int(math.sin((x + self.time_step) * 0.1) * 14)
                self.oled.create_line(x, anti_y, x + 2, anti_y, fill="#ff0055", width=1)
                
                # Flat Destructive Interference Line
                self.oled.create_line(10, wave_center_y, 220, wave_center_y, fill="#ffffff", width=2)

        # Dynamic Audio Frequency Spectrum (6 FFT Bars)
        for i in range(6):
            h = random.randint(5, int(min(55, self.noise_db.get() * 0.5)))
            x0 = 240 + (i * 15)
            self.oled.create_rectangle(x0, 105 - h, x0 + 10, 105, fill="#00ffcc", outline="")

        # Real-Time Acoustic & TinyML Classification Readouts
        db_val = self.noise_db.get()
        cls_val = self.tinyml_class.get()
        conf_val = self.tinyml_confidence.get()

        self.oled.create_text(375, 48, text=f"{db_val:.1f} dB", fill="#ffffff", font=("Courier", 13, "bold"))
        self.oled.create_text(370, 72, text=f"CLASS: {cls_val[:10]}", fill="#00ffcc", font=("Courier", 9, "bold"))
        self.oled.create_text(370, 92, text=f"CONF : {conf_val}%", fill="#00ffcc", font=("Courier", 9))

        # Divider Line
        self.oled.create_line(10, 118, 430, 118, fill="#00ffcc")

        # --- REAL-TIME SIH PERFORMANCE EVALUATION METRICS ZONE ---
        snr = self.snr_val.get() if ("ON" in self.anc_mode.get() and not self.dsp_bypass.get()) else 2.1
        stoi = self.stoi_val.get() if ("ON" in self.anc_mode.get() and not self.dsp_bypass.get()) else 0.42
        pesq = self.pesq_val.get() if ("ON" in self.anc_mode.get() and not self.dsp_bypass.get()) else 1.10
        lat = self.latency_ms.get()

        self.oled.create_text(110, 133, text=f"SNR : {snr:.1f} dB (>15dB)", fill="#00ffcc", font=("Courier", 9, "bold"))
        self.oled.create_text(110, 150, text=f"STOI: {stoi:.2f} (>0.85)", fill="#00ffcc", font=("Courier", 9, "bold"))
        self.oled.create_text(320, 133, text=f"PESQ: {pesq:.2f} (>2.5)", fill="#00ffcc", font=("Courier", 9, "bold"))
        self.oled.create_text(320, 150, text=f"LATENCY: {lat:.1f} ms", fill="#00ffcc", font=("Courier", 9, "bold"))

        # Footer Status
        dsp_str = "BYPASS" if self.dsp_bypass.get() else "HYBRID-AI"
        self.oled.create_line(10, 168, 430, 168, fill="#00ffcc")
        self.oled.create_text(85, 190, text=f"DSP: {dsp_str}", fill="#ffffff", font=("Courier", 9))
        self.oled.create_text(230, 190, text="MIC ARRAY: ACTIVE", fill="#ffffff", font=("Courier", 8))
        self.oled.create_text(370, 190, text=f"VOL: {self.volume_level.get()}%", fill="#ffffff", font=("Courier", 9))

        # Hazard Alert Flash Effect
        current_scene_info = self.scenes.get(self.env_scene.get(), {})
        if current_scene_info.get("hazard", False) and "ON" in self.anc_mode.get():
            self.siren_flash = not self.siren_flash
            flash_color = "#771111" if self.siren_flash else "#26262e"
            self.box_frame.configure(bg=flash_color)
            self.oled.create_rectangle(230, 32, 430, 112, fill="#ff0000", outline="")
            self.oled.create_text(330, 72, text=f"SAFETY PASS-THROUGH\n{cls_val.upper()}", fill="#ffffff", font=("Courier", 9, "bold"))
        else:
            self.box_frame.configure(bg="#26262e")

    def _toggle_anc(self):
        modes = ["ANC: ON", "ANC: OFF", "ANC: AMBIENT"]
        idx = modes.index(self.anc_mode.get())
        self.anc_mode.set(modes[(idx + 1) % len(modes)])
        self.draw_oled()

    def _on_vol_change(self, val):
        self.overlay_counter = 4
        self.draw_oled()

    def _on_scene_change(self, event):
        scene = self.scenes[self.env_scene.get()]
        self.noise_db.set(scene["db"])
        self.tinyml_class.set(scene["class"])
        self.snr_val.set(scene["snr"])
        self.stoi_val.set(scene["stoi"])
        self.pesq_val.set(scene["pesq"])
        self.tinyml_confidence.set(random.randint(92, 99))
        self.draw_oled()

    def _update_loop(self):
        if self.power_state.get():
            # Step phase timer for smooth wave scrolling animation
            self.time_step = (self.time_step + 4) % 360
            
            # Subtle real-time dB fluctuation
            cur_db = self.noise_db.get()
            self.noise_db.set(max(30.0, min(130.0, cur_db + random.uniform(-0.8, 0.8))))
            
            # Subtle metrics jitter
            if "ON" in self.anc_mode.get() and not self.dsp_bypass.get():
                self.snr_val.set(max(15.0, min(25.0, self.snr_val.get() + random.uniform(-0.05, 0.05))))
                self.latency_ms.set(max(2.5, min(5.0, 3.8 + random.uniform(-0.2, 0.2))))

            self.draw_oled()

        self.after(100, self._update_loop)

if __name__ == "__main__":
    app = CompleteHardwareBoxSimulator()
    app.mainloop()