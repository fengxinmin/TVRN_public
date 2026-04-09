import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

class DSCQSVideoEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("DSCQS Video Quality Evaluation Tool")
        
        self.base_path_1 = "./TVRN"
        self.base_path_2 = "./GIMM_VFI"
        
        self.sequence_folders = []
        self.current_sequence_index = -1
        
        self.video1_frames = []
        self.video2_frames = []
        
        self.current_frame = 0
        self.max_frames = 0
        
        self.playing = False
        self.play_thread = None
        self.stop_event = threading.Event()
        
        self.create_widgets()
        self.scan_sequences()
        
    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)
        
        self.seq_label = tk.Label(top_frame, text="Current Sequence: None", font=("Arial", 12, "bold"))
        self.seq_label.pack(side=tk.LEFT, padx=10)
        
        self.prev_seq_btn = tk.Button(top_frame, text="< Prev Seq", command=self.prev_sequence)
        self.prev_seq_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_seq_btn = tk.Button(top_frame, text="Next Seq >", command=self.next_sequence)
        self.next_seq_btn.pack(side=tk.LEFT, padx=5)
        
        video_frame = tk.Frame(self.root)
        video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        left_frame = tk.LabelFrame(video_frame, text="Video 1", padx=5, pady=5)
        left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.left_label = tk.Label(left_frame)
        self.left_label.pack()
        
        right_frame = tk.LabelFrame(video_frame, text="Video 2", padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.right_label = tk.Label(right_frame)
        self.right_label.pack()
        
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="<<", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="<", command=self.step_backward).pack(side=tk.LEFT, padx=5)
        
        self.play_button = tk.Button(control_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text=">", command=self.step_forward).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text=">>", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.IntVar()
        self.progress_scale = tk.Scale(
            control_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.progress_var,
            command=self.on_progress_change,
            state=tk.DISABLED
        )
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.frame_label = tk.Label(control_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        score_frame = tk.Frame(self.root)
        score_frame.pack(pady=10)
        
        tk.Label(score_frame, text="Video 1 Score (1-5):").grid(row=0, column=0, padx=5)
        self.score1_var = tk.StringVar(value="3")
        score1_combo = ttk.Combobox(score_frame, textvariable=self.score1_var, values=["1", "2", "3", "4", "5"], width=5)
        score1_combo.grid(row=0, column=1, padx=5)
        
        tk.Label(score_frame, text="Video 2 Score (1-5):").grid(row=0, column=2, padx=5)
        self.score2_var = tk.StringVar(value="3")
        score2_combo = ttk.Combobox(score_frame, textvariable=self.score2_var, values=["1", "2", "3", "4", "5"], width=5)
        score2_combo.grid(row=0, column=3, padx=5)
        
        tk.Button(score_frame, text="Save Score", command=self.save_scores).grid(row=0, column=4, padx=10)
        
        self.result_text = tk.Text(self.root, height=8, width=80)
        self.result_text.pack(pady=10)
        
    def scan_sequences(self):
        if not os.path.exists(self.base_path_1) or not os.path.exists(self.base_path_2):
            messagebox.showerror("Path Error", f"Directories not found:\n{self.base_path_1}\n{self.base_path_2}")
            return

        dirs1 = set([d for d in os.listdir(self.base_path_1) if os.path.isdir(os.path.join(self.base_path_1, d))])
        dirs2 = set([d for d in os.listdir(self.base_path_2) if os.path.isdir(os.path.join(self.base_path_2, d))])
        
        common_dirs = sorted(list(dirs1.intersection(dirs2)), reverse=True)
        
        if not common_dirs:
            messagebox.showwarning("No Sequences", "No common subfolders found between TVRN and GIMM_VFI.")
            return
            
        self.sequence_folders = common_dirs
        self.current_sequence_index = 0
        self.load_current_sequence()

    def load_current_sequence(self):
        if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.sequence_folders):
            return
            
        seq_name = self.sequence_folders[self.current_sequence_index]
        self.seq_label.config(text=f"Sequence: {seq_name} ({self.current_sequence_index+1}/{len(self.sequence_folders)})")
        
        path1 = os.path.join(self.base_path_1, seq_name)
        path2 = os.path.join(self.base_path_2, seq_name)
        
        self.video1_frames = self.load_frames_from_folder(path1)
        self.video2_frames = self.load_frames_from_folder(path2)
        
        self.update_max_frames()

    def next_sequence(self):
        if not self.sequence_folders:
            return
        self.stop_play()
        self.current_sequence_index = (self.current_sequence_index + 1) % len(self.sequence_folders)
        self.load_current_sequence()

    def prev_sequence(self):
        if not self.sequence_folders:
            return
        self.stop_play()
        self.current_sequence_index = (self.current_sequence_index - 1) % len(self.sequence_folders)
        self.load_current_sequence()
        
    def load_video1(self):
        folder_path = filedialog.askdirectory(title="Select Video 1 PNG Folder")
        if folder_path:
            self.video1_frames = self.load_frames_from_folder(folder_path)
            self.update_max_frames()
            
    def load_video2(self):
        folder_path = filedialog.askdirectory(title="Select Video 2 PNG Folder")
        if folder_path:
            self.video2_frames = self.load_frames_from_folder(folder_path)
            self.update_max_frames()
    
    def load_frames_from_folder(self, folder_path):
        frames = []
        if not os.path.exists(folder_path):
            return frames
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        png_files.sort()
        
        for filename in png_files:
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath)
                frames.append(img)
            except Exception as e:
                print(f"Failed to load image {filepath}: {e}")
                
        return frames
    
    def update_max_frames(self):
        len1 = len(self.video1_frames)
        len2 = len(self.video2_frames)
        
        if len1 > 0 and len2 > 0:
            self.max_frames = min(len1, len2)
            self.progress_scale.config(to=self.max_frames-1, state=tk.NORMAL)
            self.current_frame = 0
            self.show_current_frame()
        elif len1 > 0:
            self.max_frames = len1
            self.progress_scale.config(to=self.max_frames-1, state=tk.NORMAL)
            self.current_frame = 0
            self.show_current_frame()
        elif len2 > 0:
            self.max_frames = len2
            self.progress_scale.config(to=self.max_frames-1, state=tk.NORMAL)
            self.current_frame = 0
            self.show_current_frame()
        else:
            self.max_frames = 0
            self.progress_scale.config(state=tk.DISABLED)
            self.left_label.config(image="")
            self.right_label.config(image="")
            self.frame_label.config(text="Frame: 0/0")
    
    def show_current_frame(self):
        if self.max_frames <= 0:
            return

        if self.current_frame >= self.max_frames:
            self.current_frame = self.max_frames - 1
            
        if self.current_frame < 0:
            self.current_frame = 0
            
        if self.video1_frames and self.current_frame < len(self.video1_frames):
            frame1 = self.video1_frames[self.current_frame]
            display_img1 = self.resize_image(frame1, 400, 300)
            self.photo1 = ImageTk.PhotoImage(display_img1)
            self.left_label.config(image=self.photo1)
        else:
            self.left_label.config(image="")
        
        if self.video2_frames and self.current_frame < len(self.video2_frames):
            frame2 = self.video2_frames[self.current_frame]
            display_img2 = self.resize_image(frame2, 400, 300)
            self.photo2 = ImageTk.PhotoImage(display_img2)
            self.right_label.config(image=self.photo2)
        else:
            self.right_label.config(image="")
        
        self.progress_var.set(self.current_frame)
        self.frame_label.config(text=f"Frame: {self.current_frame+1}/{self.max_frames}")
    
    def resize_image(self, image, max_width, max_height):
        original_width, original_height = image.size
        ratio = min(max_width/original_width, max_height/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def prev_frame(self):
        if self.max_frames > 0:
            self.current_frame = 0
            self.show_current_frame()
    
    def next_frame(self):
        if self.max_frames > 0:
            self.current_frame = self.max_frames - 1
            self.show_current_frame()
    
    def step_forward(self):
        if self.current_frame < self.max_frames - 1:
            self.current_frame += 1
            self.show_current_frame()
    
    def step_backward(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_current_frame()
    
    def on_progress_change(self, value):
        self.current_frame = int(float(value))
        self.show_current_frame()
    
    def toggle_play(self):
        if self.playing:
            self.stop_play()
        else:
            self.start_play()
    
    def start_play(self):
        if self.max_frames <= 1:
            return
            
        self.playing = True
        self.play_button.config(text="Pause")
        self.stop_event.clear()
        self.play_thread = threading.Thread(target=self.play_loop)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def stop_play(self):
        self.playing = False
        self.stop_event.set()
        self.play_button.config(text="Play")
    
    def play_loop(self):
        while self.playing and self.current_frame < self.max_frames - 1:
            if self.stop_event.is_set():
                break
            self.step_forward()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.1)
        
        if self.current_frame >= self.max_frames - 1:
            self.stop_play()
    
    def save_scores(self):
        score1 = self.score1_var.get()
        score2 = self.score2_var.get()
        
        seq_name = "Unknown"
        if 0 <= self.current_sequence_index < len(self.sequence_folders):
            seq_name = self.sequence_folders[self.current_sequence_index]
            
        result = f"Seq: {seq_name}, Frame ({self.current_frame+1}/{self.max_frames}): V1={score1}, V2={score2}\n"
        self.result_text.insert(tk.END, result)
        self.result_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = DSCQSVideoEvaluator(root)
    root.mainloop()