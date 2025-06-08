import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import cv2
import os
import shutil
import csv
import subprocess
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

# مسیرهای اصلی پروژه
DATA_DIR = "data"
MODEL_PATH = "trained_model.yml"
LOG_PATH = "access_log.csv"

# تنظیمات اولیه OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("[×] خطا: فایل Haar Cascade بارگذاری نشد!")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ایجاد پوشه کاربر
def create_user_folder(user_id, user_name):
    folder_path = f"{DATA_DIR}/{user_id}_{user_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[✔] پوشه کاربر ایجاد شد: {folder_path}")
    else:
        print("[!] این پوشه از قبل وجود دارد.")
    return folder_path

# جمع‌آوری چهره‌ها
def capture_faces(app, user_id, user_name, num_samples=20, ip_stream_url="http://192.168.1.3:4747/video"):
    if app.is_processing:
        messagebox.showwarning("هشدار", "در حال انجام عملیات دیگری هستید!")
        return
    app.is_processing = True
    folder_path = create_user_folder(user_id, user_name)

    print(f"[🔍] در حال اتصال به دوربین: {ip_stream_url}")
    cap = cv2.VideoCapture(ip_stream_url if ip_stream_url else 0)
    if not cap.isOpened():
        messagebox.showerror("خطا", "نمی‌توان به دوربین متصل شد! آدرس IP یا اتصال را بررسی کنید.")
        app.is_processing = False
        return

    sample_count = 0
    app.cap = cap
    app.canvas.delete("all")  # پاک کردن کادر

    def update_frame():
        nonlocal sample_count
        if not app.is_processing:
            return
        ret, frame = app.cap.read()
        if not ret:
            messagebox.showerror("خطا", "خواندن تصویر از دوربین با خطا مواجه شد.")
            app.stop_camera()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            sample_count += 1
            face_img = gray[y:y + h, x:x + w]
            img_path = os.path.join(folder_path, f"{sample_count}.jpg")
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {sample_count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        app.display_frame(frame)
        if sample_count >= num_samples:
            messagebox.showinfo("موفقیت", f"{sample_count} تصویر برای {user_name} ذخیره شد!")
            app.stop_camera()
            app.is_processing = False
            return
        app.root.after(10, update_frame)

    update_frame()

# ثبت ورود موفق
def log_access(user_id, user_name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, user_name, now])

# تشخیص چهره
def recognize_face(app):
    if app.is_processing:
        messagebox.showwarning("هشدار", "در حال انجام عملیات دیگری هستید!")
        return
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("خطا", "فایل مدل یافت نشد. ابتدا مدل را آموزش دهید.")
        return

    app.is_processing = True
    recognizer.read(MODEL_PATH)
    cap = cv2.VideoCapture(0)  # برای تشخیص از وب‌کم استفاده می‌کنیم
    if not cap.isOpened():
        messagebox.showerror("خطا", "نمی‌توان به وب‌کم متصل شد!")
        app.is_processing = False
        return

    app.cap = cap
    app.canvas.delete("all")
    recognized = False

    def update_recognition():
        nonlocal recognized
        if not app.is_processing:
            return
        ret, frame = app.cap.read()
        if not ret:
            messagebox.showerror("خطا", "خواندن تصویر از دوربین با خطا مواجه شد.")
            app.stop_camera()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_img)
            if confidence < 70:
                user_folder = os.listdir(DATA_DIR)[label]
                user_id, user_name = user_folder.split("_", 1)
                log_access(user_id, user_name)
                messagebox.showinfo("ورود موفق", f"خوش آمدید، {user_name} (ID: {user_id})")
                recognized = True
                app.stop_camera()
                app.is_processing = False
                return
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        app.display_frame(frame)
        if not recognized:
            app.root.after(10, update_recognition)

    update_recognition()

# آموزش مدل
def train_model():
    try:
        subprocess.run(["python", "train_model.py"], check=True)
        messagebox.showinfo("آموزش مدل", "مدل با موفقیت آموزش داده شد.")
    except subprocess.CalledProcessError:
        messagebox.showerror("خطا", "آموزش مدل با خطا مواجه شد.")

# لیست کاربران
def list_users():
    if not os.path.exists(DATA_DIR):
        messagebox.showinfo("لیست کاربران", "هیچ کاربری ثبت نشده است.")
        return
    users = os.listdir(DATA_DIR)
    if not users:
        messagebox.showinfo("لیست کاربران", "هیچ کاربری وجود ندارد.")
        return
    win = tk.Toplevel()
    win.title("📋 لیست کاربران")
    tree = ttk.Treeview(win, columns=("ID", "Name"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Name", text="نام کاربر")
    tree.pack(fill=tk.BOTH, expand=True)
    for folder in users:
        if "_" in folder:
            user_id, user_name = folder.split("_", 1)
            tree.insert("", "end", values=(user_id, user_name))

# حذف کاربر
def delete_user():
    user = simpledialog.askstring("حذف کاربر", "نام پوشه کاربر (مثلاً 1_Ali) را وارد کنید:")
    if not user:
        return
    path = os.path.join(DATA_DIR, user)
    if os.path.exists(path):
        shutil.rmtree(path)
        messagebox.showinfo("حذف شد", f"کاربر {user} با موفقیت حذف شد.")
    else:
        messagebox.showerror("خطا", "این کاربر وجود ندارد.")

# تغییر نام کاربر
def rename_user():
    old_name = simpledialog.askstring("ویرایش نام", "نام فعلی پوشه کاربر (مثلاً 1_Ali):")
    if not old_name or "_" not in old_name:
        return
    user_id, _ = old_name.split("_", 1)
    new_name = simpledialog.askstring("ویرایش نام", "نام جدید کاربر را وارد کنید:")
    if not new_name:
        return
    old_path = os.path.join(DATA_DIR, old_name)
    new_path = os.path.join(DATA_DIR, f"{user_id}_{new_name}")
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        messagebox.showinfo("ویرایش موفق", "نام کاربر با موفقیت تغییر یافت.")
    else:
        messagebox.showerror("خطا", "پوشه کاربر یافت نشد.")

# مشاهده گزارش ورود
def view_logs():
    if not os.path.exists(LOG_PATH):
        messagebox.showinfo("گزارش ورود", "هیچ گزارشی ثبت نشده است.")
        return
    win = tk.Toplevel()
    win.title("📊 گزارش ورود کاربران")
    tree = ttk.Treeview(win, columns=("ID", "Name", "Time"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Name", text="نام")
    tree.heading("Time", text="زمان")
    tree.pack(fill=tk.BOTH, expand=True)
    with open(LOG_PATH, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:
                tree.insert("", "end", values=(row[0], row[1], row[2]))

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("سیستم احراز هویت با تشخیص چهره")
        self.root.geometry("600x600")
        self.root.config(bg="#F5F5F5")
        self.cap = None
        self.is_processing = False

        # کادر برای نمایش تصویر دوربین
        self.canvas = tk.Canvas(root, width=320, height=240, bg="black")
        self.canvas.pack(pady=10)

        # استایل دکمه‌ها
        style = {"font": ("Vazirmatn", 12), "bg": "#1976D2", "fg": "white", "width": 30, "height": 2}

        # دکمه‌ها
        tk.Label(root, text="سیستم تشخیص چهره", font=("Vazirmatn", 16, "bold"), bg="#F5F5F5").pack(pady=10)
        tk.Button(root, text="ثبت‌نام کاربر جدید", command=self.register_new_user, **style).pack(pady=5)
        tk.Button(root, text="ورود با تشخیص چهره", command=lambda: recognize_face(self), **style).pack(pady=5)
        tk.Button(root, text="آموزش دوباره مدل", command=train_model, **style).pack(pady=5)
        tk.Button(root, text="نمایش لیست کاربران", command=list_users, **style).pack(pady=5)
        tk.Button(root, text="ویرایش نام کاربر", command=rename_user, **style).pack(pady=5)
        tk.Button(root, text="حذف کاربر", command=delete_user, **style).pack(pady=5)
        tk.Button(root, text="نمایش گزارش ورود", command=view_logs, **style).pack(pady=5)
        tk.Button(root, text="خروج", command=self.on_closing, **style).pack(pady=10)

    def register_new_user(self):
        user_id = simpledialog.askstring("ثبت‌نام کاربر", "آیدی عددی کاربر را وارد کنید:")
        if not user_id:
            return
        user_name = simpledialog.askstring("ثبت‌نام کاربر", "نام کاربر را وارد کنید:")
        if not user_name:
            return
        ip_url = simpledialog.askstring("دوربین IP", "آدرس دوربین IP را وارد کنید (خالی برای وب‌کم):")
        capture_faces(self, user_id, user_name, ip_stream_url=ip_url if ip_url else None)
        train_model()

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 240))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def stop_camera(self):
        self.is_processing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.canvas.delete("all")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()