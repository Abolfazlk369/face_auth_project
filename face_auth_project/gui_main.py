import cv2
import os
import shutil
import csv
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

# مسیرهای اصلی پروژه
DATA_DIR = "data"
MODEL_PATH = "trained_model.yml"
LOG_PATH = "access_log.csv"

# ثبت ورود موفق
def log_access(user_id, user_name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, user_name, now])

# تشخیص چهره و ورود
def recognize_face():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("خطا", "فایل مدل یافت نشد. ابتدا مدل را آموزش دهید.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                break
            else:
                messagebox.showwarning("خطا", "چهره شناسایی نشد.")

        if recognized:
            break

        cv2.imshow("تشخیص چهره", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ثبت‌نام کاربر جدید با اجرای capture_faces.py
def register_new_user():
    user_id = simpledialog.askstring("ثبت‌نام کاربر", "آیدی عددی کاربر را وارد کنید:")
    if not user_id:
        return
    user_name = simpledialog.askstring("ثبت‌نام کاربر", "نام کاربر را وارد کنید:")
    if not user_name:
        return

    try:
        subprocess.run(["python", "capture_faces.py", user_id, user_name], check=True)
        messagebox.showinfo("موفقیت", "تصاویر کاربر با موفقیت ثبت شدند.")
        train_model()
    except subprocess.CalledProcessError:
        messagebox.showerror("خطا", "در اجرای ثبت‌نام مشکلی پیش آمد.")

# آموزش مدل با اجرای train_model.py
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

# رابط گرافیکی اصلی
def main_gui():
    root = tk.Tk()
    root.title("سیستم احراز هویت با تشخیص چهره")
    root.geometry("500x550")
    root.config(bg="#F5F5F5")

    style = {"font": ("Vazirmatn", 12), "bg": "#1976D2", "fg": "white", "width": 30, "height": 2}

    tk.Label(root, text="سیستم تشخیص چهره", font=("Vazirmatn", 16, "bold"), bg="#F5F5F5").pack(pady=20)

    tk.Button(root, text="ورود با تشخیص چهره", command=recognize_face, **style).pack(pady=5)
    tk.Button(root, text="ثبت‌نام کاربر جدید", command=register_new_user, **style).pack(pady=5)
    tk.Button(root, text="آموزش دوباره مدل", command=train_model, **style).pack(pady=5)
    tk.Button(root, text="نمایش لیست کاربران", command=list_users, **style).pack(pady=5)
    tk.Button(root, text="ویرایش نام کاربر", command=rename_user, **style).pack(pady=5)
    tk.Button(root, text="حذف کاربر", command=delete_user, **style).pack(pady=5)
    tk.Button(root, text="نمایش گزارش ورود", command=view_logs, **style).pack(pady=5)
    tk.Button(root, text="خروج", command=root.destroy, **style).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
