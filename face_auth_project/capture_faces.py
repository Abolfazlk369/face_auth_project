import cv2
import os
import sys

def create_user_folder(user_id, user_name):
    folder_path = f"data/{user_id}_{user_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[✔] پوشه کاربر ایجاد شد: {folder_path}")
    else:
        print("[!] این پوشه از قبل وجود دارد.")
    return folder_path

def capture_faces(user_id, user_name, num_samples=20, ip_stream_url="http://192.168.1.3:8080/video"):
    folder_path = create_user_folder(user_id, user_name)

    print(f"[🔍] در حال اتصال به دوربین IP: {ip_stream_url}")

    # اتصال به دوربین IP
    cap = cv2.VideoCapture(ip_stream_url)
    if not cap.isOpened():
        print("[×] خطا: نمی‌توان به دوربین IP متصل شد! لطفاً آدرس IP، پورت یا اتصال شبکه را بررسی کنید.")
        return

    # بارگذاری مدل تشخیص چهره
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("[×] خطا: فایل Haar Cascade بارگذاری نشد!")
        return

    sample_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[×] خواندن تصویر از دوربین با خطا مواجه شد.")
            break

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

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[!] خروج با کلید Q.")
            break
        elif sample_count >= num_samples:
            print("[✔] گرفتن تصاویر تکمیل شد.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        user_id = sys.argv[1]
        user_name = sys.argv[2]
        ip_url = sys.argv[3]
        capture_faces(user_id, user_name, ip_stream_url=ip_url)
    else:
        print("📸 ثبت نام کاربر جدید")
        user_id = input("🆔 آیدی عددی کاربر را وارد کنید: ")
        user_name = input("👤 نام کاربر را وارد کنید: ")
        capture_faces(user_id, user_name)