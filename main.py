import cv2
import time
import os
import smtplib
import threading
import logging
import warnings
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ['KMP_WARNINGS'] = '0'
logging.getLogger("ultralytics").setLevel(logging.ERROR)
#Отключение предупреждений Ultralyticss

load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
CONFIDENCE = float(os.getenv("CONFIDENCE", 0.5))
#Подгрузка данных почты через файл окружения (.env)

ARCHIVE_DIR = "archive_records"
MAX_EMAIL_SIZE = 15 * 1024 * 1024


def send_smart_notification(subject, body, file_path=None):
    if not SENDER_EMAIL or not SENDER_PASSWORD: return
    msg = MIMEMultipart()
    msg['From'], msg['To'], msg['Subject'] = SENDER_EMAIL, RECEIVER_EMAIL, subject

    file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
    file_sent = False

    # Проверка лимита вложений (15 МБ)
    if file_path and file_size < MAX_EMAIL_SIZE:
        body += f"\n\nВидео во вложении ({round(file_size / 1024 / 1024, 2)} МБ)."
        try:
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
                msg.attach(part)
            file_sent = True
        except:
            pass
    else:
        body += f"\n\n[ВНИМАНИЕ] Файл сохранен только локально (превышен лимит): {file_path}"

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        if file_sent and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"\n[log] Ошибка почты: {e}")


class HybridSecurityMonitor:
    def __init__(self):
        # Загрузка предобученной модели YOLOv8 (версия Nano)
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(0)
        self.recording = False
        self.out = None
        self.last_seen = 0
        self.start_time_ts = None
        self.unique_ids = set()
        self.wait_start = None
        if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR)

    def run(self):
        print(f"[log] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Система активна.")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break

            # Запуск трекинга c отслеживанием класса 0 (люди)
            results = self.model.track(frame, classes=[0], conf=CONFIDENCE, persist=True, verbose=False)
            found = results[0].boxes.id is not None
            now = time.time()
            now_dt = datetime.now()
            ts_log = now_dt.strftime('%Y-%m-%d %H:%M:%S')

            if found:
                self.last_seen = now
                if self.wait_start is not None:
                    print(f"\r{' ' * 50}\r[log] [{ts_log}] Человек снова в кадре! Запись продолжается.")
                    self.wait_start = None

                if not self.recording:
                    print(f"[log] [{ts_log}] Объект обнаружен! Начинаю запись.")
                    self.recording = True
                    self.start_time_ts = now_dt
                    self.unique_ids.clear()
                    self.current_file = os.path.join(ARCHIVE_DIR, f"rec_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
                    # Настройка видеопотока: кодек mp4v, 15 FPS
                    self.out = cv2.VideoWriter(self.current_file, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (640, 480))

                # Подсчет уникальных ID людей для статистики
                for obj_id in results[0].boxes.id.int().tolist(): self.unique_ids.add(obj_id)

            if self.recording:
                self.out.write(cv2.resize(frame, (640, 480)))

                if not found:
                    if self.wait_start is None:
                        print(f"[log] [{ts_log}] Объект потерян. Ожидание завершения...")
                        self.wait_start = now

                    elapsed = now - self.wait_start
                    remaining = 10 - int(elapsed)

                    if remaining >= 0:
                        print(f"\r[log] Завершение сессии через: {remaining} сек.".ljust(50), end="", flush=True)

                    # Логика завершения сессии после 10 секунд отсутствия
                    if elapsed > 10:
                        end_dt = datetime.now()
                        print(
                            f"\r{' ' * 50}\r[log] [{end_dt.strftime('%Y-%m-%d %H:%M:%S')}] Время истекло. Формирую отчет.")
                        self.recording = False
                        self.out.release()

                        duration = round((end_dt - self.start_time_ts).total_seconds(), 1)
                        subject = f"Security Alert: {ts_log}"
                        body = (f"ОТЧЕТ СИСТЕМЫ\n"
                                f"--------------------------\n"
                                f"Начало: {self.start_time_ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"Конец:  {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"Длительность: {duration} сек.\n"
                                f"Всего людей: {len(self.unique_ids)}")

                        # Асинхронная отправка файла по почте
                        threading.Thread(target=send_smart_notification,
                                         args=(subject, body, self.current_file)).start()
                        self.wait_start = None

            # Отрисовка рамок вокруг объектов
            cv2.imshow("Security Feed", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HybridSecurityMonitor().run()
