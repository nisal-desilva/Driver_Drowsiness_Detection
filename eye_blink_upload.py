import os
import cv2
import dlib
import RPi.GPIO as GPIO
from scipy.spatial import distance
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont
import asyncio
from telegram import Bot
import numpy as np

# ------------------- TELEGRAM SETUP -------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

bot = Bot(token=BOT_TOKEN) if BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN" else None


async def send_telegram_alert(message):
    if not bot or CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        print("Telegram alert skipped (configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID).")
        return
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print("Telegram send error:", e)


# ------------------- GPIO SETUP -------------------
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

rgb_pins = [
    {"R": 5, "G": 6},     # LED 1
    {"R": 16, "G": 19},   # LED 2
    {"R": 20, "G": 21},   # LED 3
    {"R": 23, "G": 24}    # LED 4
]

for led in rgb_pins:
    GPIO.setup(led["R"], GPIO.OUT)
    GPIO.setup(led["G"], GPIO.OUT)


def set_color_all(r, g):
    """Set color of all 4 RGB LEDs without blue pin"""
    for led in rgb_pins:
        GPIO.output(led["R"], r)
        GPIO.output(led["G"], g)


def set_green():
    set_color_all(0, 1)


def set_red():
    set_color_all(1, 0)


# ------------------- OLED SETUP -------------------
serial = i2c(port=1, address=0x3C)
oled = ssd1306(serial)


def display_text(text):
    image = Image.new("1", oled.size)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (oled.width - w) // 2
    y = (oled.height - h) // 2
    draw.text((x, y), text, font=font, fill=255)
    oled.display(image)


# ------------------- DROWSINESS DETECTION -------------------
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 5
frame_counter = 0
alert_sent = False

PREDICTOR_PATH = os.getenv(
    "SHAPE_PREDICTOR_PATH",
    "shape_predictor_68_face_landmarks.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# ------------------- IMAGE ENHANCEMENT -------------------
def enhance_for_dark(gray):
    """Automatically brighten and enhance image for low-light conditions"""
    avg_brightness = np.mean(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if avg_brightness < 40:
        gamma = 3.0
    elif avg_brightness < 80:
        gamma = 2.0
    else:
        gamma = 1.0

    inv_gamma = 1.0 / gamma
    table = np.array([
        (i / 255.0) ** inv_gamma * 255
        for i in range(256)
    ], dtype="uint8")
    return cv2.LUT(gray, table)


# ------------------- CAMERA SETUP -------------------
cap = cv2.VideoCapture(0)
SHOW_PREVIEW = os.getenv("SHOW_PREVIEW", "1") == "1"

# ------------------- MAIN LOOP -------------------
try:
    while True:
        for _ in range(5):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = enhance_for_dark(gray)
        faces = detector(gray, 0)

        low_ear_in_frame = False
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                low_ear_in_frame = True
                break

        if low_ear_in_frame:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                eyes_closed = True
            else:
                eyes_closed = False
        else:
            frame_counter = 0
            eyes_closed = False
            alert_sent = False

        if eyes_closed:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            set_red()
            display_text("WAKE UP")

            if not alert_sent:
                asyncio.run(send_telegram_alert("Driver appears sleepy! Wake up!"))
                alert_sent = True
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            set_green()
            display_text("HAPPY JOURNEY")

        if SHOW_PREVIEW:
            cv2.imshow("Eye Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    GPIO.cleanup()
    display_text("")