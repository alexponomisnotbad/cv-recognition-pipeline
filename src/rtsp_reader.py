import cv2

# Замените на вашу RTSP-ссылку
# rtsp_url = "rtsp://192.168.137.196:8554/live/camera1"
rtsp_url = "rtsp://localhost:8554/live/camera1"
# Инициализация VideoCapture
cap = cv2.VideoCapture(rtsp_url)


# Проверка успешного открытия потока
if not cap.isOpened():
    print("Ошибка: не удалось открыть RTSP-поток")
else:
    while True:
        ret, frame = cap.read() # Считываем кадр

        if not ret: 
            print("Ошибка: не удалось получить кадр (поток завершен или недоступен)")
            break

        # Отображение кадра
        cv2.imshow('RTSP Stream', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()