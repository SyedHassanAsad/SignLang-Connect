from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def count_fingers(hand_landmarks, image):
    finger_coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
    thumb_coord = (4, 2)
    hand_list = []

    h, w, _ = image.shape  # Move this line inside the count_fingers function

    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        hand_list.append((cx, cy))

    up_count = 0
    for coordinate in finger_coord:
        if hand_list[coordinate[0]][1] < hand_list[coordinate[1]][1]:
            up_count += 1

    if hand_list[thumb_coord[0]][0] > hand_list[thumb_coord[1]][0]:
        up_count += 1

    return up_count

def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, image = cap.read()

        if not success:
            break

        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(RGB_image)
        multi_landmarks = results.multi_hand_landmarks

        if multi_landmarks:
            for hand_landmarks in multi_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                up_count = count_fingers(hand_landmarks, image)
                cv2.putText(image, "Number Sign: {}".format(up_count), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
