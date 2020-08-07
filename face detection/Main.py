from flask import Flask, render_template, Response
from Video import Camera


app = Flask(__name__)


@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/get_video')
def get_video():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)