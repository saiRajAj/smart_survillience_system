from flask import (
    Flask,
    Response,
    jsonify,
    render_template
)

from threading import Thread

import config

from safevision import SafeVision


app = Flask(__name__)

vision = None

startup_error = None


try:

    vision = SafeVision()

    Thread(
        target=vision.run,
        daemon=True
    ).start()

except Exception as e:

    startup_error = str(e)

    print(
        "SafeVision startup error:",
        startup_error
    )


@app.route("/")
def index():

    return render_template(
        "index.html",
        startup_error=startup_error
    )


@app.route("/video_feed")
def video_feed():

    if vision is None:

        return Response(
            "SafeVision failed to start",
            status=503
        )

    return Response(

        vision.mjpeg(),

        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route("/api/status")
def status():

    if vision is None:

        return jsonify({
            "error":
                startup_error
        }), 503

    with vision.lock:

        return jsonify(
            vision.status
        )


@app.route("/api/incidents")
def incidents():

    if vision is None:

        return jsonify({
            "error":
                startup_error
        }), 503

    return jsonify(
        vision.db.recent(10)
    )


@app.route("/api/health")
def health():

    return jsonify({

        "status":
            "OK"
            if vision
            else
            "ERROR",

        "camera":
            vision.status.get(
                "camera"
            )
            if vision
            else
            "OFFLINE",

        "error":
            startup_error
    })


if __name__ == "__main__":

    app.run(

        host=config.HOST,

        port=config.PORT,

        debug=False,

        threaded=True
    )
