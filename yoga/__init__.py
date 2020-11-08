
from flask import Flask, render_template, Response, jsonify
from flask_wtf import FlaskForm
from wtforms import SubmitField
import cv2
from imutils.video import WebcamVideoStream
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
accuracy = None
btn = None
img_path = './yoga/static/images/'
model_kps = None


def dist(kps1, kps2):
    d = 0
    for i in range(0, 17):
        x = kps1[i][0]-kps2[i][0]
        y = kps1[i][1]-kps2[i][1]
        d = d+(x*x)+(y*y)
    return d


def parse_output(heatmap_data, offset_data, threshold):
    '''
    Input:
      heatmap_data - hetmaps for an image. Three dimension array
      offset_data - offset vectors for an image. Three dimension array
      threshold - probability threshold for the keypoints. Scalar value
    Output:
      array with coordinates of the keypoints and flags for those that have
      low probability
    '''

    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(heatmap_data.shape[-1]):

        joint_heatmap = heatmap_data[..., i]
        max_val_pos = np.squeeze(np.argwhere(
            joint_heatmap == np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos/8*257, dtype=np.int32)
        pose_kps[i, 0] = int(
            remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i])
        pose_kps[i, 1] = int(
            remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i+joint_num])
        max_prob = np.max(joint_heatmap)

        if max_prob > threshold:
            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                pose_kps[i, 2] = 1

    return pose_kps


def draw_kps(show_img, kps, ratio=None):
    for i in range(5, kps.shape[0]):
        if kps[i, 2]:
            if isinstance(ratio, tuple):
                cv2.circle(show_img, (int(round(kps[i, 1]*ratio[1])), int(
                    round(kps[i, 0]*ratio[0]))), 2, (0, 255, 255), round(int(1*ratio[1])))
                continue
            cv2.circle(show_img, (kps[i, 1], kps[i, 0]), 2, (0, 255, 255), -1)
    return show_img


def join_point(img, kps):

    body_parts = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11),
                  (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

    for part in body_parts:
        cv2.line(img, (kps[part[0]][1], kps[part[0]][0]), (kps[part[1]][1], kps[part[1]][0]),
                 color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=2)


def detect(img):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    img_new = cv2.resize(img, (width, height))

    template_input = np.expand_dims(img_new.copy(), axis=0)

    floating_model = input_details[0]['dtype'] == np.float32

    if floating_model:
        template_input = (np.float32(template_input) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], template_input)
    interpreter.invoke()
    template_output_data = interpreter.get_tensor(output_details[0]['index'])
    template_offset_data = interpreter.get_tensor(output_details[1]['index'])
    template_heatmaps = np.squeeze(template_output_data)
    template_offsets = np.squeeze(template_offset_data)

    template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
    template_show = np.array(template_show*255, np.uint8)
    template_kps = parse_output(template_heatmaps, template_offsets, 0.3)

    for i in range(0, 17):
        if(template_kps[i][0] > 256):
            template_kps[i][0] = 256

        if(template_kps[i][1] > 256):
            template_kps[i][1] = 256

    img_kps = draw_kps(template_show.copy(), template_kps)

    template_pose = np.zeros_like(template_show)
    join_point(template_pose, template_kps[:, :2])
    img_line = cv2.addWeighted(img_kps, 1, template_pose, 1, 0)
    img_line_kps = draw_kps(template_pose.copy(), template_kps)

    kps_line_og = cv2.resize(img_line_kps, (img.shape[1], img.shape[0]))

    img_line_kps_og = cv2.addWeighted(img, 1, kps_line_og, 1, 0)

    return (template_kps, img_line_kps, img_line_kps_og)


class VideoCamera(object):

    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        global accuracy
        global btn
        image = self.stream.read()
        kps_img, kps_line, image = detect(image)
        d = dist(model_kps, kps_img)
        if(d < 3000):
            accuracy = "95+"
            btn = "success"
        elif (d < 7000):
            accuracy = "85-95"
            btn = "success"
        elif (d < 15000):
            accuracy = "75-85"
            btn = "warning"
        elif (d < 40000):
            accuracy = "55-75"
            btn = "warning"
        else:
            accuracy = "<50"
            btn = "danger"
        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        # data.append(name)
        return data


@app.route("/change_label", methods=['POST'])
def change_label():
    # Return the text you want the label to be
    li = []
    li.append(accuracy)
    li.append(btn)
    return jsonify(li)


class typeForm(FlaskForm):
    dfdp = SubmitField('Downward Facing Dog Pose')
    mount_p = SubmitField('Mountain-Pose')
    tree_p = SubmitField('Lotus-Pose')
    plank = SubmitField('Plank')
    bridge_p = SubmitField('Warrior-Pose')
    triangle_p = SubmitField('Triangle-Pose')


def gen(camera):
    while True:
        data = camera.get_frame()

        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen():
#     cap = cv2.VideoCapture(0)
#     while(cap.isOpened()):

#         ret, image = cap.read()
#         if(ret):
#             frame = cv2.imencode('.jpg', img)[1].tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             time.sleep(0.5)
#         else:
#             break


@app.route('/video_feed')
def video_feed():

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    # print(os.path.basename)
    form = typeForm()
    global model_kps
    if(form.validate_on_submit()):
        title = "Plank"
        yoga = None
        global accuracy
        type = "success"
        if(form.dfdp.data):
            title = "Downward Facing Dog Pose"
            yoga = "downward_dog"
        if(form.mount_p.data):
            title = "Mountain-Pose"
            yoga = "mountain"
        if(form.tree_p.data):
            title = "Lotus-Pose"
            yoga = "Lotus"
        if(form.plank.data):
            title = "Plank"
            yoga = "plank"
        if(form.bridge_p.data):
            title = "Warrior-Pose"
            yoga = "warrior"
        if(form.triangle_p.data):
            title = "Triangle-Pose"
            yoga = "triangle"

        img = cv2.imread(img_path+yoga+'.jpg')
        model_kps, _, _ = detect(img)
        return render_template('yoga.html', title=title, yoga=yoga, type=type)

    # print(img.shape[1])
    # kps, og = detect(img)
    # cv2.imwrite("tmp_line.jpg", img)
    # cv2.imwrite("tmp_pose.jpg", og)
    return render_template('home.html', form=form)  # , kps=kps, og=og)
