# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from infer import Predictor, args

from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def index():
    if request.method == 'POST':
        f = request.files['file'] # 获取上传的图片
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        # 储存图片
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 对上传的图片进行预测
        predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path,
                              threshold=args.threshold)
        start = time.time()
        img = cv2.imread(upload_path)
        boxes, names = predictor.recognition(img)
        predictor.draw_face(img, boxes, names)

        return render_template('upload_ok.html', userinput=names, val1=time.time())

    return render_template('index.html')


if __name__ == '__main__':
    # app.debug = True
    app.run()
