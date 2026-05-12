import json
import time
from email.utils import formataddr
import torch
import pymysql
import pandas as pd
import string
import torch as th
import nltk
import re
import torch.nn.functional as F
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from logging.handlers import TimedRotatingFileHandler
from flask import request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from torch_geometric.data import DataLoader
from Process.process import loadBiData, loadTree
from Twitter.GCN_Twitter import Net
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from io import BytesIO
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, render_template,  session, send_file
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import random
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"


# 设置日志文件夹和日志文件名
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)  # 确保目录存在
log_filename = os.path.join(log_folder, 'app_log.log')

# 设置按天分割的日志处理器
file_handler = TimedRotatingFileHandler(
    filename=log_filename,
    when='midnight',           # 每天午夜创建新日志文件
    interval=1,
    backupCount=60,             # 保留最近 60 天的日志
    encoding='utf-8'
)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 获取日志记录器并配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 避免重复添加处理器（防止多次运行时重复输出）
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())  # 控制台输出


# 获取日志记录器
logger = logging.getLogger()


# Flask应用
app = Flask(__name__)
app.secret_key = 'secret_key'  # 用于Session加密

# 连接数据库（假设你已经在PyCharm配置好，直接写连接）
db = pymysql.connect(host='localhost', user='root', password='root', database='fake', charset='utf8mb4')
cursor = db.cursor()

def get_db():
    global db
    try:
        db.ping(reconnect=True)
    except:
        db = pymysql.connect(host='localhost', user='root', password='root', database='fake', charset='utf8mb4')
    cursor = db.cursor()
    return db, cursor



# --- 路由 ---

# ---------- 发送验证码邮件 ----------
def send_verification_email(to_email, code):
    smtp_server = 'smtp.qq.com'
    smtp_port = 587
    from_email = '3237695086@qq.com'              # 发件邮箱
    from_password = 'vziedixzkckmcjje'              # SMTP 授权码

    subject = '找回密码验证码'
    content = f'您的验证码是：{code}，请在5分钟内使用。'

    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = formataddr(("系统管理员", from_email))  # 正确格式
    message['To'] = formataddr(("", to_email))
    message['Subject'] = Header(subject, 'utf-8')

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, [to_email], message.as_string())
        server.quit()
        print(f"邮件已发送至 {to_email}")
        return True
    except Exception as e:
        print(f"发送失败: {e}")
        return False


# ---------- 忘记密码：输入邮箱 ----------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        db, cursor = get_db()

        # 同时匹配用户名和邮箱
        cursor.execute("SELECT * FROM users WHERE username = %s AND email = %s", (username, email))
        user = cursor.fetchone()

        if user:
            verification_code = str(random.randint(100000, 999999))
            session['reset_email'] = email
            session['reset_username'] = username
            session['verification_code'] = verification_code
            session['verification_code_time'] = time.time()  # 记录发送验证码时间戳

            if send_verification_email(email, verification_code):
                return redirect(url_for('reset_password'))
            else:
                return render_template('forgot_password.html', error="邮件发送失败，请稍后重试")
        else:
            return render_template('forgot_password.html', error="用户名与邮箱不匹配")
    return render_template('forgot_password.html')


# ---------- 重置密码 ----------
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        code = request.form['code']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            return render_template('reset_password.html', error="两次密码不一致")

        # 验证码过期检测，超过5分钟(300秒)视为过期
        send_time = session.get('verification_code_time')
        if not send_time or time.time() - send_time > 300:
            return render_template('reset_password.html', error="验证码已过期，请重新获取")

        if code != session.get('verification_code'):
            return render_template('reset_password.html', error="验证码错误")

        email = session.get('reset_email')
        hashed_password = generate_password_hash(new_password)

        db, cursor = get_db()
        try:
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, email))
            db.commit()
            # 清理 session
            session.pop('reset_email', None)
            session.pop('reset_username', None)
            session.pop('verification_code', None)
            session.pop('verification_code_time', None)
            return redirect(url_for('login'))
        except:
            db.rollback()
            return render_template('reset_password.html', error="密码重置失败")
    return render_template('reset_password.html')

from flask import jsonify, request

@app.route('/resend_verification_code', methods=['POST'])
def resend_verification_code():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')

    if not username or not email:
        return jsonify({'success': False, 'message': '缺少用户名或邮箱参数'})

    db, cursor = get_db()
    cursor.execute("SELECT * FROM users WHERE username = %s AND email = %s", (username, email))
    user = cursor.fetchone()

    if not user:
        return jsonify({'success': False, 'message': '用户名与邮箱不匹配'})

    verification_code = str(random.randint(100000, 999999))
    session['reset_email'] = email
    session['reset_username'] = username
    session['verification_code'] = verification_code
    session['verification_code_time'] = time.time()

    if send_verification_email(email, verification_code):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': '邮件发送失败，请稍后重试'})



# 注册页
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        nickname = request.form['nickname']  # 获取nickname字段
        email = request.form['email']  # 获取email字段
        phone = request.form['phone']  # 获取phone字段
        hashed_password = generate_password_hash(password)
        db, cursor = get_db()
        sql = "INSERT INTO users (username, password, email, nickname, phone) VALUES (%s, %s,%s, %s, %s)"
        try:
            cursor.execute(sql, (username, hashed_password, email, nickname, phone))
            db.commit()
            return redirect(url_for('login'))
        except:
            db.rollback()
            return render_template('register.html', error="注册失败，用户名可能已存在")
    return render_template('register.html')


# 登录成功后跳转到 index 页面
@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')



@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db, cursor = get_db()
        # 查找用户名对应的 id（user_id）、哈希密码、权限和头像
        sql = "SELECT id, password, permission, avatar FROM users WHERE username=%s"
        cursor.execute(sql, (username,))
        result = cursor.fetchone()

        if result:
            user_id = result[0]
            hashed_password = result[1]
            if check_password_hash(hashed_password, password):
                session['username'] = username
                session['user_id'] = user_id  # 保存 user_id 到 session
                logger.info(f"用户 {username} 登录成功")

                # 保存权限和头像
                session['permission'] = result[2]
                session['avatar'] = result[3]

                return redirect(url_for('index'))

        logger.warning(f"用户 {username} 登录失败，用户名或密码错误")
        return render_template('login.html', error="用户名或密码错误")

    return render_template('login.html')



@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db, cursor = get_db()
    # 使用 user_id 查询用户信息
    sql = "SELECT username, email, nickname, phone, avatar FROM users WHERE id=%s"
    cursor.execute(sql, (session['user_id'],))
    user = cursor.fetchone()

    if user:
        user_info = {
            'username': user[0],
            'email': user[1],
            'nickname': user[2],
            'phone': user[3],
            'avatar': user[4],
        }
        return render_template('profile.html', user=user_info)
    else:
        return render_template('profile.html', error="未找到用户信息")



@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db, cursor = get_db()

    if request.method == 'POST':
        # 处理表单提交逻辑
        nickname = request.form['nickname']
        email = request.form['email']
        phone = request.form['phone']
        avatar = request.files.get('avatar')

        # 处理头像上传
        avatar_filename = None
        if avatar and avatar.filename:
            filename = secure_filename(avatar.filename)
            upload_path = os.path.join('static/uploads', filename)
            avatar.save(upload_path)
            avatar_filename = filename

            # 更新数据库（含头像）
            sql = """
            UPDATE users
            SET nickname=%s, email=%s, phone=%s, avatar=%s
            WHERE id=%s
            """
            cursor.execute(sql, (nickname, email, phone, avatar_filename, session['user_id']))
            logger.info(f"用户ID {session['user_id']} 更新了个人资料（含头像）")
        else:
            # 更新数据库（不含头像）
            sql = """
            UPDATE users
            SET nickname=%s, email=%s, phone=%s
            WHERE id=%s
            """
            cursor.execute(sql, (nickname, email, phone, session['user_id']))
            logger.info(f"用户ID {session['user_id']} 更新了个人资料（不含头像）")

        db.commit()  # 提交事务
        flash("资料更新成功")
        return redirect(url_for('profile'))

    # 处理 GET 请求，渲染编辑资料页面
    sql = "SELECT nickname, email, phone, avatar FROM users WHERE id=%s"
    cursor.execute(sql, (session['user_id'],))
    user = cursor.fetchone()
    if user:
        return render_template('edit_profile.html', user=user)
    else:
        return render_template('edit_profile.html', error="未找到用户信息")


# 设备和模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "model/bert_lense"
model_bert = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)



# 文本预处理
def text_preprocessing(text):
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation + "–—−±×÷"), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_keywords(text, top_k=5):
    # 基本清理
    text = re.sub(r'\W+', ' ', text.lower())

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    words = vectorizer.get_feature_names_out()

    # 排序
    sorted_indices = scores.argsort()[::-1]
    raw_keywords = [words[i] for i in sorted_indices]

    # 词形还原并去重
    seen = set()
    keywords = []
    for word in raw_keywords:
        lemma = lemmatizer.lemmatize(word)
        if lemma not in seen and lemma not in stop_words and len(lemma) > 2:
            seen.add(lemma)
            keywords.append(word)
        if len(keywords) >= top_k:
            break

    return keywords

def detect_text(text):
    cleaned_text = text_preprocessing(text)
    inputs = tokenizer([cleaned_text], return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model_bert.eval()
    with torch.no_grad():
        outputs = model_bert(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        prob = torch.softmax(logits, dim=1)[0][1].item()

    label_name = "Fake" if pred == 1 else "Real"

    # 提取关键词和解释
    keywords = extract_keywords(text)
    explanations = generate_explanations(keywords)

    return label_name, prob, keywords, explanations

def generate_explanations(keywords):
    # 这里可以实现基于关键词的解释生成逻辑
    # 例如：假设返回固定的解释
    explanations = []
    for keyword in keywords:
        explanations.append(f"关键词“{keyword}”在文本中频繁出现并具有辨别力。")
    return explanations


@app.route('/detectbert', methods=['GET', 'POST'])
def detectbert():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        text = request.form['text']

        label, prob, keywords, explanations = detect_text(text)
        if label != "Fake":
            prob = 1 - prob

        # 将关键词转换为逗号分隔的字符串
        keywords_str = ', '.join(keywords)

        try:
            db, cursor = get_db()
            sql = """
                INSERT INTO records (user_id, username, text, result, confidence, time, keywords, model_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                session['user_id'],
                session['username'],
                text,
                label,
                prob,
                datetime.now(),
                keywords_str,
                'BERT'
            ))
            db.commit()
            logger.info(f"检测记录成功插入数据库: 用户ID={session['user_id']} 用户名={session['username']}")

        except Exception as e:
            logger.error(f"数据库插入失败: {e}")
            result = {'error': '保存检测记录失败，请联系管理员'}

        result = {
            'label': label,
            'confidence': f"{prob * 100:.4f}%",
            'confidence_decimal': prob,
            'keywords_explained': list(zip(keywords, explanations)),
            'keywords': keywords_str
        }

    return render_template('detectbert.html', result=result)


# 图神经网络模型加载
model_gcn = Net(5000, 64, 64).to(device)
model_gcn.load_state_dict(th.load("GCNTwitter.pt", map_location=device))

def evaluate_single_input(model, input_data, treeDic, datasetname):
    _, testdata_list = loadBiData(datasetname, treeDic, input_data, input_data, TDdroprate=0.2, BUdroprate=0.2)
    test_loader = DataLoader(testdata_list, batch_size=1, shuffle=False)

    model.eval()
    with th.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            prob = F.softmax(out, dim=1)
            confidence, pred = prob.max(1)

            # 提取中间信息（这里只展示前3个节点和前10条边）
            node_features_sample = batch_data.x[:3].cpu().tolist()
            edge_index_sample = batch_data.edge_index[:, :10].cpu().tolist()

            return pred.item(), confidence.item(), node_features_sample, edge_index_sample


def test_model_with_input(model, datasetname, input_data):
    treeDic = loadTree(datasetname)
    prediction, confidence, node_features, edge_index = evaluate_single_input(model, input_data, treeDic, datasetname)
    return prediction, confidence, node_features, edge_index

def build_graph_json(edge_index):
    node_ids = set(edge_index[0] + edge_index[1])  # 获取所有出现过的节点 ID
    nodes = [{"data": {"id": str(i), "label": f"Node {i}"}} for i in node_ids]
    edges = [{"data": {"source": str(s), "target": str(t)}} for s, t in zip(edge_index[0], edge_index[1])]
    return {"nodes": nodes, "edges": edges}

@app.route('/detectgcn', methods=['GET', 'POST'])
def detectgcn():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        file = request.files.get('npz_file', None)

        if file and file.filename.endswith('.npz'):
            try:
                # 保存上传的文件到本地临时目录
                upload_path = os.path.join('uploads', file.filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(upload_path)

                # 提取文件名作为 ID
                filename = os.path.splitext(file.filename)[0]
                input_data = [filename]

                # 模型推理
                prediction, confidence, node_features, edge_index = test_model_with_input(
                    model_gcn, "Twitter15", input_data
                )

                input_description = f"From uploaded {file.filename}"

            except Exception as e:
                return render_template('detectgcn.html', result={'error': f'文件处理出错: {str(e)}'})

        else:
            input_data = request.form['input_data'].split(',')
            input_data = [id.strip() for id in input_data if id.strip()]
            if not input_data:
                return render_template('detectgcn.html', result={'error': '请上传文件或输入ID！'})

            prediction, confidence = test_model_with_input(model_gcn, "Twitter15", input_data)
            input_description = ','.join(input_data)
            edge_index = []  # 避免引用未定义变量

        # 保存检测记录到数据库
        try:
            db, cursor = get_db()
            sql = """
                INSERT INTO records (user_id, username, text, result, confidence, time, keywords, model_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                session['user_id'],
                session['username'],
                input_description,
                "Fake" if prediction in [1, 2] else "Real",
                confidence,
                datetime.now(),
                json.dumps(edge_index),
                'GCN'
            ))
            db.commit()
            logger.info(f"GCN检测记录插入成功: 用户ID={session['user_id']} 用户名={session['username']}")

        except Exception as e:
            logger.error(f"数据库插入失败: {e}")
            result = {'error': '检测结果保存失败，请联系管理员'}

        # 构建图数据用于前端展示
        graph_data = build_graph_json(edge_index)

        result = {
            'label': "Fake" if prediction in [1, 3] else "Real",
            'confidence': f"{confidence * 100:.4f}%",
            'edge_index_sample': edge_index,
            'graph_json': json.dumps(graph_data)
        }

    return render_template('detectgcn.html', result=result)




@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    model = request.args.get('model', '')  # 可选的模型类型筛选
    db, cursor = get_db()

    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    # 构建 SQL 查询条件
    where_clauses = []
    params = []

    # 非管理员用户只能看自己的记录
    if session['permission'] != 'admin':
        where_clauses.append("user_id = %s")
        params.append(session['user_id'])

    # 如果指定了模型类型
    if model:
        where_clauses.append("model_type = %s")
        params.append(model)

    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # 查询语句
    if session['permission'] == 'admin':
        sql = f"""
            SELECT text, result, confidence, time, keywords, username
            FROM records
            {where_sql}
            ORDER BY time DESC
            LIMIT %s OFFSET %s
        """
    else:
        sql = f"""
            SELECT text, result, confidence, time, keywords
            FROM records
            {where_sql}
            ORDER BY time DESC
            LIMIT %s OFFSET %s
        """

    cursor.execute(sql, (*params, per_page, offset))
    records = cursor.fetchall()
    logger.info(f"用户 {session['username']} 查看历史记录")

    # 为非管理员用户补全 username 字段以统一结构
    if session['permission'] != 'admin':
        records = [record + (session['username'],) for record in records]

    # 查询总记录数以进行分页
    count_sql = f"SELECT COUNT(*) FROM records {where_sql}"
    cursor.execute(count_sql, params)
    total_records = cursor.fetchone()[0]
    total_pages = (total_records + per_page - 1) // per_page  # 向上取整

    return render_template('history.html', records=records, page=page, total_pages=total_pages)





@app.route('/manage_users', methods=['GET', 'POST'])
def manage_users():
    if 'username' not in session or session.get('permission') != 'admin':
        return redirect(url_for('login'))  # 仅管理员可访问

    db, cursor = get_db()

    # 分页参数
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    # 处理 POST 请求（添加、删除、更新）
    if request.method == 'POST':
        action = request.form.get('action')

        try:
            if action == 'add':
                username = request.form['username']
                password = request.form['password']
                permission = request.form['permission']

                # 检查用户名是否已存在
                cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    return render_template('manage_users.html', users=[], page=page, total_pages=0, error="用户名已存在")

                hashed_password = generate_password_hash(password)
                sql = "INSERT INTO users (username, password, permission) VALUES (%s, %s, %s)"
                cursor.execute(sql, (username, hashed_password, permission))
                db.commit()
                logger.info(f"管理员 {session['username']} 添加用户：{username}")

            elif action == 'delete':
                user_id = request.form['user_id']
                # 防止管理员误删自己
                if int(user_id) == session.get('user_id'):
                    return render_template('manage_users.html', users=[], page=page, total_pages=0, error="不能删除当前登录管理员")
                sql = "DELETE FROM users WHERE id = %s"
                cursor.execute(sql, (user_id,))
                db.commit()
                logger.info(f"管理员 {session['username']} 删除用户 ID: {user_id}")

            elif action == 'update':
                user_id = request.form['user_id']
                new_permission = request.form['new_permission']
                sql = "UPDATE users SET permission = %s WHERE id = %s"
                cursor.execute(sql, (new_permission, user_id))
                db.commit()
                logger.info(f"管理员 {session['username']} 更新用户权限 ID: {user_id} -> {new_permission}")

            return redirect(url_for('manage_users'))

        except Exception as e:
            logger.error(f"管理员操作用户失败: {str(e)}")
            return render_template('manage_users.html', users=[], page=page, total_pages=0, error=f"操作失败: {str(e)}")

    # 查询用户数据（用于 GET 渲染）
    cursor.execute("SELECT id, username, permission FROM users ORDER BY username LIMIT %s OFFSET %s", (per_page, offset))
    users = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    total_pages = (total_users + per_page - 1) // per_page  # 向上取整

    logger.info(f"管理员 {session['username']} 查看用户管理页面")
    return render_template('manage_users.html', users=users, page=page, total_pages=total_pages)



@app.route('/export')
def export():
    if 'username' not in session:
        return redirect(url_for('login'))

    db, cursor = get_db()

    # 获取当前用户的 user_id（用于日志记录）
    cursor.execute("SELECT id FROM users WHERE username = %s", (session['username'],))
    user_row = cursor.fetchone()
    if not user_row:
        return "用户不存在", 400

    user_id = user_row[0]

    # 查询记录（确保选择了 keywords 字段）
    sql = "SELECT id, text, result, confidence, time, keywords FROM records WHERE username = %s"
    cursor.execute(sql, (session['username'],))
    records = cursor.fetchall()

    # 处理 keywords 字段，如果是 JSON 或列表类型，转换为字符串
    processed_records = []
    for record in records:
        record_id, text, result, confidence, time, keywords = record
        if isinstance(keywords, (list, dict)):  # 如果是列表或字典类型
            keywords = ', '.join(str(keyword) for keyword in keywords)  # 转换为逗号分隔的字符串
        processed_records.append((record_id, text, result, confidence, time, keywords))

    # 生成 DataFrame，包括 keywords（作为字符串）
    df = pd.DataFrame(processed_records, columns=['ID', 'Text', 'Result', 'Confidence', 'Time', 'Keywords'])

    logger.info(f"用户导出记录: {session['username']} (ID: {user_id})")

    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)

    return send_file(output, as_attachment=True, download_name='records.csv', mimetype='text/csv')



# 退出
@app.route('/logout')
def logout():
    username = session.get('username')
    logger.info(f"用户 {username} 登出")  # 记录用户登出
    session.clear()
    return redirect(url_for('login'))




if __name__ == '__main__':
    db, cursor = get_db()
    # cursor.execute(...)
    app.run(host='0.0.0.0', port=5000, ssl_context=('localhost+2.pem', 'localhost+2-key.pem'), debug=True)