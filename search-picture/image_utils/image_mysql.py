import datetime
import pandas as pd
import pymysql

# 连接数据库
class Image_mysql:
    def __init__(self, db_info=None, table_name="image_information"):
        self.db_info = {}
        if not db_info:
            db_info = {}
        self.db_info['host'] = db_info['host'] if 'host' in db_info else "localhost"
        self.db_info['user'] = db_info['user'] if 'user' in db_info else "root"
        self.db_info['password'] = db_info['password'] if 'password' in db_info else "topview"
        self.db_info['port'] = db_info['port'] if 'port' in db_info else 3306
        self.db_info['database'] = db_info['database'] if 'database' in db_info else "document-ml"
        self.db_info['charset'] = db_info['charset'] if 'charset' in db_info else "utf8"
        self.table_name = table_name
        self.init_db()

    def conn_mysql(self):
        conn = pymysql.connect(**self.db_info)
        return conn

    def init_db(self):
        conn = self.conn_mysql()
        sql = ''' 
            CREATE TABLE IF NOT EXISTS `%s` (
                  `id` INT(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
                  `mid` INT(32) DEFAULT NULL COMMENT 'milvus的id',
                  `uid` INT(32) DEFAULT NULL COMMENT '图片d',
                  `json` TEXT COMMENT '图片向量',
                  `date` DATETIME DEFAULT NULL COMMENT '建立时间',
                  `state` TINYINT(1) DEFAULT NULL COMMENT '图片状态',
                  PRIMARY KEY (`id`)
                ) ENGINE=INNODB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
            '''%(self.table_name)
        cursor = conn.cursor()
        try:
            # 执行sql语句
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            # 如果发生错误，则回滚事务
            print(e.args)
            conn.rollback()
        conn.close()

    def insert_img_info(self, mid, uid, json):
        conn = self.conn_mysql()
        cursor = conn.cursor()
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = "INSERT `image_information`(`mid`, `uid`, `json`, `date`, `state`) " \
              "VALUES('%d','%d','%s','%s','%d');" % (
                  mid, uid, json, date, 1)
        try:
            # 执行sql语句
            cursor.execute(sql)
            lastid = int(cursor.lastrowid)
            conn.commit()
        except Exception as e:
            # 如果发生错误，则回滚事务
            print(e.args)
            lastid = -1
            conn.rollback()
        conn.close()
        return lastid

    def load_img_info(self):
        db = self.conn_mysql()
        cursor = db.cursor()
        # 获得数据库的图片数据
        try:
            sql = "select * from image_information where state=1"
            img_info = pd.read_sql(sql, db)
        except:
            msg = "An error occurred while inserting data into the database"
            db.close()
            return None, None, None, -1, msg
        db.close()
        # 获得数据库的特定字段内容
        try:
            mids = img_info['mid']
            uids = img_info['uid']
            feature_strs = img_info['json']
            img_num = img_info.shape[0]
        except NameError:
            msg = "field error in database"
            return None, None, None, -1, msg
        return mids, uids, feature_strs, img_num, None


