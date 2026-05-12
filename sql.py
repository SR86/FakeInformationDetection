import pymysql

# 连接信息
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='root',
    database='fake',
    charset='utf8mb4'
)

output_file = 'fake_schema_export.sql'

with connection.cursor() as cursor, open(output_file, 'w', encoding='utf-8') as f:
    # 获取所有表名
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"SHOW CREATE TABLE `{table}`;")
        result = cursor.fetchone()
        table_name = result[0]
        create_stmt = result[1]

        # 写入文件
        f.write(f"-- ----------------------------\n")
        f.write(f"-- Table structure for `{table_name}`\n")
        f.write(f"-- ----------------------------\n")
        f.write(f"{create_stmt};\n\n")

print(f"✅ 所有建表语句已导出至: {output_file}")
