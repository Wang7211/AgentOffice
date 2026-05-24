CREATE DATABASE IF NOT EXISTS agentoffice
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE agentoffice;

CREATE TABLE IF NOT EXISTS sys_user (
  id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
  username VARCHAR(32) NOT NULL COMMENT '登录账号',
  password VARCHAR(128) NOT NULL COMMENT '加密存储密码',
  nickname VARCHAR(32) NULL COMMENT '用户昵称',
  role VARCHAR(16) NOT NULL DEFAULT 'user' COMMENT '角色：admin/user',
  avatar VARCHAR(255) NULL COMMENT '头像地址',
  status TINYINT NOT NULL DEFAULT 1 COMMENT '账号状态：0禁用、1正常',
  is_delete TINYINT NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删、1已删',
  create_time DATETIME(3) NOT NULL COMMENT '创建时间',
  update_time DATETIME(3) NOT NULL COMMENT '更新时间',
  UNIQUE KEY uk_sys_user_name (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息表';

CREATE TABLE IF NOT EXISTS chat_session (
  session_id VARCHAR(64) PRIMARY KEY COMMENT '会话唯一ID',
  user_id INT NOT NULL COMMENT '关联用户ID',
  session_name VARCHAR(128) NULL COMMENT '会话名称',
  model_name VARCHAR(32) NOT NULL COMMENT '当前使用大模型名称',
  status TINYINT NOT NULL DEFAULT 1 COMMENT '会话状态',
  is_delete TINYINT NOT NULL DEFAULT 0 COMMENT '逻辑删除',
  create_time DATETIME(3) NOT NULL COMMENT '创建时间',
  update_time DATETIME(3) NOT NULL COMMENT '更新时间',
  KEY idx_chat_session_user (user_id, is_delete)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话信息表';

CREATE TABLE IF NOT EXISTS chat_record (
  id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
  session_id VARCHAR(64) NOT NULL COMMENT '关联会话ID',
  role VARCHAR(16) NOT NULL COMMENT '消息角色',
  content LONGTEXT NOT NULL COMMENT '对话文本内容',
  token_cost INT NOT NULL DEFAULT 0 COMMENT '本次消息消耗Token数量',
  is_delete TINYINT NOT NULL DEFAULT 0 COMMENT '逻辑删除',
  create_time DATETIME(3) NOT NULL COMMENT '创建时间',
  KEY idx_chat_record_session (session_id, is_delete)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话记录表';

CREATE TABLE IF NOT EXISTS tool_record (
  id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
  chat_record_id INT NOT NULL COMMENT '关联对话记录ID',
  tool_name VARCHAR(64) NOT NULL COMMENT '工具名称',
  tool_input TEXT NOT NULL COMMENT '工具入参JSON字符串',
  tool_result LONGTEXT NULL COMMENT '工具返回结果',
  cost_time FLOAT NOT NULL DEFAULT 0 COMMENT '工具执行耗时',
  status TINYINT NOT NULL DEFAULT 1 COMMENT '执行状态',
  error_msg TEXT NULL COMMENT '异常报错信息',
  create_time DATETIME(3) NOT NULL COMMENT '调用时间',
  KEY idx_tool_record_chat (chat_record_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用日志表';

CREATE TABLE IF NOT EXISTS knowledge_file (
  id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
  file_name VARCHAR(255) NOT NULL COMMENT '文件原始名称',
  file_suffix VARCHAR(16) NOT NULL COMMENT '文件后缀',
  file_size INT NOT NULL COMMENT '文件大小KB',
  save_path VARCHAR(255) NOT NULL COMMENT '服务器存储路径',
  upload_user_id INT NOT NULL COMMENT '上传人用户ID',
  is_delete TINYINT NOT NULL DEFAULT 0 COMMENT '逻辑删除',
  create_time DATETIME(3) NOT NULL COMMENT '上传时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库文件表';

CREATE TABLE IF NOT EXISTS knowledge_chunk (
  id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
  file_id INT NOT NULL COMMENT '关联知识库文件ID',
  chunk_text TEXT NOT NULL COMMENT '分片文本内容',
  vector_id VARCHAR(64) NOT NULL COMMENT 'Milvus向量唯一ID',
  chunk_index INT NOT NULL COMMENT '文本分片序号',
  is_delete TINYINT NOT NULL DEFAULT 0 COMMENT '逻辑删除',
  create_time DATETIME(3) NOT NULL COMMENT '分片创建时间',
  KEY idx_knowledge_chunk_file (file_id, is_delete),
  KEY idx_knowledge_chunk_order (file_id, chunk_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='向量分片表';

CREATE TABLE IF NOT EXISTS system_config (
  id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
  config_key VARCHAR(64) NOT NULL COMMENT '配置项唯一键名',
  config_value VARCHAR(512) NULL COMMENT '配置项对应值',
  config_name VARCHAR(128) NOT NULL COMMENT '配置项中文名称',
  config_type VARCHAR(32) NOT NULL COMMENT '配置类型',
  remark VARCHAR(255) NULL COMMENT '配置备注',
  is_delete TINYINT NOT NULL DEFAULT 0 COMMENT '逻辑删除',
  create_time DATETIME(3) NOT NULL COMMENT '记录创建时间',
  update_time DATETIME(3) NOT NULL COMMENT '记录更新时间',
  UNIQUE KEY uk_config_key (config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='系统配置表';
