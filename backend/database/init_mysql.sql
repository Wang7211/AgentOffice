-- MySQL 本地开发环境初始化脚本。
-- 请使用 root 或具备 CREATE USER / CREATE DATABASE / GRANT 权限的账号执行。

CREATE DATABASE IF NOT EXISTS `agentoffice`
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'agentoffice'@'localhost'
    IDENTIFIED BY 'agentoffice123';

CREATE USER IF NOT EXISTS 'agentoffice'@'%'
    IDENTIFIED BY 'agentoffice123';

GRANT ALL PRIVILEGES ON `agentoffice`.* TO 'agentoffice'@'localhost';
GRANT ALL PRIVILEGES ON `agentoffice`.* TO 'agentoffice'@'%';

FLUSH PRIVILEGES;
