import { useState } from 'react';
import { Card, Form, Input, Button, message, Typography } from 'antd';
import {
  UserOutlined,
  LockOutlined,
  SafetyCertificateOutlined,
  DatabaseOutlined,
  ApiOutlined,
} from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';
import { login, getMe } from '../api/auth';
import { useAuthStore } from '../stores/authStore';

const { Text } = Typography;

export default function Login() {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);

  const onFinish = async (values: { username: string; password: string }) => {
    setLoading(true);
    try {
      const result = await login(values);
      setAuth(result.access_token, {
        user_id: result.user_id,
        username: result.username,
        nickname: result.nickname,
        role: result.role,
        avatar: result.avatar,
      });
      getMe().then((user) => setAuth(result.access_token, user)).catch(() => {});
      message.success({ content: '登录成功', duration: 2 });
      navigate('/chat');
    } catch (err: any) {
      message.error(err.response?.data?.message || err.message || '登录失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell">
      <section className="auth-brief">
        <div className="auth-brand">
          <span className="brand-icon large">
            <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
          </span>
          <span>
            <strong>AgentOffice</strong>
            <small>企业 AI 办公工作台</small>
          </span>
        </div>
        <div className="auth-brief-copy">
          <h1>统一入口</h1>
          <p>面向知识库问答、工具调用、链路追踪和用户权限的内部工作台。</p>
        </div>
        <div className="auth-capabilities">
          <span><DatabaseOutlined /> 知识库隔离</span>
          <span><ApiOutlined /> 调用链路</span>
          <span><SafetyCertificateOutlined /> 权限控制</span>
        </div>
      </section>

      <Card className="auth-card">
        <div className="auth-header">
          <h2>登录</h2>
          <p>使用你的工作账号进入 AgentOffice。</p>
        </div>
        <Form
          name="login"
          onFinish={onFinish}
          layout="vertical"
          size="large"
          autoComplete="off"
          requiredMark={false}
        >
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="请输入用户名"
            />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="请输入密码"
            />
          </Form.Item>
          <Form.Item className="auth-submit">
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
              size="large"
            >
              登录
            </Button>
          </Form.Item>
          <div className="auth-switch">
            <Text type="secondary">
              还没有账号？ <Link to="/register">创建账号</Link>
            </Text>
          </div>
        </Form>
      </Card>
    </div>
  );
}
