import { useState } from 'react';
import { Card, Form, Input, Button, message, Typography } from 'antd';
import { UserOutlined, LockOutlined, RobotOutlined } from '@ant-design/icons';
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
    <div className="auth-container">
      <Card className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
          </div>
          <h1>AgentOffice</h1>
          <p>企业智能助手 · 登录</p>
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
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input
              prefix={<UserOutlined style={{ color: 'var(--gray-400)' }} />}
              placeholder="用户名"
              style={{ height: 44 }}
            />
          </Form.Item>
          <Form.Item
            name="password"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: 'var(--gray-400)' }} />}
              placeholder="密码"
              style={{ height: 44 }}
            />
          </Form.Item>
          <Form.Item style={{ marginTop: 28, marginBottom: 20 }}>
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
              size="large"
              style={{ height: 46, fontSize: 15, fontWeight: 600 }}
            >
              登 录
            </Button>
          </Form.Item>
          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 13 }}>
              还没有账号？{' '}
              <Link to="/register" style={{ fontWeight: 600 }}>
                立即注册
              </Link>
            </Text>
          </div>
        </Form>
      </Card>
    </div>
  );
}
