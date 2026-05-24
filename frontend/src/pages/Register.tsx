import { useState } from 'react';
import { Card, Form, Input, Button, message, Typography } from 'antd';
import { UserOutlined, LockOutlined, SmileOutlined, RobotOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';
import { register } from '../api/auth';
import { useAuthStore } from '../stores/authStore';

const { Text } = Typography;

export default function Register() {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);

  const onFinish = async (values: {
    username: string;
    password: string;
    nickname: string;
  }) => {
    setLoading(true);
    try {
      const result = await register(values);
      setAuth(result.access_token, {
        user_id: result.user_id,
        username: result.username,
        nickname: result.nickname,
        role: result.role,
        avatar: result.avatar,
      });
      message.success({ content: '注册成功，欢迎加入！', duration: 2 });
      navigate('/chat');
    } catch (err: any) {
      message.error(err.response?.data?.message || err.message || '注册失败');
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
          <h1>创建账号</h1>
          <p>注册 AgentOffice 企业智能助手</p>
        </div>
        <Form
          name="register"
          onFinish={onFinish}
          layout="vertical"
          size="large"
          autoComplete="off"
          requiredMark={false}
        >
          <Form.Item
            name="username"
            rules={[
              { required: true, message: '请输入用户名' },
              { min: 3, message: '用户名至少 3 个字符' },
            ]}
          >
            <Input
              prefix={<UserOutlined style={{ color: 'var(--gray-400)' }} />}
              placeholder="用户名"
              style={{ height: 44 }}
            />
          </Form.Item>
          <Form.Item name="nickname">
            <Input
              prefix={<SmileOutlined style={{ color: 'var(--gray-400)' }} />}
              placeholder="昵称（选填）"
              style={{ height: 44 }}
            />
          </Form.Item>
          <Form.Item
            name="password"
            rules={[
              { required: true, message: '请输入密码' },
              { min: 6, message: '密码至少 6 个字符' },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: 'var(--gray-400)' }} />}
              placeholder="密码"
              style={{ height: 44 }}
            />
          </Form.Item>
          <Form.Item
            name="confirmPassword"
            dependencies={['password']}
            rules={[
              { required: true, message: '请确认密码' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('password') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('两次输入的密码不一致'));
                },
              }),
            ]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: 'var(--gray-400)' }} />}
              placeholder="确认密码"
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
              注 册
            </Button>
          </Form.Item>
          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 13 }}>
              已有账号？{' '}
              <Link to="/login" style={{ fontWeight: 600 }}>
                返回登录
              </Link>
            </Text>
          </div>
        </Form>
      </Card>
    </div>
  );
}
