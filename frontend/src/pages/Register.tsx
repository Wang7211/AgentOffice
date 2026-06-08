import { useState } from 'react';
import { Card, Form, Input, Button, message, Typography } from 'antd';
import {
  UserOutlined,
  LockOutlined,
  SmileOutlined,
  SafetyCertificateOutlined,
  DatabaseOutlined,
  ApiOutlined,
} from '@ant-design/icons';
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
      message.success({ content: '注册成功', duration: 2 });
      navigate('/chat');
    } catch (err: any) {
      message.error(err.response?.data?.message || err.message || '注册失败');
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
          <h1>创建账号</h1>
          <p>账号创建后可进入独立会话空间，管理员可在后台管理用户与系统配置。</p>
        </div>
        <div className="auth-capabilities">
          <span><DatabaseOutlined /> 知识库隔离</span>
          <span><ApiOutlined /> 调用链路</span>
          <span><SafetyCertificateOutlined /> 权限控制</span>
        </div>
      </section>

      <Card className="auth-card">
        <div className="auth-header">
          <h2>注册</h2>
          <p>填写账号信息后进入 AgentOffice。</p>
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
            label="用户名"
            rules={[
              { required: true, message: '请输入用户名' },
              { min: 3, message: '用户名至少 3 个字符' },
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="请输入用户名"
            />
          </Form.Item>
          <Form.Item name="nickname" label="昵称">
            <Input
              prefix={<SmileOutlined />}
              placeholder="可选"
            />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: true, message: '请输入密码' },
              { min: 6, message: '密码至少 6 个字符' },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="请输入密码"
            />
          </Form.Item>
          <Form.Item
            name="confirmPassword"
            label="确认密码"
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
              prefix={<LockOutlined />}
              placeholder="再次输入密码"
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
              注册
            </Button>
          </Form.Item>
          <div className="auth-switch">
            <Text type="secondary">
              已有账号？ <Link to="/login">返回登录</Link>
            </Text>
          </div>
        </Form>
      </Card>
    </div>
  );
}
