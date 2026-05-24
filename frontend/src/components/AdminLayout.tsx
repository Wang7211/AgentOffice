import { useState } from 'react';
import { Layout, Menu, Typography, Button } from 'antd';
import {
  DashboardOutlined,
  DatabaseOutlined,
  ApiOutlined,
  UserOutlined,
  SettingOutlined,
  MessageOutlined,
  LogoutOutlined,
  RobotOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/admin', icon: <DashboardOutlined />, label: 'Dashboard' },
  { key: '/admin/knowledge', icon: <DatabaseOutlined />, label: '知识库管理' },
  { key: '/admin/traces', icon: <ApiOutlined />, label: '链路追踪' },
  { key: '/admin/users', icon: <UserOutlined />, label: '用户管理' },
  { key: '/admin/settings', icon: <SettingOutlined />, label: '系统设置' },
];

export default function AdminLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="light"
        width={240}
        style={{
          borderRight: '1px solid var(--gray-200)',
          boxShadow: collapsed ? 'none' : '1px 0 10px rgba(0,0,0,0.04)',
        }}
      >
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 10,
            borderBottom: '1px solid var(--gray-100)',
            padding: '0 16px',
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              minWidth: 32,
              background: 'linear-gradient(135deg, var(--primary), #8b5cf6)',
              borderRadius: 10,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: 16,
            }}
          >
            <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" style={{ width: 24, height: 24, borderRadius: 6 }} />
          </div>
          {!collapsed && (
            <span
              style={{
                fontSize: 17,
                fontWeight: 700,
                background: 'linear-gradient(135deg, var(--primary), #8b5cf6)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                whiteSpace: 'nowrap',
              }}
            >
              AgentOffice
            </span>
          )}
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 'none', marginTop: 4 }}
        />
      </Sider>
      <Layout>
        <Header
          className="admin-header"
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{
              marginRight: 'auto',
              width: 40,
              height: 40,
              color: 'var(--gray-500)',
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Button
              type="primary"
              ghost
              icon={<MessageOutlined />}
              onClick={() => navigate('/chat')}
              size="small"
              style={{ borderRadius: 8, fontWeight: 500 }}
            >
              返回问答
            </Button>
            <div style={{ width: 1, height: 24, background: 'var(--gray-200)', margin: '0 4px' }} />
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 10,
                background: 'linear-gradient(135deg, var(--primary), #8b5cf6)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontSize: 13,
                fontWeight: 600,
              }}
            >
              {(user?.nickname || user?.username || 'U')[0].toUpperCase()}
            </div>
            <Text strong style={{ fontSize: 13, color: 'var(--gray-700)' }}>
              {user?.nickname || user?.username}
            </Text>
            <Button
              type="text"
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              danger
              style={{ color: 'var(--gray-400)' }}
            />
          </div>
        </Header>
        <Content className="admin-content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
