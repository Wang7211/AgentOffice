import { useState } from 'react';
import { Layout, Menu, Typography, Button, Tag } from 'antd';
import {
  DashboardOutlined,
  DatabaseOutlined,
  ApiOutlined,
  UserOutlined,
  SettingOutlined,
  MessageOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/admin', icon: <DashboardOutlined />, label: '总览' },
  { key: '/admin/knowledge', icon: <DatabaseOutlined />, label: '知识库' },
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
    <Layout className="admin-shell">
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="light"
        width={248}
        className="admin-sider"
        trigger={null}
      >
        <div className="admin-brand">
          <span className="brand-icon">
            <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
          </span>
          {!collapsed && (
            <span className="brand-copy">
              <strong>AgentOffice</strong>
              <span>Admin Console</span>
            </span>
          )}
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          className="admin-menu"
        />
      </Sider>
      <Layout>
        <Header className="admin-header">
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            className="icon-button"
          />
          <div className="admin-header-title">
            <strong>运营控制台</strong>
            <span>知识库、调用链路与用户权限</span>
          </div>
          <div className="admin-header-actions">
            <Button
              icon={<MessageOutlined />}
              onClick={() => navigate('/chat')}
            >
              返回对话
            </Button>
            <Tag className="role-tag">Admin</Tag>
            <div className="admin-user">
              <span className="user-avatar small">
                {(user?.nickname || user?.username || 'U')[0].toUpperCase()}
              </span>
              <Text strong>{user?.nickname || user?.username}</Text>
            </div>
            <Button
              type="text"
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              danger
              className="icon-button"
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
