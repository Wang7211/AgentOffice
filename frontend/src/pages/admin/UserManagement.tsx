import { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Select,
  Table,
  Tag,
} from 'antd';
import {
  DeleteOutlined,
  EditOutlined,
  ReloadOutlined,
  SearchOutlined,
  TeamOutlined,
  UserOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { deleteUser, listUsers, updateUser } from '../../api/admin';
import type { UserItem } from '../../api/admin';

export default function UserManagement() {
  const [users, setUsers] = useState<UserItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [keyword, setKeyword] = useState('');

  const [editOpen, setEditOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<UserItem | null>(null);
  const [form] = Form.useForm();

  const loadUsers = async (nextPage: number) => {
    setLoading(true);
    try {
      const result = await listUsers(nextPage, 20, keyword || undefined);
      setUsers(result.items);
      setTotal(result.total);
      setPage(result.page);
    } catch {
      message.error('加载用户列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsers(1);
  }, [keyword]);

  const userStats = useMemo(() => {
    const adminCount = users.filter((user) => user.role === 'admin').length;
    const activeCount = users.filter((user) => user.status === 1).length;
    const disabledCount = users.length - activeCount;
    return [
      { label: '用户总数', value: total || users.length, icon: <TeamOutlined />, tone: 'blue' },
      { label: '管理员', value: adminCount, icon: <UserOutlined />, tone: 'violet' },
      { label: '正常账号', value: activeCount, icon: <UserOutlined />, tone: 'green' },
      { label: '禁用账号', value: disabledCount, icon: <UserOutlined />, tone: 'amber' },
    ];
  }, [total, users]);

  const handleEdit = (user: UserItem) => {
    setEditingUser(user);
    form.setFieldsValue({
      nickname: user.nickname,
      role: user.role,
      status: user.status,
      password: '',
    });
    setEditOpen(true);
  };

  const handleSave = async () => {
    if (!editingUser) return;
    try {
      const values = form.getFieldsValue();
      const payload: any = {};
      if (values.nickname) payload.nickname = values.nickname;
      if (values.role) payload.role = values.role;
      if (values.status !== undefined) payload.status = values.status;
      if (values.password) payload.password = values.password;
      await updateUser(editingUser.id, payload);
      message.success('用户信息已更新');
      setEditOpen(false);
      loadUsers(page);
    } catch (err: any) {
      message.error(err.response?.data?.message || '更新失败');
    }
  };

  const handleDelete = async (userId: number) => {
    try {
      await deleteUser(userId);
      message.success('用户已删除');
      loadUsers(page);
    } catch {
      message.error('删除失败');
    }
  };

  const columns = [
    {
      title: '用户',
      dataIndex: 'username',
      key: 'username',
      render: (name: string, record: UserItem) => (
        <span className="user-table-cell">
          <span className="user-avatar small">{(record.nickname || name || 'U')[0].toUpperCase()}</span>
          <span>
            <strong>{name}</strong>
            <small>{record.nickname || '未设置昵称'}</small>
          </span>
        </span>
      ),
    },
    {
      title: '角色',
      dataIndex: 'role',
      key: 'role',
      width: 120,
      render: (role: string) =>
        role === 'admin' ? (
          <Tag className="status-pill danger">管理员</Tag>
        ) : (
          <Tag className="status-pill info">用户</Tag>
        ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: number) =>
        status === 1 ? (
          <Tag className="status-pill success">正常</Tag>
        ) : (
          <Tag className="status-pill danger">禁用</Tag>
        ),
    },
    {
      title: '注册时间',
      dataIndex: 'create_time',
      key: 'create_time',
      width: 180,
      render: (time: string) => (
        <span className="table-muted">{dayjs(time).format('YYYY/MM/DD HH:mm')}</span>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 160,
      render: (_: any, record: UserItem) => (
        <div className="table-actions">
          <Button icon={<EditOutlined />} onClick={() => handleEdit(record)}>
            编辑
          </Button>
          <Popconfirm
            title="删除此用户？"
            description="删除后无法恢复。"
            onConfirm={() => handleDelete(record.id)}
            okText="删除"
            cancelText="取消"
          >
            <Button danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </div>
      ),
    },
  ];

  return (
    <div className="admin-page fade-in">
      <div className="admin-breadcrumb">首页 / 用户管理</div>
      <div className="admin-page-toolbar">
        <div>
          <h1>用户管理</h1>
          <p>管理账号状态、角色权限与登录身份</p>
        </div>
        <div className="toolbar-actions">
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索用户名"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            allowClear
            className="toolbar-search"
          />
          <Button icon={<ReloadOutlined />} onClick={() => loadUsers(page)}>
            刷新
          </Button>
        </div>
      </div>

      <div className="stat-strip">
        {userStats.map((item) => (
          <div className="summary-tile" key={item.label}>
            <span className={`summary-icon ${item.tone}`}>{item.icon}</span>
            <div>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </div>
            <Tag>当前页</Tag>
          </div>
        ))}
      </div>

      <section className="ops-panel table-panel">
        <Table
          dataSource={users}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            onChange: loadUsers,
            showTotal: (count) => `共 ${count} 个用户`,
          }}
        />
      </section>

      <Modal
        title={
          <span className="modal-title">
            <EditOutlined />
            编辑用户 / {editingUser?.username}
          </span>
        }
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        onOk={handleSave}
        okText="保存"
        cancelText="取消"
        width={520}
      >
        <Form form={form} layout="vertical" className="admin-form">
          <Form.Item label="昵称" name="nickname">
            <Input placeholder="输入用户昵称" />
          </Form.Item>
          <Form.Item label="角色" name="role">
            <Select
              options={[
                { label: '普通用户', value: 'user' },
                { label: '管理员', value: 'admin' },
              ]}
            />
          </Form.Item>
          <Form.Item label="状态" name="status">
            <Select
              options={[
                { label: '正常', value: 1 },
                { label: '禁用', value: 0 },
              ]}
            />
          </Form.Item>
          <Form.Item label="新密码（留空不修改）" name="password">
            <Input.Password placeholder="输入新密码" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
