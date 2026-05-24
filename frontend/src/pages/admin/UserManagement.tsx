import { useEffect, useState } from 'react';
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Typography,
  message,
  Space,
  Popconfirm,
} from 'antd';
import { EditOutlined, DeleteOutlined, UserOutlined } from '@ant-design/icons';
import { listUsers, updateUser, deleteUser } from '../../api/admin';
import type { UserItem } from '../../api/admin';
import dayjs from 'dayjs';

const { Title } = Typography;

export default function UserManagement() {
  const [users, setUsers] = useState<UserItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [keyword, setKeyword] = useState('');

  const [editOpen, setEditOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<UserItem | null>(null);
  const [form] = Form.useForm();

  const loadUsers = async (p: number) => {
    setLoading(true);
    try {
      const result = await listUsers(p, 20, keyword || undefined);
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
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username',
      width: 130,
      render: (name: string) => (
        <span>
          <UserOutlined style={{ marginRight: 6, color: 'var(--gray-400)' }} />
          {name}
        </span>
      ),
    },
    { title: '昵称', dataIndex: 'nickname', key: 'nickname', width: 130, render: (v: string | null) => v || '-' },
    {
      title: '角色',
      dataIndex: 'role',
      key: 'role',
      width: 100,
      render: (role: string) =>
        role === 'admin' ? (
          <Tag color="red" style={{ borderRadius: 6 }}>管理员</Tag>
        ) : (
          <Tag color="blue" style={{ borderRadius: 6 }}>用户</Tag>
        ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (s: number) =>
        s === 1
          ? <Tag color="success" style={{ borderRadius: 6 }}>正常</Tag>
          : <Tag color="error" style={{ borderRadius: 6 }}>禁用</Tag>,
    },
    {
      title: '注册时间',
      dataIndex: 'create_time',
      key: 'create_time',
      width: 170,
      render: (t: string) => (
        <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>
          {dayjs(t).format('YYYY-MM-DD HH:mm')}
        </span>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 130,
      render: (_: any, record: UserItem) => (
        <Space>
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)} style={{ padding: 0 }}>
            编辑
          </Button>
          <Popconfirm
            title="确定删除此用户？"
            description="删除后无法恢复"
            onConfirm={() => handleDelete(record.id)}
            okText="删除"
            cancelText="取消"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />} style={{ padding: 0 }}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <Title level={4}>用户管理</Title>
        <Input.Search
          placeholder="搜索用户名..."
          style={{ width: 240 }}
          onSearch={(v) => setKeyword(v)}
          allowClear
        />
      </div>

      <Card className="admin-card" bodyStyle={{ padding: 0 }}>
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
            showTotal: (t) => `共 ${t} 个用户`,
            style: { paddingRight: 16 },
          }}
        />
      </Card>

      <Modal
        title={
          <span>
            <EditOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
            编辑用户 — {editingUser?.username}
          </span>
        }
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        onOk={handleSave}
        okText="保存"
        cancelText="取消"
        width={480}
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
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
