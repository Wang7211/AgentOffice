import { useEffect, useState } from 'react';
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Typography,
  Tag,
  message,
  Spin,
} from 'antd';
import { EditOutlined, SettingOutlined } from '@ant-design/icons';
import { listConfig, updateConfig } from '../../api/admin';
import type { ConfigItem } from '../../api/admin';

const { Title } = Typography;

const typeColorMap: Record<string, string> = {
  model: 'purple',
  file: 'blue',
  vector: 'green',
  general: 'orange',
};

export default function SystemSettings() {
  const [configs, setConfigs] = useState<ConfigItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [editOpen, setEditOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<ConfigItem | null>(null);
  const [form] = Form.useForm();

  const loadConfigs = async () => {
    setLoading(true);
    try {
      const data = await listConfig();
      setConfigs(data);
    } catch {
      message.error('加载配置失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfigs();
  }, []);

  const handleEdit = (config: ConfigItem) => {
    setEditingConfig(config);
    form.setFieldsValue({ config_value: config.config_value });
    setEditOpen(true);
  };

  const handleSave = async () => {
    if (!editingConfig) return;
    try {
      const { config_value } = form.getFieldsValue();
      await updateConfig(editingConfig.id, config_value);
      message.success('配置已更新');
      setEditOpen(false);
      loadConfigs();
    } catch (err: any) {
      message.error(err.response?.data?.message || '更新失败');
    }
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    {
      title: '配置键',
      dataIndex: 'config_key',
      key: 'config_key',
      width: 160,
      render: (key: string) => (
        <code style={{ fontSize: 12, color: 'var(--gray-600)' }}>{key}</code>
      ),
    },
    { title: '名称', dataIndex: 'config_name', key: 'config_name', width: 150 },
    { title: '值', dataIndex: 'config_value', key: 'config_value', ellipsis: true },
    {
      title: '类型',
      dataIndex: 'config_type',
      key: 'config_type',
      width: 90,
      render: (t: string) => (
        <Tag color={typeColorMap[t] || 'default'} style={{ borderRadius: 6 }}>
          {t}
        </Tag>
      ),
    },
    {
      title: '备注',
      dataIndex: 'remark',
      key: 'remark',
      width: 150,
      render: (v: string | null) => v || <span style={{ color: 'var(--gray-400)' }}>-</span>,
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (_: any, record: ConfigItem) => (
        <Button
          type="link"
          size="small"
          icon={<EditOutlined />}
          onClick={() => handleEdit(record)}
          style={{ padding: 0 }}
        >
          编辑
        </Button>
      ),
    },
  ];

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 120 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <Title level={4}>系统设置</Title>
        <Tag icon={<SettingOutlined />} color="processing" style={{ borderRadius: 8 }}>
          系统配置
        </Tag>
      </div>

      <Card className="admin-card" bodyStyle={{ padding: 0 }}>
        <Table
          dataSource={configs}
          columns={columns}
          rowKey="id"
          pagination={false}
        />
      </Card>

      <Modal
        title={
          <span>
            <EditOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
            编辑配置 — {editingConfig?.config_key}
          </span>
        }
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        onOk={handleSave}
        okText="保存"
        cancelText="取消"
        width={520}
      >
        {editingConfig && (
          <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
            <Form.Item label="配置键">
              <Input value={editingConfig.config_key} disabled />
            </Form.Item>
            <Form.Item label="配置名称">
              <Input value={editingConfig.config_name} disabled />
            </Form.Item>
            <Form.Item
              label="配置值"
              name="config_value"
              rules={[{ required: true, message: '请输入配置值' }]}
            >
              <Input.TextArea rows={3} placeholder="请输入配置值" />
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
}
