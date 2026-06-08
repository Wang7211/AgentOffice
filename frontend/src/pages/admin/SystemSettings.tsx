import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  Button,
  Form,
  Input,
  message,
  Modal,
  Spin,
  Table,
  Tag,
} from 'antd';
import {
  ApiOutlined,
  DatabaseOutlined,
  EditOutlined,
  ReloadOutlined,
  SearchOutlined,
  SettingOutlined,
  SlidersOutlined,
} from '@ant-design/icons';
import { listConfig, updateConfig } from '../../api/admin';
import type { ConfigItem } from '../../api/admin';

const typeMeta: Record<string, { label: string; icon: ReactNode; tone: string }> = {
  model: { label: '模型配置', icon: <ApiOutlined />, tone: 'violet' },
  file: { label: '文件配置', icon: <DatabaseOutlined />, tone: 'blue' },
  vector: { label: '向量配置', icon: <DatabaseOutlined />, tone: 'green' },
  general: { label: '全局配置', icon: <SlidersOutlined />, tone: 'amber' },
};

export default function SystemSettings() {
  const [configs, setConfigs] = useState<ConfigItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [keyword, setKeyword] = useState('');
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

  const filteredConfigs = useMemo(() => {
    const q = keyword.trim().toLowerCase();
    if (!q) return configs;
    return configs.filter((item) =>
      [item.config_key, item.config_name, item.config_value, item.remark]
        .join(' ')
        .toLowerCase()
        .includes(q),
    );
  }, [configs, keyword]);

  const groups = useMemo(() => {
    return Object.entries(typeMeta).map(([type, meta]) => ({
      type,
      ...meta,
      count: configs.filter((item) => item.config_type === type).length,
    }));
  }, [configs]);

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
    {
      title: '配置键',
      dataIndex: 'config_key',
      key: 'config_key',
      width: 220,
      render: (key: string) => <code className="config-key">{key}</code>,
    },
    {
      title: '名称',
      dataIndex: 'config_name',
      key: 'config_name',
      width: 180,
      render: (name: string) => <strong>{name}</strong>,
    },
    {
      title: '值',
      dataIndex: 'config_value',
      key: 'config_value',
      ellipsis: true,
      render: (value: string) => <span className="config-value">{value}</span>,
    },
    {
      title: '类型',
      dataIndex: 'config_type',
      key: 'config_type',
      width: 120,
      render: (type: string) => <Tag className="soft-tag">{typeMeta[type]?.label || type}</Tag>,
    },
    {
      title: '备注',
      dataIndex: 'remark',
      key: 'remark',
      width: 180,
      render: (remark: string | null) => remark || <span className="table-muted">-</span>,
    },
    {
      title: '操作',
      key: 'actions',
      width: 110,
      render: (_: any, record: ConfigItem) => (
        <Button icon={<EditOutlined />} onClick={() => handleEdit(record)}>
          编辑
        </Button>
      ),
    },
  ];

  if (loading) {
    return (
      <div className="admin-loading">
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div className="admin-page fade-in">
      <div className="admin-breadcrumb">首页 / 系统设置</div>
      <div className="admin-page-toolbar">
        <div>
          <h1>系统设置</h1>
          <p>只展示 application 级配置，避免运行中配置被误改</p>
        </div>
        <div className="toolbar-actions">
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索配置键或名称"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            allowClear
            className="toolbar-search"
          />
          <Button icon={<ReloadOutlined />} onClick={loadConfigs}>
            刷新
          </Button>
        </div>
      </div>

      <div className="stat-strip">
        {groups.map((group) => (
          <div className="summary-tile" key={group.type}>
            <span className={`summary-icon ${group.tone}`}>{group.icon}</span>
            <div>
              <small>{group.label}</small>
              <strong>{group.count}</strong>
            </div>
            <Tag>application</Tag>
          </div>
        ))}
      </div>

      <section className="ops-panel table-panel">
        <div className="panel-title">
          <span><SettingOutlined /> 配置列表</span>
        </div>
        <Table
          dataSource={filteredConfigs}
          columns={columns}
          rowKey="id"
          pagination={false}
        />
      </section>

      <Modal
        title={
          <span className="modal-title">
            <EditOutlined />
            编辑配置 / {editingConfig?.config_key}
          </span>
        }
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        onOk={handleSave}
        okText="保存"
        cancelText="取消"
        width={560}
      >
        {editingConfig && (
          <Form form={form} layout="vertical" className="admin-form">
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
              <Input.TextArea rows={4} placeholder="请输入配置值" />
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
}
