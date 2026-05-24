import { useEffect, useState } from 'react';
import {
  Card,
  Table,
  Tag,
  Typography,
  message,
  Select,
  Space,
  Modal,
  Descriptions,
  Spin,
  Button,
} from 'antd';
import { EyeOutlined, ReloadOutlined, ApiOutlined } from '@ant-design/icons';
import { listTraces, getTraceDetail } from '../../api/admin';
import type { TraceItem, TraceDetail } from '../../api/admin';
import dayjs from 'dayjs';

const { Title, Text, Paragraph } = Typography;

export default function TraceView() {
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [toolFilter, setToolFilter] = useState<string | undefined>(undefined);
  const [statusFilter, setStatusFilter] = useState<number | undefined>(undefined);

  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detail, setDetail] = useState<TraceDetail | null>(null);

  const loadTraces = async (p: number) => {
    setLoading(true);
    try {
      const result = await listTraces(p, 20, toolFilter, statusFilter);
      setTraces(result.items);
      setTotal(result.total);
      setPage(result.page);
    } catch {
      message.error('加载链路数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTraces(1);
  }, [toolFilter, statusFilter]);

  const handleViewDetail = async (id: number) => {
    setDetailLoading(true);
    setDetailOpen(true);
    try {
      const data = await getTraceDetail(id);
      setDetail(data);
    } catch {
      message.error('加载详情失败');
    } finally {
      setDetailLoading(false);
    }
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: '工具',
      dataIndex: 'tool_name',
      key: 'tool_name',
      width: 120,
      render: (name: string) => (
        <Tag color="blue" style={{ borderRadius: 6 }}>
          <ApiOutlined style={{ marginRight: 4 }} />
          {name}
        </Tag>
      ),
    },
    {
      title: '会话',
      dataIndex: 'session_name',
      key: 'session_name',
      ellipsis: true,
      width: 150,
      render: (v: string | null) => v || <span style={{ color: 'var(--gray-400)' }}>-</span>,
    },
    {
      title: '输入',
      dataIndex: 'tool_input',
      key: 'tool_input',
      ellipsis: true,
      render: (input: string) => (
        <Paragraph
          ellipsis={{ rows: 1 }}
          copyable={{ text: input }}
          style={{ margin: 0, maxWidth: 200, fontSize: 12 }}
        >
          {input}
        </Paragraph>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (s: number) =>
        s === 1 ? (
          <Tag color="success" style={{ borderRadius: 6 }}>成功</Tag>
        ) : (
          <Tag color="error" style={{ borderRadius: 6 }}>失败</Tag>
        ),
    },
    {
      title: '耗时',
      dataIndex: 'cost_time',
      key: 'cost_time',
      width: 80,
      render: (t: number) => (
        <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
          {(t * 1000).toFixed(0)}ms
        </span>
      ),
    },
    {
      title: '时间',
      dataIndex: 'create_time',
      key: 'create_time',
      width: 160,
      render: (t: string) => (
        <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>
          {dayjs(t).format('YYYY-MM-DD HH:mm:ss')}
        </span>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (_: any, record: TraceItem) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetail(record.id)}
          style={{ padding: 0 }}
        >
          详情
        </Button>
      ),
    },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <Title level={4}>链路追踪</Title>
        <Space>
          <Tag icon={<ApiOutlined />} color="processing" style={{ borderRadius: 8 }}>
            实时监控
          </Tag>
        </Space>
      </div>

      <Card className="admin-card" style={{ marginBottom: 20 }}>
        <Space wrap>
          <span style={{ fontSize: 13, color: 'var(--gray-600)' }}>工具筛选：</span>
          <Select
            allowClear
            placeholder="全部工具"
            style={{ width: 140 }}
            value={toolFilter}
            onChange={(v) => setToolFilter(v)}
            options={[
              { label: '全部工具', value: undefined },
              { label: 'knowledge', value: 'knowledge' },
              { label: 'search', value: 'search' },
              { label: 'browser', value: 'browser' },
              { label: 'time', value: 'time' },
              { label: 'code', value: 'code' },
              { label: 'file', value: 'file' },
            ]}
          />
          <span style={{ fontSize: 13, color: 'var(--gray-600)' }}>状态：</span>
          <Select
            allowClear
            placeholder="全部状态"
            style={{ width: 120 }}
            value={statusFilter}
            onChange={(v) => setStatusFilter(v)}
            options={[
              { label: '全部状态', value: undefined },
              { label: '成功', value: 1 },
              { label: '失败', value: 0 },
            ]}
          />
          <Button icon={<ReloadOutlined />} onClick={() => loadTraces(page)}>
            刷新
          </Button>
        </Space>
      </Card>

      <Card className="admin-card" bodyStyle={{ padding: 0 }}>
        <Table
          dataSource={traces}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            onChange: loadTraces,
            showTotal: (t) => `共 ${t} 条记录`,
            style: { paddingRight: 16 },
          }}
          size="small"
        />
      </Card>

      <Modal
        title={
          <span>
            <ApiOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
            调用详情
          </span>
        }
        open={detailOpen}
        onCancel={() => setDetailOpen(false)}
        footer={null}
        width={720}
        className="fade-in"
      >
        {detailLoading ? (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Spin />
          </div>
        ) : detail ? (
          <>
            <Descriptions
              column={2}
              size="small"
              style={{ marginBottom: 20 }}
              labelStyle={{ color: 'var(--gray-500)', fontWeight: 500 }}
            >
              <Descriptions.Item label="工具名称">
                <Tag color="blue" style={{ borderRadius: 6 }}>{detail.tool_name}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {detail.status === 1 ? (
                  <Tag color="success" style={{ borderRadius: 6 }}>成功</Tag>
                ) : (
                  <Tag color="error" style={{ borderRadius: 6 }}>失败</Tag>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="耗时">
                <span style={{ fontFamily: 'monospace' }}>{(detail.cost_time * 1000).toFixed(0)} ms</span>
              </Descriptions.Item>
              <Descriptions.Item label="会话">{detail.session_name || '-'}</Descriptions.Item>
              <Descriptions.Item label="执行时间">
                {dayjs(detail.create_time).format('YYYY-MM-DD HH:mm:ss')}
              </Descriptions.Item>
              <Descriptions.Item label="错误信息">
                {detail.error_msg || <span style={{ color: 'var(--gray-400)' }}>无</span>}
              </Descriptions.Item>
            </Descriptions>

            {detail.user_message && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)', marginBottom: 6 }}>
                  用户消息
                </div>
                <div
                  style={{
                    background: 'var(--gray-50)',
                    borderRadius: 10,
                    padding: '10px 14px',
                    fontSize: 13,
                    color: 'var(--gray-700)',
                  }}
                >
                  {detail.user_message}
                </div>
              </div>
            )}

            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)', marginBottom: 6 }}>
                工具输入
              </div>
              <pre
                style={{
                  background: '#1e293b',
                  color: '#e2e8f0',
                  borderRadius: 10,
                  padding: '12px 16px',
                  fontSize: 12,
                  lineHeight: 1.6,
                  overflow: 'auto',
                  maxHeight: 300,
                  margin: 0,
                }}
              >
                {(() => {
                  try {
                    return JSON.stringify(JSON.parse(detail.tool_input), null, 2);
                  } catch {
                    return detail.tool_input;
                  }
                })()}
              </pre>
            </div>

            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)', marginBottom: 6 }}>
                工具输出
              </div>
              <Paragraph
                ellipsis={{ rows: 6, expandable: true, symbol: '展开全部' }}
                style={{
                  background: 'var(--gray-50)',
                  borderRadius: 10,
                  padding: '10px 14px',
                  fontSize: 13,
                  margin: 0,
                }}
              >
                {detail.tool_result || <span style={{ color: 'var(--gray-400)' }}>（空）</span>}
              </Paragraph>
            </div>

            {detail.assistant_message && (
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)', marginBottom: 6 }}>
                  AI 回复
                </div>
                <Paragraph
                  ellipsis={{ rows: 5, expandable: true, symbol: '展开全部' }}
                  style={{
                    background: 'var(--gray-50)',
                    borderRadius: 10,
                    padding: '10px 14px',
                    fontSize: 13,
                    margin: 0,
                  }}
                >
                  {detail.assistant_message}
                </Paragraph>
              </div>
            )}
          </>
        ) : (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--gray-400)' }}>
            加载失败
          </div>
        )}
      </Modal>
    </div>
  );
}
