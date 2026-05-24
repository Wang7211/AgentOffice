import { useEffect, useState } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Typography,
  Spin,
  Tag,
} from 'antd';
import {
  UserOutlined,
  MessageOutlined,
  FileOutlined,
  ApiOutlined,
  TeamOutlined,
  RiseOutlined,
  ClusterOutlined,
} from '@ant-design/icons';
import { getDashboard } from '../../api/admin';
import type { DashboardStats } from '../../api/admin';
import dayjs from 'dayjs';

const { Title } = Typography;

const statCards = [
  {
    key: 'user_count',
    title: '用户总数',
    icon: <TeamOutlined />,
    color: '#4f6ef7',
    bg: 'rgba(79,110,247,0.08)',
  },
  {
    key: 'session_count',
    title: '会话总数',
    icon: <MessageOutlined />,
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.08)',
  },
  {
    key: 'file_count',
    title: '知识库文件',
    icon: <FileOutlined />,
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.08)',
  },
  {
    key: 'tool_call_count',
    title: '工具调用',
    icon: <ApiOutlined />,
    color: '#8b5cf6',
    bg: 'rgba(139,92,246,0.08)',
  },
];

const iconMap: Record<string, React.ReactNode> = {
  user_count: <TeamOutlined />,
  session_count: <MessageOutlined />,
  file_count: <FileOutlined />,
  tool_call_count: <ApiOutlined />,
};

const colorMap: Record<string, string> = {
  user_count: '#4f6ef7',
  session_count: '#22c55e',
  file_count: '#f59e0b',
  tool_call_count: '#8b5cf6',
};

const bgMap: Record<string, string> = {
  user_count: 'rgba(79,110,247,0.08)',
  session_count: 'rgba(34,197,94,0.08)',
  file_count: 'rgba(245,158,11,0.08)',
  tool_call_count: 'rgba(139,92,246,0.08)',
};

export default function Dashboard() {
  const [data, setData] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboard()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 120 }}>
        <Spin size="large" />
      </div>
    );
  }
  if (!data) return <div>加载失败</div>;

  const toolColumns = [
    { title: '工具', dataIndex: 'tool_name', key: 'tool_name' },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
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
      render: (t: number) => `${(t * 1000).toFixed(0)} ms`,
    },
    {
      title: '时间',
      dataIndex: 'create_time',
      key: 'create_time',
      render: (t: string) => (
        <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>
          {dayjs(t).format('MM-DD HH:mm')}
        </span>
      ),
    },
  ];

  const sessionColumns = [
    {
      title: '会话名称',
      dataIndex: 'session_name',
      key: 'session_name',
      ellipsis: true,
    },
    {
      title: '创建时间',
      dataIndex: 'create_time',
      key: 'create_time',
      render: (t: string) => (
        <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>
          {dayjs(t).format('MM-DD HH:mm')}
        </span>
      ),
    },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <Title level={4}>Dashboard</Title>
        <Tag icon={<RiseOutlined />} color="processing" style={{ borderRadius: 8, fontSize: 12 }}>
          实时更新
        </Tag>
      </div>

      <Row gutter={[20, 20]}>
        {statCards.map((card) => {
          const value = (data as any)[card.key] ?? 0;
          return (
            <Col xs={24} sm={12} md={6} key={card.key}>
              <Card className="stat-card" style={{ border: 'none' }}>
                <div
                  className="stat-icon"
                  style={{ background: card.bg, color: card.color }}
                >
                  {card.icon}
                </div>
                <Statistic
                  title={card.title}
                  value={value}
                  valueStyle={{ color: card.color, fontSize: 30 }}
                />
              </Card>
            </Col>
          );
        })}
      </Row>

      <Row gutter={[20, 20]} style={{ marginTop: 24 }}>
        <Col xs={24} md={8}>
          <Card
            className="admin-card"
            title={
              <span style={{ fontSize: 14, fontWeight: 600 }}>
                <ClusterOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
                知识库概况
              </span>
            }
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>总文件数</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--gray-800)' }}>{data.file_count}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>总文本分片</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--gray-800)' }}>{data.chunk_count}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>今日对话</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--success)' }}>{data.today_chat_count}</div>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} md={16}>
          <Card
            className="admin-card"
            title={
              <span style={{ fontSize: 14, fontWeight: 600 }}>
                <MessageOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
                最近会话
              </span>
            }
          >
            <Table
              dataSource={data.recent_sessions}
              columns={sessionColumns}
              rowKey="session_id"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>
      </Row>

      <Card
        className="admin-card"
        title={
          <span style={{ fontSize: 14, fontWeight: 600 }}>
            <ApiOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
            最近工具调用
          </span>
        }
        style={{ marginTop: 20 }}
      >
        <Table
          dataSource={data.recent_tools}
          columns={toolColumns}
          rowKey="id"
          size="small"
          pagination={false}
        />
      </Card>
    </div>
  );
}
