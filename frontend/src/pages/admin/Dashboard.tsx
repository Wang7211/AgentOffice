import { useEffect, useState } from 'react';
import { Button, Progress, Spin, Table, Tag } from 'antd';
import {
  ApiOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  MessageOutlined,
  ReloadOutlined,
  RiseOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { getDashboard } from '../../api/admin';
import type { DashboardStats } from '../../api/admin';

const metricCards = [
  {
    key: 'user_count',
    label: '活跃用户',
    icon: <TeamOutlined />,
    tone: 'blue',
    hint: '当前账号规模',
  },
  {
    key: 'session_count',
    label: '会话数',
    icon: <MessageOutlined />,
    tone: 'violet',
    hint: '累计会话',
  },
  {
    key: 'tool_call_count',
    label: '消息数',
    icon: <ApiOutlined />,
    tone: 'amber',
    hint: '工具调用量',
  },
  {
    key: 'today_chat_count',
    label: '会话深度',
    icon: <BarChartOutlined />,
    tone: 'cyan',
    hint: '今日对话',
  },
];

export default function Dashboard() {
  const [data, setData] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const result = await getDashboard();
      setData(result);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="admin-loading">
        <Spin size="large" />
      </div>
    );
  }

  if (!data) {
    return <div className="admin-empty-state">加载失败</div>;
  }

  const successCount =
    data.tool_success_count ?? data.recent_tools.filter((item) => item.status === 1).length;
  const failureCount =
    data.tool_failure_count ?? data.recent_tools.filter((item) => item.status !== 1).length;
  const totalToolCount = successCount + failureCount;
  const successRate = totalToolCount
    ? Math.round((successCount / totalToolCount) * 100)
    : 0;
  const avgMs = data.avg_tool_cost !== undefined
    ? Math.round(data.avg_tool_cost * 1000)
    : data.recent_tools.length
    ? Math.round(
        data.recent_tools.reduce((sum, item) => sum + item.cost_time * 1000, 0) /
          data.recent_tools.length,
      )
    : 0;
  const p95Ms = data.p95_tool_cost !== undefined
    ? Math.round(data.p95_tool_cost * 1000)
    : data.recent_tools.length
    ? Math.round(Math.max(...data.recent_tools.map((item) => item.cost_time * 1000)))
    : 0;
  const activity = data.hourly_activity ?? [];
  const maxActivity = Math.max(
    1,
    ...activity.map((item) => item.chat_count + item.tool_count),
  );

  const toolColumns = [
    {
      title: 'Trace Name',
      dataIndex: 'tool_name',
      key: 'tool_name',
      render: (name: string) => <strong>{name}</strong>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 110,
      render: (status: number) =>
        status === 1 ? (
          <Tag className="status-pill success">SUCCESS</Tag>
        ) : (
          <Tag className="status-pill danger">FAILED</Tag>
        ),
    },
    {
      title: '耗时',
      dataIndex: 'cost_time',
      key: 'cost_time',
      width: 110,
      render: (cost: number) => <strong>{(cost * 1000).toFixed(0)}ms</strong>,
    },
    {
      title: '执行时间',
      dataIndex: 'create_time',
      key: 'create_time',
      width: 160,
      render: (time: string) => (
        <span className="table-muted">{dayjs(time).format('YYYY/MM/DD HH:mm')}</span>
      ),
    },
  ];

  const sessionColumns = [
    {
      title: '会话名称',
      dataIndex: 'session_name',
      key: 'session_name',
      ellipsis: true,
      render: (name: string) => <strong>{name || '未命名会话'}</strong>,
    },
    {
      title: '创建时间',
      dataIndex: 'create_time',
      key: 'create_time',
      width: 160,
      render: (time: string) => (
        <span className="table-muted">{dayjs(time).format('YYYY/MM/DD HH:mm')}</span>
      ),
    },
  ];

  return (
    <div className="admin-page admin-dashboard fade-in">
      <div className="admin-breadcrumb">首页 / Dashboard</div>
      <div className="admin-page-toolbar">
        <div>
          <h1>Dashboard</h1>
          <p>会话、知识库与工具调用的运营概览</p>
        </div>
        <div className="toolbar-actions">
          <div className="range-switch">
            <button className="active">24h</button>
            <button>7d</button>
            <button>30d</button>
          </div>
          <span className="live-dot">实时</span>
          <Button icon={<ReloadOutlined />} onClick={loadData} />
        </div>
      </div>

      <div className="dashboard-layout">
        <div className="dashboard-main">
          <section className="ops-panel">
            <div className="panel-title">核心指标</div>
            <div className="metric-grid">
              {metricCards.map((card) => {
                const value = (data as any)[card.key] ?? 0;
                return (
                  <div className="metric-tile" key={card.key}>
                    <div>
                      <strong>{value}</strong>
                      <span>{card.label}</span>
                      <small>{card.hint}</small>
                    </div>
                    <span className={`metric-icon ${card.tone}`}>{card.icon}</span>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="ops-panel">
            <div className="panel-title">流量概览</div>
            <div className="wide-chart">
              <div className="chart-grid-lines" />
              {activity.length > 0 ? (
                <div className="activity-chart">
                  {activity.map((item) => (
                    <div className="activity-column" key={item.label}>
                      <span
                        className="activity-bar chat"
                        style={{
                          height: `${Math.max(3, (item.chat_count / maxActivity) * 120)}px`,
                        }}
                      />
                      <span
                        className="activity-bar tool"
                        style={{
                          height: `${Math.max(3, (item.tool_count / maxActivity) * 120)}px`,
                        }}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="chart-line blue" />
              )}
              <div className="chart-axis">
                {(activity.length > 0
                  ? activity.filter((_, index) => index % 4 === 0 || index === activity.length - 1)
                  : [{ label: '22:00' }, { label: '02:00' }, { label: '06:00' }, { label: '10:00' }, { label: '14:00' }, { label: '18:00' }]
                ).map((item) => (
                  <span key={item.label}>{item.label}</span>
                ))}
              </div>
            </div>
          </section>

          <section className="ops-panel">
            <div className="panel-title">趋势分析</div>
            <div className="trend-grid">
              <div className="mini-chart">
                <header>
                  <span>会话趋势</span>
                  <small>单位：次</small>
                </header>
                <div className="mini-plot green" />
              </div>
              <div className="mini-chart">
                <header>
                  <span>活跃用户趋势</span>
                  <small>单位：人</small>
                </header>
                <div className="mini-plot violet" />
              </div>
              <div className="mini-chart">
                <header>
                  <span>响应时间趋势</span>
                  <small>单位：毫秒</small>
                </header>
                <div className="threshold-lines">
                  <i className="warn" />
                  <i className="ok" />
                  <span className="baseline" />
                </div>
              </div>
              <div className="mini-chart">
                <header>
                  <span>质量趋势</span>
                  <small>单位：%</small>
                </header>
                <div className="threshold-lines quality">
                  <i className="warn" />
                  <i className="danger" />
                  <span className="baseline" />
                </div>
              </div>
            </div>
          </section>

          <section className="ops-panel table-panel">
            <div className="panel-title">最近会话</div>
            <Table
              dataSource={data.recent_sessions}
              columns={sessionColumns}
              rowKey="session_id"
              pagination={false}
              size="middle"
            />
          </section>

          <section className="ops-panel table-panel">
            <div className="panel-title">最近工具调用</div>
            <Table
              dataSource={data.recent_tools}
              columns={toolColumns}
              rowKey="id"
              pagination={false}
              size="middle"
            />
          </section>
        </div>

        <aside className="dashboard-rail">
          <section className="ops-panel rail-panel">
            <div className="panel-title split">
              <span>AI 性能</span>
              <Tag>暂无数据</Tag>
            </div>
            <Progress
              type="circle"
              percent={successRate}
              size={96}
              strokeColor={successRate >= 90 ? '#16a34a' : '#ef4444'}
              format={(value) => `${value}%`}
            />
            <div className="rail-metrics">
              <div>
                <ClockCircleOutlined />
                <span>平均响应</span>
                <strong>{avgMs}ms</strong>
              </div>
              <div>
                <ClockCircleOutlined />
                <span>P95 响应</span>
                <strong>{p95Ms}ms</strong>
              </div>
            </div>
          </section>

          <section className="ops-panel rail-panel">
            <div className="panel-title split">
              <span>质量快照</span>
              <small>滚动 24h</small>
            </div>
            <div className="quality-bars">
              <div>
                <span style={{ height: `${Math.max(4, 100 - successRate)}%` }} />
                <strong>{100 - successRate}%</strong>
                <small>错误率</small>
              </div>
              <div>
                <span style={{ height: `${Math.min(96, Math.max(8, avgMs / 20))}%` }} />
                <strong>{avgMs}ms</strong>
                <small>平均耗时</small>
              </div>
              <div>
                <span style={{ height: `${Math.min(96, Math.max(8, data.file_count * 8))}%` }} />
                <strong>{data.file_count}</strong>
                <small>文件数</small>
              </div>
            </div>
          </section>

          <section className="ops-panel rail-panel">
            <div className="panel-title">运营洞察</div>
            <div className="insight-note">
              <Tag color="blue" icon={<RiseOutlined />}>趋势</Tag>
              <strong>
                {data.recent_sessions.length > 0 ? '最近会话保持活跃' : '暂无会话数据'}
              </strong>
              <p>
                当前共有 {data.file_count} 个知识库文件、{data.chunk_count} 个文本分片。
                工具调用成功率为 {successRate}%。
              </p>
            </div>
          </section>

          <section className="ops-panel rail-panel compact">
            <div className="rail-row">
              <DatabaseOutlined />
              <span>知识库文件</span>
              <strong>{data.file_count}</strong>
            </div>
            <div className="rail-row">
              <FileTextOutlined />
              <span>文本分片</span>
              <strong>{data.chunk_count}</strong>
            </div>
            <div className="rail-row">
              <MessageOutlined />
              <span>今日对话</span>
              <strong>{data.today_chat_count}</strong>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
