import { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Descriptions,
  Drawer,
  Input,
  message,
  Select,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd';
import {
  ApiOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  ReloadOutlined,
  SearchOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { getTraceDetail, listTraces } from '../../api/admin';
import type { TraceDetail, TraceItem } from '../../api/admin';

const { Paragraph } = Typography;

export default function TraceView() {
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [toolFilter, setToolFilter] = useState<string | undefined>(undefined);
  const [statusFilter, setStatusFilter] = useState<number | undefined>(undefined);
  const [keyword, setKeyword] = useState('');

  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detail, setDetail] = useState<TraceDetail | null>(null);

  const loadTraces = async (nextPage: number) => {
    setLoading(true);
    try {
      const result = await listTraces(nextPage, 20, toolFilter, statusFilter);
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

  const visibleTraces = useMemo(() => {
    const q = keyword.trim().toLowerCase();
    if (!q) return traces;
    return traces.filter((trace) =>
      [trace.id, trace.tool_name, trace.session_name, trace.tool_input]
        .join(' ')
        .toLowerCase()
        .includes(q),
    );
  }, [keyword, traces]);

  const successCount = traces.filter((trace) => trace.status === 1).length;
  const failureCount = traces.filter((trace) => trace.status !== 1).length;
  const avgCost = traces.length
    ? traces.reduce((sum, trace) => sum + trace.cost_time, 0) / traces.length
    : 0;
  const p95Cost = traces.length
    ? Math.max(...traces.map((trace) => trace.cost_time))
    : 0;

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
      title: 'Trace Name',
      dataIndex: 'tool_name',
      key: 'tool_name',
      render: (name: string) => <strong>{name}</strong>,
    },
    {
      title: 'Trace Id',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id: number) => <span className="mono-id">{id}</span>,
    },
    {
      title: '会话 / Task ID',
      dataIndex: 'session_name',
      key: 'session_name',
      ellipsis: true,
      render: (name: string | null, record: TraceItem) => (
        <div className="trace-session-cell">
          <span>{name || '未命名会话'}</span>
          <small>{record.chat_record_id}</small>
        </div>
      ),
    },
    {
      title: '耗时',
      dataIndex: 'cost_time',
      key: 'cost_time',
      width: 110,
      render: (cost: number) => <strong>{cost.toFixed(2)}s</strong>,
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
      title: '执行时间',
      dataIndex: 'create_time',
      key: 'create_time',
      width: 170,
      render: (time: string) => dayjs(time).format('YYYY/MM/DD HH:mm:ss'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 130,
      render: (_: any, record: TraceItem) => (
        <Button icon={<EyeOutlined />} onClick={() => handleViewDetail(record.id)}>
          查看链路
        </Button>
      ),
    },
  ];

  return (
    <div className="admin-page trace-page fade-in">
      <div className="admin-breadcrumb">首页 / 链路追踪</div>
      <div className="admin-page-toolbar">
        <div>
          <h1>链路追踪</h1>
          <p>检索工具运行记录，定位慢节点与失败节点</p>
        </div>
        <div className="toolbar-actions">
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索 Trace Id / 工具 / 会话"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            className="toolbar-search wide"
            allowClear
          />
          <Button type="primary" icon={<SearchOutlined />} onClick={() => loadTraces(1)}>
            查询
          </Button>
          <Button icon={<ReloadOutlined />} onClick={() => loadTraces(page)}>
            刷新
          </Button>
        </div>
      </div>

      <div className="trace-stats">
        <div className="summary-tile">
          <span className="summary-icon green"><ThunderboltOutlined /></span>
          <div>
            <small>成功 / 失败 / 运行中</small>
            <strong>{successCount} / {failureCount} / 0</strong>
          </div>
        </div>
        <div className="summary-tile">
          <span className="summary-icon cyan"><ApiOutlined /></span>
          <div>
            <small>成功率</small>
            <strong>{traces.length ? Math.round((successCount / traces.length) * 100) : 0}%</strong>
          </div>
        </div>
        <div className="summary-tile">
          <span className="summary-icon violet"><ClockCircleOutlined /></span>
          <div>
            <small>平均耗时</small>
            <strong>{avgCost.toFixed(2)} s</strong>
          </div>
        </div>
        <div className="summary-tile">
          <span className="summary-icon amber"><ClockCircleOutlined /></span>
          <div>
            <small>P95 耗时</small>
            <strong>{p95Cost.toFixed(2)} s</strong>
          </div>
        </div>
      </div>

      <section className="ops-panel trace-filter-panel">
        <div className="inline-filter">
          <span>工具筛选</span>
          <Select
            allowClear
            placeholder="全部工具"
            value={toolFilter}
            onChange={(value) => setToolFilter(value)}
            options={[
              { label: 'knowledge', value: 'knowledge' },
              { label: 'search', value: 'search' },
              { label: 'browser', value: 'browser' },
              { label: 'time', value: 'time' },
              { label: 'code', value: 'code' },
              { label: 'file', value: 'file' },
            ]}
          />
        </div>
        <div className="inline-filter">
          <span>状态</span>
          <Select
            allowClear
            placeholder="全部状态"
            value={statusFilter}
            onChange={(value) => setStatusFilter(value)}
            options={[
              { label: '成功', value: 1 },
              { label: '失败', value: 0 },
            ]}
          />
        </div>
      </section>

      <section className="ops-panel table-panel">
        <div className="panel-title stacked">
          <span>运行列表</span>
          <small>按时间倒序查看运行记录，点击任意记录进入详情页分析慢节点与失败节点</small>
        </div>
        <Table
          dataSource={visibleTraces}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            onChange: loadTraces,
            showTotal: (count) => `第 ${page} 页，共 ${count} 条`,
          }}
        />
      </section>

      <Drawer
        title={
          <span className="drawer-title">
            <ApiOutlined />
            链路详情
            {detail && (
              <Tag className={detail.status === 1 ? 'status-pill success' : 'status-pill danger'}>
                {detail.status === 1 ? 'SUCCESS' : 'FAILED'}
              </Tag>
            )}
          </span>
        }
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        width={920}
      >
        {detailLoading ? (
          <div className="admin-loading compact">
            <Spin />
          </div>
        ) : detail ? (
          <div className="trace-detail">
            <div className="detail-meta-strip">
              <div><ClockCircleOutlined /> <strong>{detail.cost_time.toFixed(2)}s</strong><span>总耗时</span></div>
              <div><ApiOutlined /> <strong>{detail.tool_name}</strong><span>工具</span></div>
              <div><ThunderboltOutlined /> <strong>{detail.id}</strong><span>Trace ID</span></div>
            </div>

            <Descriptions column={2} size="small" className="detail-descriptions">
              <Descriptions.Item label="会话">{detail.session_name || '-'}</Descriptions.Item>
              <Descriptions.Item label="执行时间">
                {dayjs(detail.create_time).format('YYYY/MM/DD HH:mm:ss')}
              </Descriptions.Item>
              <Descriptions.Item label="错误信息" span={2}>
                {detail.error_msg || '无'}
              </Descriptions.Item>
            </Descriptions>

            <section className="trace-timeline">
              <header>
                <strong>执行时序</strong>
                <span>窗口 {detail.cost_time.toFixed(2)}s</span>
              </header>
              {[
                ['user-message', 'USER_INPUT', detail.user_message || '用户消息未记录', 8, 18],
                ['tool-input', 'TOOL_INPUT', detail.tool_input, 24, 34],
                ['tool-output', 'TOOL_OUTPUT', detail.tool_result || '（空）', 60, 32],
                ['assistant-message', 'ASSISTANT', detail.assistant_message || 'AI 回复未记录', 84, 14],
              ].map(([key, label, text, left, width]) => (
                <div className="timeline-row" key={key}>
                  <div className="timeline-node">
                    <i />
                    <span>{label}</span>
                  </div>
                  <div className="timeline-track">
                    <span style={{ left: `${left}%`, width: `${width}%` }} />
                  </div>
                </div>
              ))}
            </section>

            <section className="trace-payloads">
              <div>
                <strong>工具输入</strong>
                <pre>
                  {(() => {
                    try {
                      return JSON.stringify(JSON.parse(detail.tool_input), null, 2);
                    } catch {
                      return detail.tool_input;
                    }
                  })()}
                </pre>
              </div>
              <div>
                <strong>工具输出</strong>
                <Paragraph ellipsis={{ rows: 8, expandable: true, symbol: '展开全部' }}>
                  {detail.tool_result || '（空）'}
                </Paragraph>
              </div>
            </section>
          </div>
        ) : (
          <div className="admin-empty-state">加载失败</div>
        )}
      </Drawer>
    </div>
  );
}
