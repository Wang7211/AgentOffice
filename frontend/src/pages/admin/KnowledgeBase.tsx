import { useEffect, useState } from 'react';
import {
  Card,
  Table,
  Button,
  Upload,
  Modal,
  Typography,
  Tag,
  message,
  Space,
  Popconfirm,
  Drawer,
  List,
  Empty,
  Tooltip,
  Row,
  Col,
} from 'antd';
import {
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  FileTextOutlined,
  InboxOutlined,
  FileOutlined,
} from '@ant-design/icons';
import {
  listKnowledgeFiles,
  deleteKnowledgeFile,
  getFileChunks,
  uploadKnowledge,
} from '../../api/admin';
import type { KnowledgeFileItem, ChunkItem } from '../../api/admin';
import dayjs from 'dayjs';

const { Title, Paragraph } = Typography;

const { Dragger } = Upload;

export default function KnowledgeBase() {
  const [files, setFiles] = useState<KnowledgeFileItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [chunkDrawerOpen, setChunkDrawerOpen] = useState(false);
  const [chunkFile, setChunkFile] = useState<{ id: number; name: string } | null>(null);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunksTotal, setChunksTotal] = useState(0);
  const [chunksLoading, setChunksLoading] = useState(false);

  const loadFiles = async (p: number) => {
    setLoading(true);
    try {
      const result = await listKnowledgeFiles(p, 20);
      setFiles(result.items);
      setTotal(result.total);
      setPage(result.page);
    } catch {
      message.error('加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFiles(1);
  }, []);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const result = await uploadKnowledge(file);
      message.success(`上传成功，共 ${result.chunk_count} 个文本分片`);
      loadFiles(1);
    } catch (err: any) {
      message.error(err.response?.data?.message || err.message || '上传失败');
    } finally {
      setUploading(false);
    }
    return false;
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteKnowledgeFile(id);
      message.success('文件已删除');
      loadFiles(page);
    } catch {
      message.error('删除失败');
    }
  };

  const handleViewChunks = async (file: KnowledgeFileItem) => {
    setChunkFile({ id: file.id, name: file.file_name });
    setChunkDrawerOpen(true);
    setChunksLoading(true);
    try {
      const result = await getFileChunks(file.id, 1, 50);
      setChunks(result.items);
      setChunksTotal(result.total);
    } catch {
      message.error('加载分片失败');
    } finally {
      setChunksLoading(false);
    }
  };

  const columns = [
    {
      title: '文件名',
      dataIndex: 'file_name',
      key: 'file_name',
      ellipsis: true,
      render: (name: string) => (
        <span>
          <FileOutlined style={{ marginRight: 8, color: 'var(--gray-400)' }} />
          {name}
        </span>
      ),
    },
    {
      title: '类型',
      dataIndex: 'file_suffix',
      key: 'file_suffix',
      width: 80,
      render: (s: string) => <Tag>{s.toUpperCase()}</Tag>,
    },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      width: 100,
      render: (size: number) =>
        size > 1024
          ? `${(size / 1024).toFixed(1)} MB`
          : `${size} KB`,
    },
    {
      title: '分片',
      dataIndex: 'chunk_count',
      key: 'chunk_count',
      width: 80,
      render: (n: number) => (
        <span style={{ fontWeight: 600, color: 'var(--primary)' }}>{n}</span>
      ),
    },
    {
      title: '上传时间',
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
      width: 150,
      render: (_: any, record: KnowledgeFileItem) => (
        <Space>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewChunks(record)}
            style={{ padding: 0 }}
          >
            分片
          </Button>
          <Popconfirm
            title="确定删除此文件？"
            description="删除后相关分片也将不可用"
            okText="删除"
            cancelText="取消"
            onConfirm={() => handleDelete(record.id)}
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
        <Title level={4}>知识库管理</Title>
        <Upload
          beforeUpload={handleUpload}
          showUploadList={false}
          accept=".pdf,.txt,.docx"
          disabled={uploading}
        >
          <Button
            type="primary"
            icon={<UploadOutlined />}
            loading={uploading}
            style={{ fontWeight: 600 }}
          >
            上传文件
          </Button>
        </Upload>
      </div>

      <Row gutter={[20, 20]}>
        <Col xs={24} lg={6}>
          <Dragger
            beforeUpload={handleUpload}
            showUploadList={false}
            accept=".pdf,.txt,.docx"
            disabled={uploading}
            style={{
              background: 'var(--gray-50)',
              border: '2px dashed var(--gray-200)',
              borderRadius: 14,
              padding: '20px 0',
            }}
          >
            <p style={{ fontSize: 14, color: 'var(--gray-400)' }}>
              <InboxOutlined style={{ fontSize: 36, color: 'var(--primary)', display: 'block', marginBottom: 12 }} />
              点击或拖拽文件到此区域上传
            </p>
            <p style={{ fontSize: 12, color: 'var(--gray-400)' }}>支持 PDF、TXT、DOCX 格式</p>
          </Dragger>
        </Col>
        <Col xs={24} lg={18}>
          <Card className="admin-card" bodyStyle={{ padding: 0 }}>
            <Table
              dataSource={files}
              columns={columns}
              rowKey="id"
              loading={loading}
              pagination={{
                current: page,
                total,
                pageSize: 20,
                onChange: loadFiles,
                showTotal: (t) => `共 ${t} 个文件`,
                style: { paddingRight: 16 },
              }}
              locale={{
                emptyText: (
                  <Empty
                    description="暂无知识库文件"
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                  />
                ),
              }}
            />
          </Card>
        </Col>
      </Row>

      <Drawer
        title={
          <span>
            <FileTextOutlined style={{ marginRight: 8, color: 'var(--primary)' }} />
            分片预览 — {chunkFile?.name}
            <Tag style={{ marginLeft: 8 }}>{chunksTotal} 个分片</Tag>
          </span>
        }
        open={chunkDrawerOpen}
        onClose={() => {
          setChunkDrawerOpen(false);
          setChunkFile(null);
        }}
        width={640}
      >
        <List
          loading={chunksLoading}
          dataSource={chunks}
          locale={{ emptyText: <Empty description="暂无分片" /> }}
          renderItem={(chunk) => (
            <List.Item style={{ padding: 0, marginBottom: 12, border: 'none' }}>
              <Card
                size="small"
                className="admin-card"
                title={
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-500)' }}>
                    分片 #{chunk.chunk_index + 1}
                  </span>
                }
                extra={
                  <Tag style={{ fontSize: 10 }}>
                    {chunk.vector_id?.slice(0, 8)}...
                  </Tag>
                }
                style={{ width: '100%' }}
              >
                <Paragraph
                  ellipsis={{ rows: 5, expandable: true, symbol: '展开全文' }}
                  style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 13, color: 'var(--gray-600)' }}
                >
                  {chunk.chunk_text}
                </Paragraph>
              </Card>
            </List.Item>
          )}
        />
      </Drawer>
    </div>
  );
}
