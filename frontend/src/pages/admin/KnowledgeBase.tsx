import { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Drawer,
  Empty,
  Input,
  List,
  message,
  Popconfirm,
  Table,
  Tag,
  Typography,
  Upload,
} from 'antd';
import {
  DeleteOutlined,
  EyeOutlined,
  FileOutlined,
  FileTextOutlined,
  InboxOutlined,
  ReloadOutlined,
  SearchOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import {
  deleteKnowledgeFile,
  getFileChunks,
  listKnowledgeFiles,
  uploadKnowledge,
} from '../../api/admin';
import type { ChunkItem, KnowledgeFileItem } from '../../api/admin';

const { Paragraph } = Typography;
const { Dragger } = Upload;

const formatSize = (size: number) => {
  if (size > 1024) return `${(size / 1024).toFixed(1)} MB`;
  return `${size} KB`;
};

export default function KnowledgeBase() {
  const [files, setFiles] = useState<KnowledgeFileItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [keyword, setKeyword] = useState('');

  const [chunkDrawerOpen, setChunkDrawerOpen] = useState(false);
  const [chunkFile, setChunkFile] = useState<{ id: number; name: string } | null>(null);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunksTotal, setChunksTotal] = useState(0);
  const [chunksLoading, setChunksLoading] = useState(false);

  const filteredFiles = useMemo(() => {
    const q = keyword.trim().toLowerCase();
    if (!q) return files;
    return files.filter((file) => file.file_name.toLowerCase().includes(q));
  }, [files, keyword]);

  const stats = useMemo(() => {
    const chunkCount = files.reduce((sum, file) => sum + (file.chunk_count || 0), 0);
    const suffixCount = new Set(files.map((file) => file.file_suffix)).size;
    return [
      { label: '知识库文件', value: files.length, icon: <InboxOutlined />, tag: '全部' },
      { label: '文本分片', value: chunkCount, icon: <FileTextOutlined />, tag: '全部' },
      { label: '文件类型', value: suffixCount, icon: <FileOutlined />, tag: '自动识别' },
      { label: '当前页记录', value: filteredFiles.length, icon: <SearchOutlined />, tag: '筛选后' },
    ];
  }, [files, filteredFiles.length]);

  const loadFiles = async (nextPage: number) => {
    setLoading(true);
    try {
      const result = await listKnowledgeFiles(nextPage, 20);
      setFiles(result.items);
      setTotal(result.total);
      setPage(result.page);
    } catch {
      message.error('加载知识库文件失败');
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
      message.success(`上传成功，共生成 ${result.chunk_count} 个文本分片`);
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
      title: '名称',
      dataIndex: 'file_name',
      key: 'file_name',
      ellipsis: true,
      render: (name: string) => (
        <span className="name-cell">
          <FileOutlined />
          <strong>{name}</strong>
        </span>
      ),
    },
    {
      title: '类型',
      dataIndex: 'file_suffix',
      key: 'file_suffix',
      width: 100,
      render: (suffix: string) => <Tag className="soft-tag">{suffix.toUpperCase()}</Tag>,
    },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      width: 110,
      render: formatSize,
    },
    {
      title: '分片',
      dataIndex: 'chunk_count',
      key: 'chunk_count',
      width: 100,
      render: (count: number) => <strong className="numeric-accent">{count}</strong>,
    },
    {
      title: '上传时间',
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
      width: 170,
      render: (_: any, record: KnowledgeFileItem) => (
        <div className="table-actions">
          <Button icon={<EyeOutlined />} onClick={() => handleViewChunks(record)}>
            分片
          </Button>
          <Popconfirm
            title="删除此文件？"
            description="删除后相关文本分片也将不可用。"
            okText="删除"
            cancelText="取消"
            onConfirm={() => handleDelete(record.id)}
          >
            <Button danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </div>
      ),
    },
  ];

  return (
    <div className="admin-page fade-in">
      <div className="admin-breadcrumb">首页 / 知识库管理</div>
      <div className="admin-page-toolbar">
        <div>
          <h1>知识库管理</h1>
          <p>管理企业文档、文本分片与检索素材</p>
        </div>
        <div className="toolbar-actions">
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索知识库文件"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            allowClear
            className="toolbar-search"
          />
          <Button icon={<ReloadOutlined />} onClick={() => loadFiles(page)}>
            刷新
          </Button>
          <Upload
            beforeUpload={handleUpload}
            showUploadList={false}
            accept=".pdf,.txt,.docx"
            disabled={uploading}
          >
            <Button type="primary" icon={<UploadOutlined />} loading={uploading}>
              上传文档
            </Button>
          </Upload>
        </div>
      </div>

      <div className="stat-strip">
        {stats.map((item) => (
          <div className="summary-tile" key={item.label}>
            <span className="summary-icon">{item.icon}</span>
            <div>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </div>
            <Tag>{item.tag}</Tag>
          </div>
        ))}
      </div>

      <section className="ops-panel kb-workspace">
        <Dragger
          beforeUpload={handleUpload}
          showUploadList={false}
          accept=".pdf,.txt,.docx"
          disabled={uploading}
          className="kb-dropzone"
        >
          <InboxOutlined />
          <strong>拖拽文档到此处上传</strong>
          <span>支持 PDF、TXT、DOCX 格式</span>
        </Dragger>

        <Table
          dataSource={filteredFiles}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            current: page,
            total,
            pageSize: 20,
            onChange: loadFiles,
            showTotal: (count) => `共 ${count} 个文件`,
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
      </section>

      <Drawer
        title={
          <span className="drawer-title">
            <FileTextOutlined />
            分片预览 / {chunkFile?.name}
            <Tag>{chunksTotal} 个分片</Tag>
          </span>
        }
        open={chunkDrawerOpen}
        onClose={() => {
          setChunkDrawerOpen(false);
          setChunkFile(null);
        }}
        width={720}
      >
        <List
          loading={chunksLoading}
          dataSource={chunks}
          locale={{ emptyText: <Empty description="暂无分片" /> }}
          renderItem={(chunk) => (
            <List.Item className="chunk-item">
              <div className="chunk-card">
                <header>
                  <strong>分片 #{chunk.chunk_index + 1}</strong>
                  <Tag>{chunk.vector_id?.slice(0, 10)}...</Tag>
                </header>
                <Paragraph
                  ellipsis={{ rows: 5, expandable: true, symbol: '展开全文' }}
                >
                  {chunk.chunk_text}
                </Paragraph>
              </div>
            </List.Item>
          )}
        />
      </Drawer>
    </div>
  );
}
