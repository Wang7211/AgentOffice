import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Input,
  Button,
  Typography,
  Popconfirm,
  message,
  Empty,
  Tooltip,
  Dropdown,
  Tag,
} from 'antd';
import {
  SendOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  SearchOutlined,
  LogoutOutlined,
  SettingOutlined,
  RobotOutlined,
  MessageOutlined,
  HistoryOutlined,
  MoreOutlined,
  FileSearchOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useChatStore } from '../stores/chatStore';
import {
  listSessions,
  getChatHistory,
  renameSession,
  deleteSession as deleteSessionApi,
  streamChat,
} from '../api/chat';
import type { ChatSession, ChatRecord, Citation } from '../api/chat';

const { Text } = Typography;
const PENDING_SESSION_PREFIX = '__pending_session__';

const quickPrompts = [
  {
    title: '知识库检索',
    desc: '检索报销制度并整理审批条件',
    prompt: '检索知识库里的报销流程，并按适用场景整理成表格。',
    icon: <DatabaseOutlined />,
  },
  {
    title: '文档分析',
    desc: '提取风险点与待确认事项',
    prompt: '总结这份合同的关键风险点，并列出需要业务方确认的问题。',
    icon: <FileSearchOutlined />,
  },
  {
    title: '工作生成',
    desc: '输出可直接复用的办公材料',
    prompt: '生成一份项目周会纪要模板，包含进展、风险、决策和下周计划。',
    icon: <ThunderboltOutlined />,
  },
  {
    title: '系统排查',
    desc: '分析工具调用与执行链路',
    prompt: '分析最近工具调用失败的可能原因，并给出排查优先级。',
    icon: <SafetyCertificateOutlined />,
  },
];

export default function Chat() {
  const navigate = useNavigate();
  const { user, logout, isAdmin } = useAuthStore();
  const {
    sessions,
    currentSessionId,
    messages,
    streaming,
    streamingContent,
    citations,
    setCitations,
    setSessions,
    setCurrentSession,
    setMessages,
    appendMessage,
    addStreamingChunk,
    clearStreaming,
    finishStreaming,
    setStreaming,
    removeSession,
    updateSessionName,
  } = useChatStore();

  const [inputValue, setInputValue] = useState('');
  const [searchKeyword, setSearchKeyword] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [sending, setSending] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const editInputRef = useRef<any>(null);
  const charQueueRef = useRef<string[]>([]);
  const typingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamCompletedRef = useRef(false);
  const pendingHistorySessionIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const currentSession = sessions.find((s) => s.session_id === currentSessionId);
  const assistantCount = messages.filter((m) => m.role === 'assistant').length;

  const loadSessions = useCallback(async () => {
    if (!user) return;
    setLoadingSessions(true);
    try {
      const data = await listSessions(searchKeyword || undefined);
      setSessions(data);
    } catch {
      // Session list is non-blocking for the current chat surface.
    } finally {
      setLoadingSessions(false);
    }
  }, [user, searchKeyword, setSessions]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (
      !currentSessionId ||
      streaming ||
      currentSessionId.startsWith(PENDING_SESSION_PREFIX)
    ) {
      return;
    }
    getChatHistory(currentSessionId)
      .then((records) => setMessages(currentSessionId, records))
      .catch(() => {});
  }, [currentSessionId, setMessages, streaming]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  useEffect(() => {
    if (streaming) {
      typingTimerRef.current = setInterval(() => {
        const queue = charQueueRef.current;
        if (queue.length > 0) {
          addStreamingChunk(queue.shift()!);
          return;
        }

        if (!streamCompletedRef.current) return;

        const sessionId = pendingHistorySessionIdRef.current;
        pendingHistorySessionIdRef.current = null;
        streamCompletedRef.current = false;
        if (typingTimerRef.current) {
          clearInterval(typingTimerRef.current);
          typingTimerRef.current = null;
        }

        if (sessionId) {
          getChatHistory(sessionId)
            .then((records) => {
              finishStreaming(sessionId, records);
              loadSessions();
            })
            .catch(() => {
              finishStreaming();
              loadSessions();
            });
        } else {
          finishStreaming();
        }
      }, 24);
    } else if (typingTimerRef.current) {
      clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }

    return () => {
      if (typingTimerRef.current) {
        clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    };
  }, [streaming, addStreamingChunk, finishStreaming, loadSessions]);

  const resetComposer = () => {
    clearStreaming();
    setCitations([]);
    setInputValue('');
  };

  const handleNewSession = () => {
    if (sending) return;
    setCurrentSession(null);
    resetComposer();
  };

  const handleSelectSession = (session: ChatSession) => {
    if (sending) return;
    setCurrentSession(session.session_id);
    clearStreaming();
    setCitations([]);
  };

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text || sending || !user) return;

    setInputValue('');
    setSending(true);
    clearStreaming();
    setCitations([]);
    charQueueRef.current = [];
    pendingHistorySessionIdRef.current = null;
    setStreaming(true);

    const localSessionId =
      currentSessionId || `${PENDING_SESSION_PREFIX}${Date.now()}`;
    const tempUserMsg: ChatRecord = {
      id: Date.now(),
      session_id: localSessionId,
      role: 'user',
      content: text,
      create_time: new Date().toISOString(),
    };

    let targetSessionId = currentSessionId;
    if (targetSessionId) {
      appendMessage(targetSessionId, tempUserMsg);
    } else {
      setCurrentSession(localSessionId);
      appendMessage(localSessionId, tempUserMsg);
    }

    let sessionCreated = false;

    abortRef.current = streamChat(
      text,
      targetSessionId,
      (content) => {
        for (const char of content) {
          charQueueRef.current.push(char);
        }
      },
      (meta) => {
        if (meta.session_id && !sessionCreated) {
          sessionCreated = true;
          targetSessionId = meta.session_id;
          tempUserMsg.session_id = meta.session_id;
          if (currentSessionId !== meta.session_id) {
            setCurrentSession(meta.session_id);
            appendMessage(meta.session_id, tempUserMsg);
            loadSessions();
          }
        }
        if (meta.citations && Array.isArray(meta.citations)) {
          setCitations(meta.citations as Citation[]);
        }
      },
      (data) => {
        setSending(false);
        if (data?.citations && Array.isArray(data.citations)) {
          setCitations(data.citations as Citation[]);
        }
        pendingHistorySessionIdRef.current = targetSessionId;
        streamCompletedRef.current = true;
      },
      (err) => {
        setSending(false);
        streamCompletedRef.current = true;
        message.error(`发送失败：${err.message}`);
      },
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleRename = async (sessionId: string) => {
    const nextName = editName.trim();
    if (!nextName) {
      setEditingId(null);
      return;
    }

    try {
      await renameSession(sessionId, nextName);
      updateSessionName(sessionId, nextName);
      setEditingId(null);
      message.success('会话已重命名');
    } catch {
      message.error('重命名失败');
    }
  };

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSessionApi(sessionId);
      removeSession(sessionId);
      message.success('会话已删除');
    } catch {
      message.error('删除失败');
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const startEditing = (session: ChatSession) => {
    setEditingId(session.session_id);
    setEditName(session.session_name || '');
    setTimeout(() => editInputRef.current?.focus(), 100);
  };

  const getInitial = (name?: string | null) => {
    return (name || 'U')[0].toUpperCase();
  };

  const renderComposer = (compact = false) => (
    <div className="chat-input-area">
      <div className="chat-input-wrapper">
        <Input.TextArea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入问题，Enter 发送，Shift + Enter 换行"
          autoSize={{ minRows: compact ? 1 : 2, maxRows: 5 }}
          disabled={sending}
          variant="borderless"
        />
        <div className="chat-input-actions">
          <Text type="secondary" className="composer-state">
            {sending ? '正在回复...' : '企业知识库与工具链已就绪'}
          </Text>
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={sending}
            disabled={!inputValue.trim()}
            shape="circle"
            className="send-button"
          />
        </div>
      </div>
    </div>
  );

  return (
    <div className="chat-layout">
      <aside className="chat-sidebar">
        <div className="chat-sidebar-header">
          <div className="chat-sidebar-brand">
            <span className="brand-icon">
              <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
            </span>
            <span className="brand-copy">
              <strong>AgentOffice</strong>
              <span>AI 办公工作台</span>
            </span>
          </div>
          <Tag className="role-tag">{isAdmin() ? 'Admin' : 'User'}</Tag>
        </div>

        <div className="chat-quick-start">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleNewSession}
            block
            className="new-chat-button"
          >
            新建对话
          </Button>
          {isAdmin() && (
            <Button
              icon={<SettingOutlined />}
              onClick={() => navigate('/admin')}
              block
              className="quick-admin-button"
            >
              管理后台
            </Button>
          )}
        </div>

        <div className="chat-search">
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索会话"
            value={searchKeyword}
            onChange={(e) => setSearchKeyword(e.target.value)}
            allowClear
          />
        </div>

        <div className="chat-sessions">
          {sessions.length > 0 && (
            <div className="session-group-label">最近会话</div>
          )}
          {sessions.length === 0 && !loadingSessions && (
            <div className="session-empty">
              <Empty
                image={<MessageOutlined />}
                description="暂无会话"
              />
            </div>
          )}

          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={`session-item${currentSessionId === session.session_id ? ' active' : ''}`}
              onClick={() => handleSelectSession(session)}
            >
              {editingId === session.session_id ? (
                <Input
                  ref={editInputRef}
                  size="small"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onPressEnter={() => handleRename(session.session_id)}
                  onBlur={() => handleRename(session.session_id)}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <>
                  <div className="session-title">
                    <HistoryOutlined />
                    <Text ellipsis className="session-name">
                      {session.session_name || '新对话'}
                    </Text>
                  </div>
                  <div className="session-actions" onClick={(e) => e.stopPropagation()}>
                    <Tooltip title="重命名">
                      <Button
                        type="text"
                        size="small"
                        icon={<EditOutlined />}
                        onClick={() => startEditing(session)}
                      />
                    </Tooltip>
                    <Popconfirm
                      title="删除此会话？"
                      description="删除后不可恢复。"
                      onConfirm={() => handleDelete(session.session_id)}
                      okText="删除"
                      cancelText="取消"
                    >
                      <Tooltip title="删除">
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                        />
                      </Tooltip>
                    </Popconfirm>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

        <div className="chat-sidebar-footer">
          <div className="user-info">
            <div className="user-avatar">
              {user?.avatar ? (
                <img src={user.avatar} className="avatar-img" alt="avatar" />
              ) : (
                getInitial(user?.nickname || user?.username)
              )}
            </div>
            <div className="user-meta">
              <div className="user-name">{user?.nickname || user?.username}</div>
              <div className="user-role">
                {user?.role === 'admin' ? '管理员' : '成员'}
              </div>
            </div>
          </div>
          <Dropdown
            menu={{
              items: [
                ...(isAdmin()
                  ? [{
                      key: 'admin',
                      icon: <SettingOutlined />,
                      label: '后台管理',
                      onClick: () => navigate('/admin'),
                    }]
                  : []),
                {
                  key: 'logout',
                  icon: <LogoutOutlined />,
                  label: '退出登录',
                  danger: true,
                  onClick: handleLogout,
                },
              ],
            }}
            placement="topRight"
            trigger={['click']}
          >
            <Button type="text" icon={<MoreOutlined />} className="icon-button" />
          </Dropdown>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div>
            <div className="chat-header-title">
              <span className="chat-header-dot" />
              {currentSession?.session_name || '新对话'}
            </div>
            <div className="chat-header-subtitle">
              {assistantCount > 0 ? `${assistantCount} 条助手回复` : '准备开始一次新的办公会话'}
            </div>
          </div>
          <div className="chat-header-tools">
            <Tag>RAG</Tag>
            <Tag>Tools</Tag>
            <Tag>Trace</Tag>
          </div>
        </header>

        {currentSessionId || messages.length > 0 ? (
          <>
            <section className="chat-messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`message-row ${msg.role}`}>
                  <div className={`message-avatar ${msg.role}`}>
                    {msg.role === 'user' ? (
                      user?.avatar ? (
                        <img src={user.avatar} className="avatar-img" alt="avatar" />
                      ) : (
                        getInitial(user?.nickname || user?.username)
                      )
                    ) : (
                      <RobotOutlined />
                    )}
                  </div>
                  <div className="message-stack">
                    <div className="message-bubble">{msg.content}</div>
                    <div className="message-time">
                      {new Date(msg.create_time).toLocaleTimeString('zh-CN', {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  </div>
                </div>
              ))}

              {streaming && (
                <div className="message-row assistant">
                  <div className="message-avatar assistant">
                    <RobotOutlined />
                  </div>
                  <div className="message-stack">
                    <div className="message-bubble streaming">
                      {streamingContent || (
                        <div className="typing-indicator">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {!streaming && citations.length > 0 && (
                <div className="citation-strip">
                  <DatabaseOutlined />
                  <span>引用知识库文件：</span>
                  {citations.map((c, i) => (
                    <strong key={c.file_name}>
                      {i > 0 ? '、' : ''}
                      {c.file_name}
                    </strong>
                  ))}
                </div>
              )}

              <div ref={messagesEndRef} />
            </section>
            {renderComposer(true)}
          </>
        ) : (
          <>
            <section className="chat-empty-home">
              <div className="empty-command-panel">
                <div className="empty-logo">
                  <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
                </div>
                <h1>今天要处理什么？</h1>
                <div className="prompt-grid">
                  {quickPrompts.map((item) => (
                    <button
                      key={item.title}
                      className="prompt-card"
                      onClick={() => setInputValue(item.prompt)}
                    >
                      <span className="prompt-icon">{item.icon}</span>
                      <span>
                        <strong>{item.title}</strong>
                        <small>{item.desc}</small>
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </section>
            {renderComposer()}
          </>
        )}
      </main>
    </div>
  );
}
